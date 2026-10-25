# Incremental 2D orthomosaic from an image sequence

An end-to-end, CPU only pipeline that builds a seamless 2D orthomosaic from an
ordered sequence of still images.  The processing procedure follows the paper

> **Real-Time 2D Orthomosaic Mapping from UAV Video via Feature-Based Image
> Registration**, Hwang *et al.*, *Appl. Sci.* **2026**, *16*, 2133
> (`Papers/applsci-16-02133.pdf`)

restricted to still images: **no video handling is implemented at this stage.**

```
read image -> SIFT -> match -> RANSAC homography -> accumulate -> warp onto the
canvas -> blend (overwrite/feather) -> write the canvas
```

## Quick start

```bash
python main.py --dataset datasets/UAVPistachio/images_nadir_RGB \
               --output-root outputs --run-name pistachio
```

Every update of the canvas is written into `outputs/<run-name>/`:

| file | content |
| --- | --- |
| `canvas_0000_x0.1.jpg` … | the mosaic scaled down by 10 %, after every frame |
| `mosaic_final.jpg` | the final mosaic at its native resolution |
| `timings.csv` | per step timings |
| `report.json`, `config.json` | run report and the configuration used |

The native resolution canvas is written only at the end of the run; the
encoding of the full size mosaic dominates the runtime of the pipeline, so
intermediate full-size writes are off by default.

### Memory of the mosaic

The canvas stores the mosaic (three bytes per pixel) and its validity mask
(one byte per pixel), so it needs about **four bytes per pixel**.  A mosaic
that covers a long flight path therefore grows quickly: lower
`--canvas-scale` to trade resolution for memory.  The geometry is not
affected by it, the homographies are always estimated on the original
resolution of the frames.

## The pipeline

### 1. Dataset (`mosaic/dataset.py`)

The frames are discovered with a glob pattern and ordered by the number in
their file name (`DJI_0002.JPG` before `DJI_0010.JPG`), so the sequence follows
the acquisition order of the flight.

### 2. Feature detection (`mosaic/features.py`)

Every frame is converted to grayscale and processed with **SIFT**.  The paper
bounds the number of retained keypoints per frame (`N_max`) by keeping the
strongest detector responses; OpenCV exposes that budget directly through the
`nfeatures` argument of `cv2.SIFT_create`, so the pruning happens inside the
detector instead of being emulated afterwards.

Defaults follow Table 1 of the paper: `nOctaveLayers = 3`,
`contrastThreshold = 0.04`, `edgeThreshold = 10`, `sigma = 1.6`, `N_max = 500`.
Only SIFT is used (ORB, KAZE and BRISK were dropped per the paper's
requirements).

### 3. Matching (`mosaic/matching.py`)

The descriptors of the previous and of the incoming frame are matched with
**FLANN** (KD-tree, 5 trees, 50 checks) and filtered with **Lowe's ratio
test** (`0.7`).  Only the ratio test is applied; the mutual check of the
original implementation was dropped.

### 4. Robust homography (`mosaic/registration.py`)

The inter-frame geometry is modelled with a **planar homography**.  It is
estimated with **RANSAC** (reprojection threshold `5 px`) and re-estimated on
the inlier set only, which improves the stability of the model.

A frame is only accepted when the inlier support satisfies a minimum
acceptance criterion (`--min-inliers`).  Otherwise the update is rejected so
that an unreliable transform is never propagated through the accumulation.  A
grossly wrong model (implausible scale change, degenerate matrix) is rejected
as well.

### 5. Registration against the mosaic (`mosaic/store.py`)

The paper registers every frame against its **predecessor** and accumulates the
inter-frame transforms (Eq. 6):

```
G_0 = I ,   G_{k+1} = G_k @ H_{k->k+1}
```

The transform of frame `k` therefore contains the sum of all the errors made
on the way, which shows up as a growing misalignment over a long sequence. The
paper acknowledges this ("drift may accumulate as sequences length
increases"), and it is what makes the seams appear.

This implementation registers every frame against the **accumulated mosaic**.
The SIFT features of the mosaic are kept in a *store*, expressed in mosaic
coordinates.  Three sources of correspondences are used, each for a different
job:

| source | what it carries | role |
| --- | --- | --- |
| the previous frame | the **relative motion** | the prediction the grounding is validated against |
| the frames captured just before (`--grounding-window`) | the **relative motion**, pooled over several frames | the placement: re-estimated instead of chained, which cancels part of the error of the individual steps |
| the frames captured further back | the **drift** of the part of the flight the frame belongs to | the grounding: the only evidence tying that part of the flight to the rest of the mosaic |

The third group is what keeps the mosaic together.  A flight that comes back
over ground it covered earlier drifts slowly with respect to the parts that
were mapped before it, and only the correspondences with those frames carry
that drift.  They are fitted on their own, as a separate homography, and the
result is accepted only when it is supported, precise (the reprojection error
stays small) and stays within reach of the prediction.  Otherwise the frame
is placed from the recent frames instead: the relative motion is reliable,
only its absolute placement drifts.

The store selects its candidates by **spatial proximity**: the features that
are physically close to where the frame lands.  A flight that comes back over
ground it covered earlier puts the frame close to frames that were captured a
long time ago, and those are the candidates that tie it to the rest of the
mosaic.

The store keeps **one descriptor per ground location** - a new observation
replaces the previous one. Duplicated descriptors would break the ratio test,
because the second best candidate of a match would be a copy of the best one.
The store only holds keypoints and descriptors, never pixels, so it stays in
the order of a few megabytes.

`--registration frame` restores the behaviour of the paper (predecessor +
accumulation) for comparison.

### 6. Canvas and blending (`mosaic/canvas.py`, `mosaic/blending.py`)

The four corners of the incoming frame are transformed with `G_k`, the axis
aligned bounding box is computed and the canvas is padded when the frame
reaches outside of the current extent, while the existing mosaic content is
preserved.

The canvas keeps two things apart:

* the **logical extent**, which is the tight bounding box of the mosaic and
  defines where the orthomosaic actually ends,
* the **allocated buffer**, which is slightly larger so that it does not have
  to be reallocated on every step.  The slack lives outside of the logical
  extent and is never part of the mosaic.

The warped frame is then merged into the mosaic.  Two strategies are
available:

| `--blend` | what it does |
| --- | --- |
| `overwrite` | **the method of the paper** (Section 3.6): the pixels of the latest warped frame replace the existing ones in the overlap regions.  Cheap, and it never ghosts. |
| `feather` | a linear ramp around the border of the warped frame.  Hides the seam of the overwrite at the cost of mixing the two images where they overlap, which is only visible when they are misregistered. |

The weights handed to the blenders are built with the distance transform
scheme of Brown & Lowe: every pixel is attributed to the image that owns it
and the attribution changes smoothly across the overlap.

> The paper only implements the overwrite baseline.  Multi-band (Laplacian
> pyramid) blending was deliberately dropped: it mixes the high frequencies
> of both images locally, so on a scene with parallax -- trees, buildings,
> anything that is not a plane -- the two views of the same object survive
> side by side and the mosaic shows ghosting.

### 7. Output and runtime accounting (`mosaic/output.py`, `mosaic/pipeline.py`)

Each stage is timed and reported:

```
[12/86] DJI_0013.JPG
   a) SIFT on DJI_0013.JPG took 1.944 s (501 keypoints)
   b) matching took 0.009 s (134 matches)
   c) homography took 0.007 s (81/134 inliers, ratio 0.60)
   d) stitching took 2.765 s (canvas 5951x4025, 14206263 px updated)
   e) saving the canvas took 1.204 s
```

## Command line options

```
--dataset PATH           folder with the images
--pattern GLOB           glob pattern of the images            (*.JPG)
--start / --end INT      inclusive bounds of the sequence

--registration {mosaic,frame}
                         register a frame against the accumulated mosaic (no
                         drift) or against its predecessor and accumulate the
                         transforms                         (mosaic)
--min-inliers INT        minimum inlier support to accept a frame    (20)
--store-min-distance F   minimum distance between two features of the
                         mosaic store, in pixels                    (40.0)
--query-margin F         margin around the predicted position used to
                         select the store features, in pixels      (1500)
--grounding-window INT   number of recent frames whose matches describe
                         the relative motion                (6)
--store-max-features INT maximum number of features in the store    (50000)
--no-relink              drop a frame instead of re-linking it

--blend {overwrite,feather}
                         blending strategy (overwrite = the paper) (overwrite)
--blend-ramp INT         width of the transition band in pixels      (32)

--canvas-scale FLOAT     resolution of the mosaic              (1.0)

--output-root PATH       root folder of the runs               (outputs)
--run-name NAME          name of the run folder
--no-preview             do not write the downscaled canvas
--preview-scale FLOAT    scale of the downscaled canvas         (0.1)
--jpeg-quality INT       quality of the JPEGs                  (90)
--no-timings             do not write the timings CSV
--save-debug             write the match visualisations
```

## Layout

```
main.py                 command line entry point
mosaic/
  config.py             every tunable of the pipeline
  dataset.py            ordered discovery / loading of the frames
  features.py           SIFT with a bounded keypoint budget
  matching.py           FLANN matching and Lowe's ratio test
  registration.py       RANSAC homography estimation and frame acceptance
  store.py              the incremental feature store of the mosaic
  transforms.py         accumulation of the homographies
  canvas.py             growable mosaic canvas, warping and placement
  blending.py           feather / overwrite blending
  pipeline.py           the incremental stitching loop
  output.py             artefact writing
```

## Benchmarking on AWS Graviton5 (COOL vs pip OpenCV)

`COOL-Benchmark/` runs this exact pipeline on Graviton5 (`m9g`/`c9g`) and
compares the COOL Marketplace OpenCV build against pip's
`opencv-python-headless`, on the same instance, OS and Python version.

* Orchestrator + web UI: `python COOL-Benchmark/pipeline-benchmark/pipeline-orchestrator.py`
* Modes: DIY only, COOL only, Compare (DIY first, then COOL)
* Metrics: per-stage chunk and merge timings, canvas size, content %, OpenCV
  version/path
* No credentials are stored in the repository; configuration lives in a
  git-ignored `.env` and `config-marketplace.json`

Full instructions: [COOL-Benchmark/README.md](COOL-Benchmark/README.md).

On this SIFT-based mosaic the two builds are effectively at parity; the harness
reports that honestly rather than assuming an OpenCV speedup.
