# Incremental 2D orthomosaic from an image sequence

An end-to-end, CPU only pipeline that builds a seamless 2D orthomosaic from an
ordered sequence of still images.  The processing procedure follows the paper

> **Real-Time 2D Orthomosaic Mapping from UAV Video via Feature-Based Image
> Registration**, Hwang *et al.*, *Appl. Sci.* **2026**, *16*, 2133

restricted to still images.

```
read image -> SIFT -> match -> RANSAC homography -> register
           -> warp onto the canvas -> blend -> write canvas
```

## Quick start

Requires Python 3.9+ (tested on 3.12 and 3.14) and the two dependencies in
`requirements.txt` (`opencv-python-headless`, `numpy`).

The repository ships a small 86-frame dataset (half-resolution UAVPistachio)
under `COOL-Benchmark/assets/`, so the pipeline can be tried without any
download:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python main.py run --dataset COOL-Benchmark/assets --pattern '*.jpg' \
                   --output-root outputs --run-name pistachio
```

The original full-resolution dataset path used during development
(`datasets/UAVPistachio/images_nadir_RGB`) is not distributed with the
repository; point `--dataset` at wherever it lives.  The top-level form
(`python main.py --dataset ...`) is kept as an alias for `main.py run`.

A long sequence is normally processed in overlapping chunks and merged
afterwards (see *Chunked runs* below):

```bash
python main.py run-chunks --dataset COOL-Benchmark/assets --pattern '*.jpg' \
                          --chunk-size 18 --output-root outputs --prefix pistachio
python main.py merge --inputs 'outputs/pistachio_*/mosaic_final.jpg' \
                     --out outputs/merged
```

Every update of the canvas is written into `outputs/<run-name>/`:

| file | content |
| --- | --- |
| `canvas_0000_x0.1.jpg` … | the mosaic scaled down by 10 %, after every frame |
| `mosaic_final.jpg` | the final mosaic at its native resolution |
| `mosaic_final_mask.png` | validity mask of the final mosaic (0 = no data) |
| `timings.csv` | per step timings |
| `report.json`, `config.json` | run report and the configuration used |

The native resolution canvas is written only at the end of the run; the
encoding of the full size mosaic dominates the runtime of the pipeline, so
intermediate full-size writes are off by default.

## Chunked runs and merging

A long sequence is normally split into overlapping chunks so that memory and
the cost of a single run stay bounded.  `run-chunks` names each folder from the
frame range (`pistachio_001_018`, `pistachio_018_035`, …) and passes one shared
frame between consecutive chunks, so the mosaics overlap.

`merge` then combines the chunk mosaics level by level: it merges pairs, keeps
at most one unpaired mosaic for the next level, and carries that mosaic forward
until a single one is left.  The validity of each mosaic is recovered from its
black background (flood fill from the border) or from the mask written by the
pipeline, and the merge is masked so empty canvas does not leak into the
result.

The merged result is `mosaic_merged.jpg` plus `mosaic_merged_mask.png`;
per-stage merge timings are written to `merge_timings.json`.

## Tests

```bash
python -m unittest discover -s tests
```

The tests cover case-insensitive frame discovery, sequence-index frame ids in
the store, the safe homogeneous divide, and the rejection of a featureless
frame (which used to raise instead of being skipped).

## Command line

```
python main.py run         # one sequence (also the default with no subcommand)
python main.py run-chunks  # split the sequence into overlapping chunks
python main.py merge       # merge the chunk mosaics
```

`main.py run --help`, `main.py run-chunks --help` and `main.py merge --help`
list the exact arguments.  The defaults:

```
--dataset PATH           folder with the images       (datasets/UAVPistachio/...)
--pattern GLOB           glob pattern of the images, case-insensitive (*.JPG)
--start / --end INT      inclusive bounds of the sequence

--registration {mosaic,frame}
                         register against the accumulated mosaic (no drift)
                         or against the predecessor and accumulate (mosaic)
--min-inliers INT        minimum inlier support to accept a frame    (20)
--store-min-distance F   lower bound of the store replacement radius, px (40)
--query-margin F         margin around the predicted position used to
                         select the store features, in pixels      (1500)
--grounding-window INT   number of recent frames whose matches describe
                         the relative motion                         (6)
--store-max-features INT maximum number of features in the store    (50000)
--no-relink              drop a frame instead of re-linking it

--blend {overwrite,feather}
                         blending strategy (overwrite = the paper) (overwrite)
--blend-ramp INT         width of the transition band in pixels      (32)

--canvas-scale FLOAT     resolution of the mosaic                   (1.0)
--max-canvas-pixels INT  abort above this canvas size (0 = no limit)  (0)

--output-root PATH       root folder of the runs               (outputs)
--run-name NAME          name of the run folder
--no-preview             do not write the downscaled canvas
--preview-scale FLOAT    scale of the downscaled canvas         (0.1)
--jpeg-quality INT       quality of the JPEGs                  (90)
--no-timings             do not write the timings CSV
--save-debug             write the match visualisations
```

`run-chunks` adds `--chunk-size` (17), `--prefix` (default: dataset folder
name) and `--extra` (extra arguments forwarded to `run`).  Consecutive chunks
share one frame; with 86 frames and a chunk size of 18 the run produces five
chunks and processes 90 frames (4 boundary frames are counted twice).

`merge` takes `--inputs` (files or glob patterns, in order), `--out`,
`--features`, `--detect-px`, `--min-inliers`, `--min-inlier-ratio`,
`--threshold` and `--quality`.  It merges pairs level by level, carrying at
most one unpaired mosaic to the next level, and writes `mosaic_merged.jpg`,
its mask and `merge_timings.json`.

## Layout

```
main.py                 command line entry point (run / run-chunks / merge)
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
  chunker.py            overlapping chunk runs
  merge.py              level-wise merge of chunk mosaics
  output.py             artefact writing
tests/
  test_core.py          regression tests (python -m unittest discover -s tests)
requirements.txt        opencv-python-headless, numpy
COOL-Benchmark/         AWS Graviton5 benchmark harness (see its README)
```

## Benchmarking on AWS Graviton5 (COOL vs pip OpenCV)

`COOL-Benchmark/` runs this exact pipeline on Graviton5 (`m9g`/`c9g`) and
compares the COOL Marketplace OpenCV build against pip's
`opencv-python-headless`, on the same instance and OS.

* Orchestrator + web UI:
  `python3 COOL-Benchmark/pipeline-benchmark/pipeline-orchestrator.py`
  (then open <http://localhost:8080/>)
* Modes: DIY only, COOL only, Compare (DIY first, then COOL).  DIY runs in a
  dedicated Python 3.12 venv on the COOL AMI, so the only variable left is the
  OpenCV build.
* Metrics: per-stage chunk and merge timings, canvas size, content %, OpenCV
  version/path
* Results are downloaded as previews; the full-resolution mosaics stay on the
  instance and are fetched on demand (the instance is kept alive until you
  terminate it)
* No credentials are stored in the repository; configuration lives in a
  git-ignored `.env` and `config-marketplace.json`

Full instructions: [COOL-Benchmark/README.md](COOL-Benchmark/README.md).

On this SIFT-based mosaic the two builds are effectively at parity; the harness
reports that honestly rather than assuming an OpenCV speedup.
