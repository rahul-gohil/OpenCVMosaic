"""The incremental stitching loop and its runtime accounting.

The loop implements the processing procedure of the paper for a still image
sequence:

1. the frames are read in order,
2. the features of the incoming frame are detected (SIFT, bounded keypoint
   budget),
3. the descriptors are matched against the previous frame,
4. a planar homography is estimated with RANSAC and refined on the inliers,
5. the homography is accumulated into the mosaic frame,
6. the frame is warped onto the canvas and blended into the mosaic,
7. the canvas is written to the output folder.

Every stage is timed and reported, so that the per step cost of the pipeline
is documented the way the paper does it.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .canvas import MosaicCanvas
from .config import MosaicConfig
from .dataset import ImageSequence
from .features import FeatureExtractor, FeatureSet
from .matching import DescriptorMatcher
from .output import RunOutput
from .registration import estimate_homography
from .store import MosaicFeatureStore
from .transforms import TransformChain, bounding_box, transformed_corners

#: How many of the most recently placed frames are kept as fall back
#: references for a frame that does not register against its predecessor.
MAX_PLACED_REFERENCES = 6


@dataclass
class StepResult:
    """Runtime and quality figures of a single stitching step."""

    index: int
    name: str
    load_time: float = 0.0
    sift_time: float = 0.0
    match_time: float = 0.0
    homography_time: float = 0.0
    stitch_time: float = 0.0
    save_time: float = 0.0
    total_time: float = 0.0
    keypoints: int = 0
    matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_error: float = 0.0
    accepted: bool = True
    reason: str = ""
    #: How the frame was placed: "mosaic" (recent population), "grounded"
    #: (older population) or "frame" (its predecessor).
    source: str = ""
    canvas_width: int = 0
    canvas_height: int = 0
    relinked: bool = False


@dataclass
class RunReport:
    """Summary of a complete run."""

    steps: List[StepResult] = field(default_factory=list)
    total_time: float = 0.0
    accepted: int = 0
    rejected: int = 0
    canvas_width: int = 0
    canvas_height: int = 0
    output_dir: str = ""

    @property
    def stage_totals(self) -> Dict[str, float]:
        """Total time spent in each stage of the pipeline."""
        return {
            "load": sum(s.load_time for s in self.steps),
            "sift": sum(s.sift_time for s in self.steps),
            "matching": sum(s.match_time for s in self.steps),
            "homography": sum(s.homography_time for s in self.steps),
            "stitching": sum(s.stitch_time for s in self.steps),
            "saving": sum(s.save_time for s in self.steps),
        }

    @property
    def mean_step_time(self) -> float:
        if not self.steps:
            return 0.0
        return self.total_time / len(self.steps)

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["stage_totals"] = self.stage_totals
        data["mean_step_time"] = self.mean_step_time
        return data


@dataclass
class _RegistrationOutcome:
    """Everything produced by registering a single frame."""

    registration: Any
    matches: list
    source: str
    match_time: float
    homography_time: float
    mosaic_attempt: Optional[np.ndarray]
    G: Optional[np.ndarray]
    G_predicted: Optional[np.ndarray]
    local: list
    grounding: list
    reference_index: int
    reference_features: FeatureSet
    reference_transform: Optional[np.ndarray]
    relinked: bool
    rejected: bool


class StitchingPipeline:
    """Builds a 2D orthomosaic from an ordered sequence of images."""

    def __init__(
        self,
        config: MosaicConfig,
        log: Callable[[str], None] = print,
        run_name: Optional[str] = None,
    ) -> None:
        self.config = config
        self.log = log
        self.run_name = run_name
        self.sequence = ImageSequence(
            config.dataset_dir,
            pattern=config.image_pattern,
            start=config.start,
            end=config.end,
        )
        self.extractor = FeatureExtractor(config.detector)
        self.matcher = DescriptorMatcher(config.matcher)
        self.canvas = MosaicCanvas(config.canvas, config.blending)
        self.chain = TransformChain()
        self.store = MosaicFeatureStore(config.store)
        self.output = RunOutput(config.output, run_name or "run")

    # ------------------------------------------------------------------ steps
    def _register(
        self,
        reference: FeatureSet,
        incoming: FeatureSet,
        match_time: float,
    ):
        """Match and register two frames, returning the result and timings.

        Lowe's ratio test is applied with the strictest threshold that still
        yields a usable registration.  The threshold has to be relaxed on
        repetitive scenes, where the second best candidate of a true match is a
        look-alike of the same texture: the strict threshold then throws away
        the correspondences that are right.  The escalation is driven by the
        registration itself, not by the number of matches - a lot of matches
        says nothing about how many of them are consistent.
        """
        start = time.perf_counter()
        knn = self.matcher.knn_match(reference.descriptors, incoming.descriptors)
        match_time += time.perf_counter() - start

        registration = None
        matches = []
        homography_time = 0.0
        for ratio in self.config.matcher.ratio_ladder:
            candidate_matches = self.matcher.ratio_filter(knn, ratio)
            if len(candidate_matches) < 4:
                continue
            start = time.perf_counter()
            candidate = estimate_homography(
                reference, incoming, candidate_matches, self.config.registration
            )
            homography_time += time.perf_counter() - start
            if candidate.ok:
                return candidate, match_time, homography_time, candidate_matches
            if registration is None or candidate.inliers > registration.inliers:
                registration = candidate
                matches = candidate_matches
        return registration, match_time, homography_time, matches

    def _shift(
        self,
        H: np.ndarray,
        predicted: Optional[np.ndarray],
        shape: Tuple[int, int],
    ) -> float:
        """Return how far a model moves the frame away from the prediction.

        A geometrically consistent model can still describe a completely
        different part of the mosaic: scenes with repeated structures (rows of
        identical plants, a regular grid of fences, ...) admit models that are
        perfectly self consistent but place the frame somewhere else
        entirely.  The grounding model is therefore only accepted when it
        stays within reach of the prediction.
        """
        if predicted is None:
            return 0.0
        height, width = shape[:2]
        corners = transformed_corners(H, width, height)
        reference = transformed_corners(predicted, width, height)
        return float(np.abs(corners - reference).max())

    def _write_debug(
        self,
        index: int,
        reference_image: Optional[np.ndarray],
        incoming_image: np.ndarray,
        reference: FeatureSet,
        incoming: FeatureSet,
        matches,
        registration,
    ) -> None:
        """Write a visualisation of the tentative and the verified matches."""
        if not self.config.output.save_debug:
            return
        if reference_image is None:
            return
        flags = registration.inlier_flags
        if flags is None:
            flags = np.ones(len(matches), bool)
        panel = cv2.drawMatches(
            reference_image,
            [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in reference.points],
            incoming_image,
            [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in incoming.points],
            matches,
            None,
            matchesMask=flags.astype(np.uint8).tolist(),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        self.output.write_debug(panel, f"matches_{index:04d}")

    def _log_header(self, report: RunReport) -> None:
        config = self.config
        self.log("=" * 72)
        self.log(f" dataset     : {config.dataset_dir}")
        self.log(f" frames      : {len(self.sequence)}")
        self.log(f" config      : {config.summary()}")
        self.log(f" output      : {self.output.directory}")
        self.log("=" * 72)
        self.output.write_config(config.to_dict())

    def _process_first_frame(self) -> Tuple[StepResult, np.ndarray, FeatureSet]:
        """Load frame 0, extract features, initialise canvas and transforms."""
        load_start = time.perf_counter()
        image = self.sequence.load(0)
        load_time = time.perf_counter() - load_start

        sift_start = time.perf_counter()
        features = self.extractor.extract(image)
        sift_time = time.perf_counter() - sift_start

        self.canvas.initialize(image)
        self.chain.seed(0)
        if self.config.registration.mode == "mosaic":
            self.store.update(features, self.chain.get(0))

        self.log(f"[ 1/{len(self.sequence)}] {self.sequence.name(0)}")
        self.log(
            f"   a) SIFT on {self.sequence.name(0)} took {sift_time:.3f} s "
            f" ({len(features)} keypoints)"
        )
        self.log(f"   b) matching took 0.000 s (reference frame)")
        self.log(f"   c) homography took 0.000 s (reference frame)")
        self.log(
            f"   d) stitching took 0.000 s "
            f"(canvas {self.canvas.size[0]}x{self.canvas.size[1]})"
        )

        save_start = time.perf_counter()
        self.output.write_canvas(self.canvas.snapshot(copy=False), 0)
        save_time = time.perf_counter() - save_start

        step = StepResult(
            index=0,
            name=self.sequence.name(0),
            load_time=load_time,
            sift_time=sift_time,
            save_time=save_time,
            keypoints=len(features),
            canvas_width=self.canvas.size[0],
            canvas_height=self.canvas.size[1],
        )
        return step, image, features

    def _register_frame(
        self,
        image: np.ndarray,
        features: FeatureSet,
        index: int,
        parent_index: int,
        parent_features: FeatureSet,
        parent_transform: Optional[np.ndarray],
        placed: List,
    ) -> _RegistrationOutcome:
        """Register a single frame against its references.

        Tries the predecessor first, then the most recently placed frames,
        and finally the mosaic-level grounding population.  Returns a fully
        populated _RegistrationOutcome.
        """
        config = self.config
        match_time = 0.0
        homography_time = 0.0
        registration = None
        matches = []
        local = []
        grounding = []
        source = "frame"

        # -- try predecessor and recently placed references ------------------
        references = [(parent_index, parent_features, self.chain.get(parent_index))]
        if config.relink_on_failure:
            references += [r for r in reversed(placed) if r[0] != parent_index]

        reference_index = parent_index
        reference_features = parent_features
        reference_transform = self.chain.get(parent_index)
        for candidate_index, candidate_features, candidate_transform in references:
            (
                registration,
                match_time,
                homography_time,
                matches,
            ) = self._register(candidate_features, features, match_time)
            if registration.ok:
                reference_index = candidate_index
                reference_features = candidate_features
                reference_transform = candidate_transform
                break
        else:
            # None of the references worked: try the chain against the
            # predecessor anyway, the frame may still be placeable.
            (
                registration,
                match_time,
                homography_time,
                matches,
            ) = self._register(parent_features, features, match_time)
            reference_index = parent_index
            reference_features = parent_features
            reference_transform = self.chain.get(parent_index)

        G_predicted = None
        if registration.ok and reference_transform is not None:
            G_predicted = reference_transform @ registration.H

        # -- mosaic-level grounding -----------------------------------------
        mosaic_attempt = None
        if config.registration.mode == "mosaic" and len(self.store) >= 4:
            h, w = image.shape[:2]
            anchor = G_predicted
            if anchor is None:
                anchor = self.chain.get(parent_index)
            if anchor is not None:
                corners = transformed_corners(anchor, w, h)
                x0, y0, x1, y1 = bounding_box(corners)
                margin = float(config.store.query_margin)
                reference = self.store.query(
                    x0 - margin, y0 - margin, x1 + margin, y1 + margin
                )
            else:
                reference = self.store.as_featureset()

            if len(reference) >= 4:
                knn = self.matcher.knn_match(
                    reference.descriptors, features.descriptors
                )

                window = int(config.registration.grounding_window)
                for ratio in config.matcher.ratio_ladder:
                    candidates = self.matcher.ratio_filter(knn, ratio)
                    if len(candidates) < 4:
                        continue
                    local = [
                        t
                        for t in candidates
                        if reference.frame_ids[t.queryIdx] >= index - window
                    ]
                    grounding = [
                        t
                        for t in candidates
                        if reference.frame_ids[t.queryIdx] < index - window
                    ]

                    local_attempt = estimate_homography(
                        reference, features, local, self.config.registration
                    )

                    if len(grounding) >= 4:
                        grounding_config = replace(
                            self.config.registration,
                            min_inliers=config.registration.grounding_min_inliers,
                            min_inlier_ratio=config.registration.grounding_min_inlier_ratio,
                        )
                        grounding_attempt = estimate_homography(
                            reference, features, grounding, grounding_config
                        )
                        if grounding_attempt.ok:
                            shift = self._shift(
                                grounding_attempt.H, G_predicted, image.shape
                            )
                            if (
                                shift <= config.registration.grounding_max_shift
                                * float(np.hypot(*image.shape[1:]))
                                and grounding_attempt.reprojection_error
                                <= config.registration.grounding_max_residual
                            ):
                                mosaic_attempt = grounding_attempt.H
                                source = "grounded"
                                matches = grounding
                                break
                            self.log(
                                "       (grounding model rejected: "
                                f"shift {shift:.0f} px, residual "
                                f"{grounding_attempt.reprojection_error:.2f} px)"
                            )

                    if mosaic_attempt is None and local_attempt.ok:
                        mosaic_attempt = local_attempt.H
                        source = "mosaic"
                        matches = local
                        break

        # -- compute G ------------------------------------------------------
        G = None
        relinked = reference_index != parent_index
        if mosaic_attempt is not None:
            G = self.chain.set(index, mosaic_attempt)
            matches = grounding if source == "grounded" else local
        elif registration.ok and reference_transform is not None:
            G = self.chain.set(index, reference_transform @ registration.H)

        rejected = (not registration.ok) and (mosaic_attempt is None)

        return _RegistrationOutcome(
            registration=registration,
            matches=matches,
            source=source,
            match_time=match_time,
            homography_time=homography_time,
            mosaic_attempt=mosaic_attempt,
            G=G,
            G_predicted=G_predicted,
            local=local,
            grounding=grounding,
            reference_index=reference_index,
            reference_features=reference_features,
            reference_transform=reference_transform,
            relinked=relinked,
            rejected=rejected,
        )

    def _handle_rejected_frame(
        self,
        index: int,
        name: str,
        outcome: _RegistrationOutcome,
        features: FeatureSet,
        sift_time: float,
        match_time: float,
        homography_time: float,
        load_time: float,
        step_start: float,
    ) -> StepResult:
        """Log and build a StepResult for a rejected frame."""
        self.log(f"[{index + 1}/{len(self.sequence)}] {name}")
        self.log(
            f"   a) SIFT on {name} took {sift_time:.3f} s "
            f"({len(features)} keypoints)"
        )
        self.log(
            f"   b) matching took {match_time:.3f} s "
            f"({len(outcome.matches)} matches)"
        )
        self.log(
            f"   c) homography took {homography_time:.3f} s "
            f"-> REJECTED ({outcome.registration.reason})"
        )
        self.log(f"   d) stitching took 0.000 s (frame skipped)")

        step_total = time.perf_counter() - step_start
        return StepResult(
            index=index,
            name=name,
            load_time=load_time,
            sift_time=sift_time,
            match_time=match_time,
            homography_time=homography_time,
            stitch_time=0.0,
            total_time=step_total,
            keypoints=len(features),
            matches=len(outcome.matches),
            inliers=outcome.registration.inliers,
            inlier_ratio=outcome.registration.inlier_ratio,
            reprojection_error=outcome.registration.reprojection_error,
            accepted=False,
            reason=outcome.registration.reason,
            source=outcome.source,
            canvas_width=self.canvas.size[0],
            canvas_height=self.canvas.size[1],
            relinked=outcome.relinked,
        )

    def _place_and_record(
        self,
        index: int,
        name: str,
        image: np.ndarray,
        parent_image: np.ndarray,
        features: FeatureSet,
        outcome: _RegistrationOutcome,
        sift_time: float,
        match_time: float,
        homography_time: float,
        load_time: float,
        step_start: float,
        placed: List,
    ) -> StepResult:
        """Place the frame on the canvas, save, log, and return a StepResult."""
        stitch_start = time.perf_counter()
        placement = self.canvas.place(image, outcome.G)
        stitch_time = time.perf_counter() - stitch_start

        step_total = time.perf_counter() - step_start
        self.log(f"[{index + 1}/{len(self.sequence)}] {name}")
        self.log(
            f"   a) SIFT on {name} took {sift_time:.3f} s "
            f"({len(features)} keypoints)"
        )
        self.log(
            f"   b) matching took {match_time:.3f} s "
            f"({len(outcome.matches)} matches)"
        )
        self.log(
            f"   c) homography took {homography_time:.3f} s "
            f"({outcome.source}, {outcome.registration.inliers}/{outcome.registration.matches} inliers, "
            f"ratio {outcome.registration.inlier_ratio:.2f}, "
            f"residual {outcome.registration.reprojection_error:.2f} px)"
        )
        self.log(
            f"   d) stitching took {stitch_time:.3f} s "
            f"(canvas {self.canvas.size[0]}x{self.canvas.size[1]}, "
            f"{placement.written} px updated, "
            f"misalignment {placement.misalignment:.2f}, "
            f"{self.canvas.footprint_mb:.0f} MB, "
            f"store {len(self.store)} features)"
        )

        save_start = time.perf_counter()
        self.output.write_canvas(self.canvas.snapshot(copy=False), index)
        save_time = time.perf_counter() - save_start

        self.log(f"   e) saving the canvas took {save_time:.3f} s")

        step_result = StepResult(
            index=index,
            name=name,
            load_time=load_time,
            sift_time=sift_time,
            match_time=match_time,
            homography_time=homography_time,
            stitch_time=stitch_time,
            save_time=save_time,
            total_time=step_total + save_time,
            keypoints=len(features),
            matches=outcome.registration.matches,
            inliers=outcome.registration.inliers,
            inlier_ratio=outcome.registration.inlier_ratio,
            reprojection_error=outcome.registration.reprojection_error,
            accepted=True,
            reason=outcome.registration.reason,
            source=outcome.source,
            canvas_width=self.canvas.size[0],
            canvas_height=self.canvas.size[1],
            relinked=outcome.relinked,
        )

        self._write_debug(
            index,
            parent_image,
            image,
            outcome.reference_features,
            features,
            outcome.matches,
            outcome.registration,
        )

        # Track placed frames for fallback references.
        if placement.written:
            placed.append((index, features, outcome.G))
            if len(placed) > MAX_PLACED_REFERENCES:
                del placed[0]

        return step_result

    def _update_state(
        self,
        index: int,
        features: FeatureSet,
        outcome: _RegistrationOutcome,
        placed: List,
    ) -> None:
        """Update parent tracking, feature store, and placed list."""
        config = self.config
        if config.registration.mode == "mosaic":
            self.store.update(features, outcome.G)

    def _finalize(self, report: RunReport) -> RunReport:
        """Write the final mosaic, compute totals, log the summary."""
        final = self.output.write_final(
            self.canvas.snapshot(copy=False), self.canvas.mask_snapshot()
        )
        report.total_time = sum(step.total_time for step in report.steps)
        report.canvas_width = self.canvas.size[0]
        report.canvas_height = self.canvas.size[1]

        totals = report.stage_totals
        self.log("=" * 72)
        self.log(" run summary")
        self.log(f"   frames processed : {len(report.steps)}")
        self.log(f"   accepted         : {report.accepted}")
        self.log(f"   rejected         : {report.rejected}")
        self.log(f"   total time       : {report.total_time:.3f} s")
        self.log(f"   mean per step    : {report.mean_step_time:.3f} s")
        self.log(
            "   stage totals     : "
            f"load {totals['load']:.2f} s | "
            f"sift {totals['sift']:.2f} s | "
            f"matching {totals['matching']:.2f} s | "
            f"homography {totals['homography']:.2f} s | "
            f"stitching {totals['stitching']:.2f} s | "
            f"saving {totals['saving']:.2f} s"
        )
        self.log(
            f"   final canvas     : {report.canvas_width} x {report.canvas_height} px"
        )
        if final is not None:
            self.log(f"   final mosaic     : {final}")
        self.log(f"   output folder    : {self.output.directory}")
        self.log("=" * 72)

        self.output.write_timings([asdict(s) for s in report.steps])
        self.output.write_report(report.to_dict())
        return report

    # -------------------------------------------------------------------- run
    def run(self) -> RunReport:
        """Run the whole pipeline and return the report of the run."""
        config = self.config
        report = RunReport(output_dir=str(self.output.directory))

        self._log_header(report)

        # ------------------------------------------------------------ frame 0
        first_step, first_image, first_features = self._process_first_frame()
        report.steps.append(first_step)

        # ------------------------------------------------- incremental updates
        parent_image = first_image
        parent_features = first_features
        parent_index = 0
        # The frames that were actually placed on the canvas, most recent last.
        placed: List = [(0, first_features, self.chain.get(0))]

        for index in range(1, len(self.sequence)):
            step_start = time.perf_counter()
            name = self.sequence.name(index)

            # -- (a) feature detection on the incoming frame -----------------
            load_start = time.perf_counter()
            image = self.sequence.load(index)
            load_time = time.perf_counter() - load_start

            sift_start = time.perf_counter()
            features = self.extractor.extract(image)
            sift_time = time.perf_counter() - sift_start

            # -- (b)/(c) matching and robust homography ----------------------
            outcome = self._register_frame(
                image,
                features,
                index,
                parent_index,
                parent_features,
                self.chain.get(parent_index),
                placed,
            )

            if outcome.rejected:
                step = self._handle_rejected_frame(
                    index,
                    name,
                    outcome,
                    features,
                    sift_time,
                    outcome.match_time,
                    outcome.homography_time,
                    load_time,
                    step_start,
                )
                report.steps.append(step)
                report.rejected += 1
                continue

            # -- (d) warp onto the canvas and record -------------------------
            step = self._place_and_record(
                index,
                name,
                image,
                parent_image,
                features,
                outcome,
                sift_time,
                outcome.match_time,
                outcome.homography_time,
                load_time,
                step_start,
                placed,
            )
            report.steps.append(step)
            report.accepted += 1

            self._update_state(index, features, outcome, placed)
            parent_image = image
            parent_features = features
            parent_index = index

        # ------------------------------------------------------------ summary
        return self._finalize(report)