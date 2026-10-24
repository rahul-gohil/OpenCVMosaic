"""Merge chunked mosaic outputs into a single orthomosaic.

The chunked runs of the pipeline leave one mosaic per chunk.  This module
stitches them together pairwise -- SIFT, a homography and a masked composite
-- and repeats the process on the results until a single mosaic is left.

The validity of a mosaic is recovered from its empty background rather than
from a mask file: the canvas is only covered where a frame was actually
imaged, and the rest is black.  When a mask file exists next to the mosaic
(the pipeline writes ``mosaic_final_mask.png``) it is used as-is.
"""

from __future__ import annotations

import glob
import os
import time
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .config import DetectorConfig, MatcherConfig, RegistrationConfig
from .features import FeatureExtractor, FeatureSet
from .matching import DescriptorMatcher
from .registration import Registration, estimate_homography


def derive_mask(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Return the validity mask of a mosaic, recovered from its black background.

    Only the dark pixels that are *connected to the border* are empty.  The
    scene itself has very dark pixels -- the shadow of a tree, the wet mud of a
    track -- and treating them as empty would erase them from the mosaic.  The
    empty part of the canvas is the outside of the imaged area, which always
    reaches the border.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = grey.shape

    dark = (grey <= threshold).astype(np.uint8)
    filled = dark.copy()
    for seed in _border_seeds(dark):
        if filled[seed[1], seed[0]] == 1:
            cv2.floodFill(filled, None, seed, 2, flags=4)
    empty = filled == 2
    mask = (~empty).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def load_mask(path: str, image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Return the validity mask of the mosaic stored at ``path``.

    The pipeline writes the mask of every mosaic it produces, so it is used
    when it is there.  Only a mosaic that was not produced by the pipeline
    (an intermediate of a previous merge) has to have its mask recovered
    from its black background.
    """
    stem = os.path.splitext(path)[0]
    for candidate in (stem + "_mask.png",
                      os.path.join(os.path.dirname(path), "mosaic_final_mask.png")):
        if os.path.exists(candidate):
            mask = cv2.imread(candidate, cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape == image.shape[:2]:
                return mask
    return derive_mask(image, threshold)


def _border_seeds(dark: np.ndarray):
    """Yield the dark pixels on the border of the image."""
    h, w = dark.shape
    for x in range(0, w, 64):
        for y in (0, h - 1):
            if dark[y, x]:
                yield (x, y)
    for y in range(0, h, 64):
        for x in (0, w - 1):
            if dark[y, x]:
                yield (x, y)


def content_bbox(mask: np.ndarray, band: int = 4_000_000):
    """Return the bounding box of the content of a mask, scanning band by band."""
    h, w = mask.shape
    step = max(1, band // max(w, 1))
    x0, y0, x1, y1 = w, h, 0, 0
    for y in range(0, h, step):
        block = np.asarray(mask[y : y + step])
        ys, xs = np.nonzero(block)
        if len(xs) == 0:
            continue
        x0 = min(x0, int(xs.min()))
        x1 = max(x1, int(xs.max()) + 1)
        y0 = min(y0, y + int(ys.min()))
        y1 = max(y1, y + int(ys.max()) + 1)
    return x0, y0, x1, y1


def crop_to_content(image: np.ndarray, mask: np.ndarray, pad: int = 64):
    """Crop a mosaic to the bounding box of the content it holds."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return image, mask
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(image.shape[1], int(xs.max()) + pad)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(image.shape[0], int(ys.max()) + pad)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def _similarity(fs_a, fs_b, knn, matcher, registration, min_inliers):
    """Register a pair with a similarity transform instead of a homography."""
    best = None
    for ratio in matcher.config.ratio_ladder:
        matches = matcher.ratio_filter(knn, ratio)
        if len(matches) < 4:
            continue
        src = np.float32([fs_b.points[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([fs_a.points[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
        A, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=registration.ransac_threshold,
            maxIters=registration.max_iters, confidence=registration.confidence)
        if A is None or inliers is None:
            continue
        flags = inliers.ravel().astype(bool)
        n_inliers = int(flags.sum())
        if n_inliers < min_inliers:
            continue
        H = np.vstack([A, [0.0, 0.0, 1.0]])
        residual = float(np.median(np.linalg.norm(
            cv2.perspectiveTransform(src[flags], H).reshape(-1, 2)
            - dst[flags].reshape(-1, 2), axis=1)))
        if best is None or n_inliers > best.inliers:
            best = Registration(ok=True, H=H, matches=len(matches), inliers=n_inliers,
                                inlier_ratio=n_inliers / len(matches),
                                reprojection_error=residual, reason="similarity")
    return best


def stitch(a_img, a_mask, b_img, b_mask, features, detect_px, work_dir,
           min_inliers, min_inlier_ratio):
    """Merge two mosaics into one, returning the mosaic, its mask, the fit, and timings."""
    import time as _time
    ah, aw = a_img.shape[:2]
    bh, bw = b_img.shape[:2]

    scale = min(1.0, (detect_px / float(aw * ah)) ** 0.5,
                (detect_px / float(bw * bh)) ** 0.5)
    a_small = cv2.resize(a_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    b_small = cv2.resize(b_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    extractor = FeatureExtractor(DetectorConfig())
    matcher = DescriptorMatcher(MatcherConfig())
    registration = RegistrationConfig(min_inliers=min_inliers,
                                      min_inlier_ratio=min_inlier_ratio)

    _t_sift = _time.perf_counter()
    fa = extractor.extract(a_small, budget=features)
    fb = extractor.extract(b_small, budget=features)
    fs_a = FeatureSet(points=fa.points / scale, descriptors=fa.descriptors)
    fs_b = FeatureSet(points=fb.points / scale, descriptors=fb.descriptors)
    sift_time = _time.perf_counter() - _t_sift

    _t_match = _time.perf_counter()
    knn = matcher.knn_match(fa.descriptors, fb.descriptors)
    _t_homography = 0.0
    best = None
    for ratio in matcher.config.ratio_ladder:
        matches = matcher.ratio_filter(knn, ratio)
        if len(matches) < 4:
            continue
        _th = _time.perf_counter()
        candidate = estimate_homography(fs_a, fs_b, matches, registration)
        _t_homography += _time.perf_counter() - _th
        if candidate.ok and (best is None or candidate.inliers > best.inliers):
            best = candidate

    if best is None or not best.ok:
        _ts = _time.perf_counter()
        best = _similarity(fs_a, fs_b, knn, matcher, registration, min_inliers)
        _t_homography += _time.perf_counter() - _ts
    match_time = _time.perf_counter() - _t_match - _t_homography
    if best is None or not best.ok:
        raise RuntimeError(f"could not register the pair: {best.reason if best else 'no model'}")
    H = best.H
    homography_time = _t_homography

    _t_stitch_start = _time.perf_counter()
    corners = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]]).reshape(-1, 1, 2)
    wc = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    x0 = int(np.floor(min(wc[:, 0].min(), 0)))
    y0 = int(np.floor(min(wc[:, 1].min(), 0)))
    x1 = int(np.ceil(max(wc[:, 0].max(), aw)))
    y1 = int(np.ceil(max(wc[:, 1].max(), ah)))
    width, height = x1 - x0, y1 - y0

    to_canvas = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    M = to_canvas @ H

    def mapped(name, shape):
        path = os.path.join(work_dir, name)
        return path, np.memmap(path, dtype=np.uint8, mode="w+", shape=shape)

    canvas_path, canvas = mapped("canvas.tmp", (height, width, 3))
    valid_path, valid = mapped("valid.tmp", (height, width))
    warped_path, warped = mapped("warped.tmp", (height, width, 3))
    wmask_path, wmask = mapped("wmask.tmp", (height, width))

    canvas[-y0:-y0 + ah, -x0:-x0 + aw] = a_img
    valid[-y0:-y0 + ah, -x0:-x0 + aw] = a_mask

    cv2.warpPerspective(b_img, M, (width, height), dst=warped,
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0))
    cv2.warpPerspective(b_mask, M, (width, height), dst=wmask,
                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0)
    wmask[:] = cv2.erode(np.asarray(wmask), np.ones((3, 3), np.uint8))

    diff_sum = 0.0
    diff_count = 0
    band = max(1, int(4_000_000 // max(width, 1)))
    for y_start in range(0, height, band):
        y_stop = min(y_start + band, height)
        a_band = np.asarray(canvas[y_start:y_stop])
        v_band = np.asarray(valid[y_start:y_stop])
        w_band = np.asarray(warped[y_start:y_stop])
        m_band = np.asarray(wmask[y_start:y_stop])
        overlap = (m_band > 0) & (v_band > 0)
        if overlap.any():
            d = np.abs(a_band.astype(np.int16) - w_band.astype(np.int16)).mean(axis=2)
            diff_sum += float(d[overlap].sum())
            diff_count += int(overlap.sum())
            fade = cv2.distanceTransform(m_band, cv2.DIST_L2, 3)
            w = np.clip(fade / 32.0, 0.0, 1.0)[..., None]
            blended = a_band.astype(np.float32) * (1 - w) + w_band.astype(np.float32) * w
            a_band = np.where(overlap[..., None], blended, a_band).astype(np.uint8)
        only_b = (m_band > 0) & (v_band == 0)
        a_band[only_b] = w_band[only_b]
        canvas[y_start:y_stop] = a_band
        valid[y_start:y_stop] = ((v_band > 0) | (m_band > 0)).astype(np.uint8) * 255

    canvas.flush()
    valid.flush()
    del warped, wmask
    for path in (warped_path, wmask_path):
        if os.path.exists(path):
            os.remove(path)
    agreement = diff_sum / diff_count if diff_count else float("nan")
    stitch_time = _time.perf_counter() - _t_stitch_start
    merge_timings = {
        "sift": round(sift_time, 3),
        "matching": round(match_time, 3),
        "homography": round(homography_time, 3),
        "stitching": round(stitch_time, 3),
    }
    return canvas, valid, best, agreement, merge_timings


class MosaicMerger:
    """Merge chunked mosaic outputs into a single mosaic.

    Parameters
    ----------
    inputs : list of str
        Mosaic file paths (or glob patterns) to merge, in order.
    out_dir : str
        Output folder.
    features : int
        Keypoints detected on each mosaic.
    detect_px : float
        Pixel budget of the downscaled copy used for detection.
    min_inliers : int
        Minimum inliers to accept a merge.
    min_inlier_ratio : float
        Minimum inlier ratio to accept a merge.
    threshold : int
        Grey level above which a pixel counts as content.
    quality : int
        JPEG quality for output.
    log : callable
        Logging function.
    """

    def __init__(
        self,
        inputs: List[str],
        out_dir: str = "outputs/merged",
        features: int = 4000,
        detect_px: float = 4e6,
        min_inliers: int = 8,
        min_inlier_ratio: float = 0.03,
        threshold: int = 10,
        quality: int = 92,
        log: Callable[[str], None] = print,
    ) -> None:
        self.inputs = inputs
        self.out_dir = out_dir
        self.features = features
        self.detect_px = detect_px
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.threshold = threshold
        self.quality = quality
        self.log = log
        self._merge_timings = {"sift": 0.0, "matching": 0.0, "homography": 0.0, "stitching": 0.0}

    @property
    def merge_timings(self) -> dict:
        return {k: round(v, 2) for k, v in self._merge_timings.items()}

    def _resolve_paths(self) -> List[str]:
        paths = []
        for pattern in self.inputs:
            if any(c in pattern for c in "*?["):
                paths.extend(sorted(glob.glob(pattern)))
            else:
                paths.append(pattern)
        return paths

    def run(self) -> Optional[str]:
        """Execute the carry-rule merge tree. Returns the final mosaic path."""
        paths = self._resolve_paths()
        if not paths:
            self.log("no inputs")
            return None
        os.makedirs(self.out_dir, exist_ok=True)
        self.log(f"{len(paths)} mosaics to merge")

        level = list(paths)
        round_no = 0
        work = os.path.join(self.out_dir, "work")
        os.makedirs(work, exist_ok=True)
        carried: Optional[str] = None
        while len(level) > 1 or carried is not None:
            round_no += 1
            items = list(level)
            if carried is not None:
                items.append(carried)
                if len(items) >= 2:
                    head, tail = items[:-2], items[-2:]
                else:
                    head, tail = [], items
            else:
                head, tail = items, []

            pairs: List[Tuple[str, str]] = []
            i = 0
            while i + 1 < len(head):
                pairs.append((head[i], head[i + 1]))
                i += 2
            leftover = head[i] if i < len(head) else None
            if len(tail) == 2:
                pairs.append((tail[0], tail[1]))
            elif len(tail) == 1:
                leftover = tail[0]

            nxt: List[str] = []
            for left, right in pairs:
                t = time.perf_counter()
                a = cv2.imread(left)
                b = cv2.imread(right)
                if a is None or b is None:
                    self.log(f"  cannot read {left} / {right}")
                    return None
                a_mask = load_mask(left, a, self.threshold)
                b_mask = load_mask(right, b, self.threshold)
                a, a_mask = crop_to_content(a, a_mask)
                b, b_mask = crop_to_content(b, b_mask)
                try:
                    canvas, mask, reg, agreement, merge_timings = stitch(
                        a, a_mask, b, b_mask, self.features, self.detect_px,
                        work, self.min_inliers, self.min_inlier_ratio)
                    for k in ("sift", "matching", "homography", "stitching"):
                        self._merge_timings[k] = self._merge_timings.get(k, 0.0) + merge_timings.get(k, 0.0)
                except RuntimeError as error:
                    self.log(f"  {os.path.basename(left)} + {os.path.basename(right)}"
                             f" -> SKIPPED ({error})")
                    nxt.append(left)
                    continue
                cx0, cy0, cx1, cy1 = content_bbox(mask)
                cx0 = max(0, cx0 - 64); cy0 = max(0, cy0 - 64)
                cx1 = min(canvas.shape[1], cx1 + 64); cy1 = min(canvas.shape[0], cy1 + 64)
                out_path = os.path.join(self.out_dir, f"merge_r{round_no}_{len(nxt):02d}.jpg")
                cv2.imwrite(out_path, np.asarray(canvas)[cy0:cy1, cx0:cx1],
                            [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                cv2.imwrite(os.path.splitext(out_path)[0] + "_mask.png",
                            np.asarray(mask)[cy0:cy1, cx0:cx1])
                self.log(
                    f"  {os.path.basename(left)} + {os.path.basename(right)}"
                    f" -> {os.path.basename(out_path)}"
                    f"  {cx1-cx0}x{cy1-cy0}"
                    f"  inliers {reg.inliers}/{reg.matches} residual {reg.reprojection_error:.2f}px"
                    f"  agreement {agreement:.1f}"
                    f"  ({time.perf_counter()-t:.1f}s)"
                )
                del a, b, a_mask, b_mask, canvas, mask
                nxt.append(out_path)

            carried = leftover
            level = nxt
            if carried is not None and not level:
                break

        final = level[0] if level else carried
        if final is None:
            self.log("nothing to merge")
            return None
        target = os.path.join(self.out_dir, "mosaic_merged.jpg")
        if os.path.exists(final):
            img = cv2.imread(final)
            if img is not None:
                mask = load_mask(final, img, self.threshold)
                fx0, fy0, fx1, fy1 = content_bbox(mask)
                fx0 = max(0, fx0 - 64); fy0 = max(0, fy0 - 64)
                fx1 = min(img.shape[1], fx1 + 64); fy1 = min(img.shape[0], fy1 + 64)
                cv2.imwrite(target, img[fy0:fy1, fx0:fx1],
                            [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                cv2.imwrite(os.path.splitext(target)[0] + "_mask.png",
                            mask[fy0:fy1, fx0:fx1])
            else:
                os.replace(final, target)
        self.log("merged mosaic: " + target)
        # Write merge timings to JSON file
        import json as _json
        timings_path = os.path.join(self.out_dir, "merge_timings.json")
        try:
            with open(timings_path, "w") as f:
                _json.dump(self.merge_timings, f)
        except Exception:
            pass
        return target