#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#     "imageio>=2.37",
#     "numpy>=2.0",
#     "opencv-python-headless>=4.10",
#     "pillow>=11",
#     "tifffile>=2025.1",
# ]
# ///

"""Compare sharpness metrics for red- and green-aligned trichrome TIFF exports.

Examples:
  uv run scripts/compare_red_and_green_aligned_trichrome_merges.py red.tif green.tif
  uv run scripts/compare_red_and_green_aligned_trichrome_merges.py red.tif green.tif --roi 155,85,218,383
  uv run scripts/compare_red_and_green_aligned_trichrome_merges.py red.tif green.tif --profile a98c

Laplacian energy is the primary comparison. MTF50 is reported when an edge can
be measured in each image; arbitrary photographs do not always contain one.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile

from negpy.services.capture.focus_meter import sharpness
from negpy.services.capture.meter import gray_of

if __package__:
    from .measure_mtf50 import _PROFILES, _linearize, _parse_roi, _read_image, _resolve_profile, _rgb_float, slanted_edge_mtf50
else:
    from measure_mtf50 import _PROFILES, _linearize, _parse_roi, _read_image, _resolve_profile, _rgb_float, slanted_edge_mtf50

_MIN_MEAN_LEVEL = 4.0


@dataclass(frozen=True)
class ImageScores:
    laplacian_energy: float | None
    mtf50: float | None
    profile: str
    mtf50_note: str | None = None


def laplacian_energy(gray: np.ndarray) -> float | None:
    """Return the live focus-meter sharpness score for a grayscale frame."""
    image = np.asarray(gray, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) < 3:
        raise ValueError("Laplacian energy needs a 2D image at least 3 pixels on each side")
    score = sharpness(image)
    return None if score <= 0.0 else float(score)


def _score_image(image: np.ndarray, icc: bytes | None, profile: str, roi: str | None) -> ImageScores:
    rgb = _rgb_float(image)
    resolved, source = _resolve_profile(profile, icc)
    linear, weights = _linearize(rgb, resolved)
    if roi:
        try:
            ys, xs = _parse_roi(roi, linear.shape[:2])
        except argparse.ArgumentTypeError as exc:
            raise ValueError(str(exc)) from exc
        linear = linear[ys, xs]
    # The live focus meter runs on 8-bit grayscale in 0..255. TIFF data here are linearized
    # float values in 0..1, so match the app's meter by scaling the gray plane before sharpness.
    gray = gray_of(linear) * 255.0
    energy = laplacian_energy(gray)
    luminance = np.sum(linear * weights, axis=2)
    try:
        mtf50 = slanted_edge_mtf50(luminance)
        mtf_note = None
    except ValueError as exc:
        mtf50 = None
        mtf_note = str(exc)
    return ImageScores(energy, mtf50, f"{resolved} ({source})", mtf_note)


def score_tiff(path: Path, profile: str = "auto", roi: str | None = None) -> ImageScores:
    image, icc = _read_image(path)
    return _score_image(image, icc, profile, roi)


def compare_tiffs(
    red_path: Path,
    green_path: Path,
    profile: str = "auto",
    roi: str | None = None,
) -> tuple[ImageScores, ImageScores]:
    red_image, red_icc = _read_image(red_path)
    green_image, green_icc = _read_image(green_path)
    if red_image.shape[:2] != green_image.shape[:2]:
        raise ValueError(f"TIFF dimensions differ: {red_image.shape[:2]} vs {green_image.shape[:2]}")
    return (
        _score_image(red_image, red_icc, profile, roi),
        _score_image(green_image, green_icc, profile, roi),
    )


def _print_score(name: str, score: ImageScores) -> None:
    lap_text = "unavailable (too dark or degenerate)" if score.laplacian_energy is None else f"{score.laplacian_energy:.8g}"
    mtf_text = f"{score.mtf50:.5f} cycles/pixel" if score.mtf50 is not None else f"unavailable ({score.mtf50_note})"
    print(f"{name} profile: {score.profile}")
    print(f"{name} Laplacian energy: {lap_text}")
    print(f"{name} MTF50: {mtf_text}")


def _report_winner(red: ImageScores, green: ImageScores) -> None:
    if red.laplacian_energy is None or green.laplacian_energy is None:
        print("Comparison: unavailable because at least one image has no Laplacian reading")
    elif red.laplacian_energy == green.laplacian_energy:
        print("Comparison: Laplacian energy scores tie")
    else:
        winner = "red-aligned" if red.laplacian_energy > green.laplacian_energy else "green-aligned"
        high, low = sorted((red.laplacian_energy, green.laplacian_energy), reverse=True)
        relative = "" if low == 0 else f" ({(high / low - 1.0) * 100.0:.2f}% higher)"
        print(f"Comparison: {winner} scores higher{relative}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("red_tiff", type=Path, help="TIFF export aligned to the red exposure")
    parser.add_argument("green_tiff", type=Path, help="TIFF export aligned to the green exposure")
    parser.add_argument("--roi", help="Shared ROI as x,y,width,height; defaults to the full image")
    parser.add_argument(
        "--profile", choices=_PROFILES, default="auto", help="Shared profile override; auto reads each embedded ICC profile"
    )
    args = parser.parse_args(argv)
    for path in (args.red_tiff, args.green_tiff):
        if not path.is_file():
            parser.error(f"TIFF not found: {path}")
        if path.suffix.lower() not in (".tif", ".tiff"):
            parser.error(f"input must be a TIFF file: {path}")
    try:
        red, green = compare_tiffs(args.red_tiff, args.green_tiff, args.profile, args.roi)
    except (OSError, ValueError, argparse.ArgumentTypeError, tifffile.TiffFileError) as exc:
        parser.error(str(exc))
    _print_score("Red-aligned", red)
    _print_score("Green-aligned", green)
    _report_winner(red, green)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
