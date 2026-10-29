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

"""Measure slanted-edge MTF50 on an exported image.

Examples:
  uv run scripts/measure_mtf50.py merged.tif
  uv run scripts/measure_mtf50.py export.tif --roi 400,300,500,500
  uv run scripts/measure_mtf50.py export.tif --profile srgb

The default profile is detected from a recognizable embedded ICC profile. An
untagged image is treated as Adobe RGB, NegPy's default export profile.
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path
from typing import Sequence

import cv2
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageCms
import tifffile

_ADOBE_GAMMA = 563.0 / 256.0
_ADOBE_LUMA = np.array([0.2973769, 0.6273491, 0.0752741], dtype=np.float64)
_REC709_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
_PROFILES = ("auto", "srgb", "adobe-rgb", "a98c", "linear-srgb", "linear-adobe-rgb")


def _read_image(path: Path) -> tuple[np.ndarray, bytes | None]:
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        with tifffile.TiffFile(path) as tif:
            image = tif.pages[0].asarray()
            tag = tif.pages[0].tags.get(34675)
            icc = bytes(tag.value) if tag is not None else None
        return np.asarray(image), icc

    image = np.asarray(iio.imread(path))
    try:
        with Image.open(path) as opened:
            icc = opened.info.get("icc_profile")
    except Exception:
        icc = None
    return image, icc


def _profile_from_icc(icc: bytes | None) -> str | None:
    if not icc:
        return None
    try:
        profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        description = ImageCms.getProfileDescription(profile).lower()
    except Exception as exc:
        raise ValueError(f"Cannot read embedded ICC profile: {exc}") from exc
    if description.strip() == "a98c" or "adobe rgb" in description or "adobe compat" in description:
        return "adobe-rgb"
    if "srgb" in description or "iec 61966" in description:
        return "srgb"
    raise ValueError(f"Unsupported embedded profile {description!r}; pass --profile explicitly")


def _resolve_profile(requested: str, icc: bytes | None) -> tuple[str, str]:
    if requested != "auto":
        return requested, "command line"
    embedded = _profile_from_icc(icc)
    if embedded is not None:
        return embedded, "embedded ICC"
    return "adobe-rgb", "untagged NegPy default assumption"


def _linearize(rgb: np.ndarray, profile: str) -> tuple[np.ndarray, np.ndarray]:
    if profile == "a98c":
        profile = "adobe-rgb"
    if profile.startswith("linear-"):
        linear = rgb
    elif profile == "srgb":
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    else:
        linear = np.power(np.clip(rgb, 0.0, None), _ADOBE_GAMMA)
    luma_weights = _REC709_LUMA if profile.endswith("srgb") else _ADOBE_LUMA
    return np.asarray(linear, dtype=np.float64), luma_weights


def _to_unit_float(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.integer):
        if np.issubdtype(image.dtype, np.signedinteger) and image.min() < 0:
            raise ValueError("Signed images with negative samples are not supported")
        scale = float(np.iinfo(image.dtype).max)
        return image.astype(np.float64) / scale
    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError(f"Unsupported image dtype: {image.dtype}")
    result = image.astype(np.float64)
    if not np.isfinite(result).all() or result.min() < 0.0 or result.max() > 1.0:
        raise ValueError("Floating-point image samples must be finite and in [0, 1]")
    return result


def _rgb_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(_to_unit_float(image)[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected a grayscale or RGB image, got shape {image.shape}")
    return _to_unit_float(image[..., :3])


def _parse_roi(value: str, image_shape: tuple[int, int]) -> tuple[slice, slice]:
    try:
        x, y, width, height = (int(part) for part in value.split(","))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height") from exc
    image_height, image_width = image_shape
    if x < 0 or y < 0 or width < 1 or height < 1 or x + width > image_width or y + height > image_height:
        raise argparse.ArgumentTypeError("ROI must be a positive rectangle inside the image")
    return slice(y, y + height), slice(x, x + width)


def slanted_edge_mtf50(plane: np.ndarray, oversampling: int = 4) -> float:
    """Return slanted-edge MTF50 in cycles per pixel; reject unsuitable ROIs."""
    image = np.asarray(plane, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 32:
        raise ValueError("MTF50 needs a 2D ROI at least 32 pixels on each side")
    if oversampling < 2:
        raise ValueError("Oversampling must be at least 2")

    height, width = image.shape
    gradient = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    left = int(width * 0.15)
    right = max(left + 1, int(width * 0.85))
    edge_x = left + np.argmax(np.abs(gradient[:, left:right]), axis=1)
    edge_y = np.arange(height, dtype=np.float64)
    slope, intercept = np.polyfit(edge_y, edge_x, 1)
    fitted_x = slope * edge_y + intercept
    residual = edge_x - fitted_x
    if float(np.sqrt(np.mean(residual**2))) > 1.0:
        raise ValueError("ROI does not contain one straight, dominant vertical edge")
    if abs(slope) * height < 1.0:
        raise ValueError("Edge must be slanted enough to provide subpixel samples")

    left_level = float(np.median(image[:, : max(1, width // 4)]))
    right_level = float(np.median(image[:, 3 * width // 4 :]))
    if abs(right_level - left_level) < 1e-4:
        raise ValueError("ROI edge has insufficient contrast")

    distance = np.arange(width, dtype=np.float64)[None, :] - (slope * edge_y[:, None] + intercept)
    distance /= math.sqrt(1.0 + slope * slope)
    span = min(width * 0.45, height * 0.45)
    sample_mask = np.abs(distance) < span
    bins_count = int(math.ceil(2.0 * span * oversampling))
    bin_index = np.floor((distance[sample_mask] + span) * oversampling).astype(np.int64)
    values = image[sample_mask]
    inside = (bin_index >= 0) & (bin_index < bins_count)
    bin_index, values = bin_index[inside], values[inside]
    sums = np.bincount(bin_index, weights=values, minlength=bins_count)
    counts = np.bincount(bin_index, minlength=bins_count)
    valid = counts > 0
    if valid.sum() < oversampling * 16:
        raise ValueError("ROI does not provide enough slanted-edge samples")
    positions = np.arange(bins_count)
    esf = np.interp(positions, positions[valid], sums[valid] / counts[valid])

    lsf = np.diff(esf)
    lsf *= np.hanning(len(lsf))
    fft_size = 1 << int(math.ceil(math.log2(len(lsf) * 8)))
    spectrum = np.abs(np.fft.rfft(lsf, n=fft_size))
    frequency = np.fft.rfftfreq(fft_size, d=1.0 / oversampling)
    correction = np.sinc(frequency / oversampling)
    mtf = spectrum / np.maximum(correction, 1e-12)
    if mtf[0] <= 0:
        raise ValueError("ROI edge spread has no measurable modulation")
    mtf /= mtf[0]
    below = np.flatnonzero((frequency <= 0.5) & (mtf <= 0.5))
    if not len(below):
        raise ValueError("MTF50 is above the measurable Nyquist range")
    index = int(below[0])
    if index == 0:
        return float(frequency[0])
    f0, f1 = frequency[index - 1 : index + 1]
    m0, m1 = mtf[index - 1 : index + 1]
    return float(f0 + (0.5 - m0) * (f1 - f0) / (m1 - m0))


def measure(path: Path, profile: str = "auto", roi: str | None = None) -> tuple[float, dict[str, float | None], str]:
    image, icc = _read_image(path)
    rgb = _rgb_float(image)
    resolved, source = _resolve_profile(profile, icc)
    linear, weights = _linearize(rgb, resolved)
    if roi:
        ys, xs = _parse_roi(roi, linear.shape[:2])
        linear = linear[ys, xs]
    luminance = np.sum(linear * weights, axis=2)
    score = slanted_edge_mtf50(luminance)
    channels: dict[str, float | None] = {}
    for channel, name in enumerate(("R", "G", "B")):
        try:
            channels[name] = slanted_edge_mtf50(linear[..., channel])
        except ValueError:
            channels[name] = None
    return score, channels, f"{resolved} ({source})"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Exported RGB image containing a slanted edge")
    parser.add_argument("--roi", help="Edge ROI as x,y,width,height; defaults to the full image")
    parser.add_argument("--profile", choices=_PROFILES, default="auto", help="Embedded profile or transfer/primaries override")
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error(f"image not found: {args.image}")
    try:
        luminance, channels, profile = measure(args.image, args.profile, args.roi)
    except (OSError, ValueError, argparse.ArgumentTypeError, tifffile.TiffFileError) as exc:
        parser.error(str(exc))
    print(f"Profile: {profile}")
    print(f"Luminance MTF50: {luminance:.5f} cycles/pixel")
    channel_text = "  ".join(f"{name} {value:.5f}" if value is not None else f"{name} n/a" for name, value in channels.items())
    print("Channel MTF50: " + channel_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
