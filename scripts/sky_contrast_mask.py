#!/usr/bin/env -S uv run --no-project --script

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "huggingface-hub==0.36.2",
#     "imagecodecs==2026.6.6",
#     "numba==0.65.1",
#     "numpy==2.4.4",
#     "onnxruntime==1.23.2",
#     "opencv-contrib-python-headless==4.13.0.92",
#     "pillow==12.2.0",
#     "pymatting==1.1.15",
#     "pytest==9.0.2",
#     "scipy==1.17.1",
#     "tifffile==2026.6.1",
# ]
# ///
"""Standalone sky-guided contrast-mask proof of concept."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable


MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
MODEL_REVISION = "b9175de73a0a34f7843135853d27629aa6987b2f"
MODEL_FILENAME = "onnx/model.onnx"
MODEL_LICENSE = "NVIDIA Source Code License for SegFormer; research/evaluation use only, non-commercial"
SKY_CLASS_INDEX = 2
SKY_CLASS_COUNT = 150
MODEL_SIZE = 512
MODEL_MEAN = (0.485, 0.456, 0.406)
MODEL_STD = (0.229, 0.224, 0.225)
ICC_PROFILE_TAG = 34675
LUMA = (0.2126, 0.7152, 0.0722)
A98_GAMMA = 563.0 / 256.0
_A98_TO_XYZ = (
    (0.5767309, 0.1855540, 0.1881852),
    (0.2973769, 0.6273491, 0.0752741),
    (0.0270343, 0.0706872, 0.9911085),
)
_XYZ_TO_SRGB = (
    (3.2404542, -1.5371385, -0.4985314),
    (-0.9692660, 1.8760108, 0.0415560),
    (0.0556434, -0.2040259, 1.0572252),
)


def _module(name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(f"{name} is required; install the POC dependencies from scripts/sky_contrast_mask_requirements.txt") from exc


def _numpy() -> Any:
    return _module("numpy", "")


def read_rgb_tiff(path: Path) -> tuple[Any, dict[str, Any], bytes | None]:
    np = _numpy()
    tifffile = _module("tifffile", "")
    with tifffile.TiffFile(path) as source:
        if len(source.pages) != 1:
            raise ValueError("Input must be a single-page TIFF")
        page = source.pages[0]
        if int(page.photometric) != 2 or int(page.samplesperpixel or 0) != 3:
            raise ValueError("Input must be RGB with exactly three samples per pixel")
        if page.planarconfig is not None and int(page.planarconfig) != 1:
            raise ValueError("Planar RGB TIFF data is not supported; use contiguous RGB samples")
        if page.extrasamples:
            raise ValueError("TIFF alpha or extra samples are not supported")
        image = source.asarray()
        profile_tag = page.tags.get(ICC_PROFILE_TAG)
        profile = bytes(profile_tag.value) if profile_tag is not None else None
        metadata = {
            "width": int(page.imagewidth),
            "height": int(page.imagelength),
            "dtype": str(image.dtype),
            "photometric": str(page.photometric),
            "samples_per_pixel": int(page.samplesperpixel),
            "planar_configuration": str(page.planarconfig),
            "icc_profile_present": profile is not None,
        }
    if image.dtype != np.uint16 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must contain contiguous unsigned 16-bit RGB samples")
    if image.shape[:2] != (metadata["height"], metadata["width"]):
        raise ValueError("TIFF dimensions do not match the decoded RGB array")
    return image, metadata, profile


def _profile_description(profile_bytes: bytes) -> str:
    imagecms = _module("PIL.ImageCms", "")
    import io

    try:
        profile = imagecms.ImageCmsProfile(io.BytesIO(profile_bytes))
        return imagecms.getProfileDescription(profile).strip()
    except Exception as exc:
        raise ValueError("The embedded ICC profile is invalid or unreadable") from exc


def srgb_to_linear(rgb: Any) -> Any:
    np = _numpy()
    values = np.asarray(rgb, dtype=np.float64)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: Any) -> Any:
    np = _numpy()
    values = np.asarray(rgb, dtype=np.float64)
    positive = np.maximum(values, 0.0)
    return np.where(positive <= 0.0031308, 12.92 * positive, 1.055 * positive ** (1.0 / 2.4) - 0.055)


def _a98_profile_matches(profile: bytes, description: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", description.lower())
    header_ids = {profile[start : start + 4].decode("ascii", errors="ignore").strip().lower() for start in (48, 52)}
    return "a98c" in normalized or "adobergb1998" in normalized or "a98c" in header_ids


def _a98_to_linear_srgb(encoded: Any) -> Any:
    np = _numpy()
    linear_a98 = np.asarray(encoded, dtype=np.float64) ** A98_GAMMA
    matrix = np.asarray(_XYZ_TO_SRGB) @ np.asarray(_A98_TO_XYZ)
    return linear_a98 @ matrix.T


def _linear_srgb_to_a98(linear_srgb: Any) -> Any:
    np = _numpy()
    matrix = np.asarray(_XYZ_TO_SRGB) @ np.asarray(_A98_TO_XYZ)
    linear_a98 = np.asarray(linear_srgb, dtype=np.float64) @ np.linalg.inv(matrix).T
    return np.sign(linear_a98) * np.abs(linear_a98) ** (1.0 / A98_GAMMA)


def prepare_input(
    samples: Any,
    representation: str,
    log_min_log2: float | None = None,
    log_max_log2: float | None = None,
    profile: bytes | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    np = _numpy()
    values = np.asarray(samples, dtype=np.float64) / 65535.0
    if not np.isfinite(values).all():
        raise ValueError("Input contains non-finite sample values")

    profile_name = None
    if profile is not None:
        profile_name = _profile_description(profile)
        normalized_name = re.sub(r"[^a-z0-9]", "", profile_name.lower())
        if representation == "a98c":
            if not _a98_profile_matches(profile, profile_name):
                raise ValueError(f"Embedded ICC profile {profile_name!r} is not an A98C or Adobe RGB (1998) profile")
        elif representation == "srgb":
            if "srgb" not in normalized_name:
                raise ValueError(f"Embedded ICC profile {profile_name!r} is not an sRGB profile")
        else:
            raise ValueError("Embedded ICC profiles require --representation srgb or a98c")

    log_mapping = None
    if representation == "linear":
        linear = values
        display = linear_to_srgb(linear)
    elif representation == "srgb":
        linear = srgb_to_linear(values)
        display = values
    elif representation == "a98c":
        if profile is None:
            raise ValueError("--representation a98c requires an embedded A98C or Adobe RGB (1998) ICC profile")
        linear = _a98_to_linear_srgb(values)
        display = linear_to_srgb(linear)
    elif representation == "log":
        if profile is not None:
            raise ValueError("Log-space TIFF data cannot use an embedded color profile")
        if log_min_log2 is None or log_max_log2 is None:
            raise ValueError("Log input requires --log-min-log2 and --log-max-log2")
        if not math.isfinite(log_min_log2) or not math.isfinite(log_max_log2) or log_min_log2 >= log_max_log2:
            raise ValueError("Log mapping bounds must be finite and log-min-log2 must be below log-max-log2")
        if log_min_log2 < -32 or log_max_log2 > 32:
            raise ValueError("Log mapping bounds must be between -32 and 32 stops")
        log_values = log_min_log2 + values * (log_max_log2 - log_min_log2)
        linear = np.exp2(log_values)
        display = linear_to_srgb(linear)
        log_mapping = {"min_log2": log_min_log2, "max_log2": log_max_log2}
    else:
        raise ValueError("Representation must be linear, srgb, or log")

    return (
        linear,
        np.clip(display, 0.0, 1.0),
        {
            "representation": representation,
            "input_primaries": "Adobe RGB (1998)" if representation == "a98c" else "sRGB",
            "icc_profile_description": profile_name,
            "icc_conversion": (
                "Adobe RGB (1998) gamma and matrix primaries converted to linear sRGB"
                if representation == "a98c"
                else "identity; embedded profile identifies sRGB primaries"
                if profile is not None
                else "none; sRGB primaries assumed"
            ),
            "log_display_mapping": log_mapping,
        },
    )


def model_input_size(session: Any) -> tuple[int | None, int | None]:
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError("ONNX model must have exactly one image input")
    shape = inputs[0].shape
    if len(shape) != 4:
        raise ValueError(f"ONNX input must use NCHW RGB layout; found {shape}")
    channel_dim = shape[1]
    if isinstance(channel_dim, str):
        channel_dim = int(channel_dim) if channel_dim.isdecimal() else None
    if channel_dim not in (3, None):
        raise ValueError(f"ONNX input must use NCHW RGB layout; found {shape}")
    height, width = shape[2], shape[3]
    return (height if isinstance(height, int) else None, width if isinstance(width, int) else None)


def infer_sky_probability(display_rgb: Any, session: Any) -> tuple[Any, dict[str, Any]]:
    np = _numpy()
    cv2 = _module("cv2", "")
    input_info = session.get_inputs()[0]
    if getattr(input_info, "type", "tensor(float)") != "tensor(float)":
        raise ValueError("ONNX model input must be float32")
    model_height, model_width = model_input_size(session)
    height, width = display_rgb.shape[:2]
    if model_height is None and model_width is None:
        resized_height = MODEL_SIZE
        resized_width = max(1, int(round(width * MODEL_SIZE / height))) if height <= width else MODEL_SIZE
        if height > width:
            resized_height = max(1, int(round(height * MODEL_SIZE / width)))
    elif model_height and model_width:
        resized_height, resized_width = model_height, model_width
    else:
        raise ValueError("ONNX model must have both fixed spatial dimensions or both dynamic dimensions")

    resized = cv2.resize(display_rgb.astype(np.float32), (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    mean = np.asarray(MODEL_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(MODEL_STD, dtype=np.float32).reshape(1, 1, 3)
    tensor = ((resized - mean) / std).transpose(2, 0, 1)[None, ...].astype(np.float32)
    outputs = session.run(None, {input_info.name: tensor})
    if not outputs:
        raise ValueError("ONNX model returned no outputs")
    logits = np.asarray(outputs[0])
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != SKY_CLASS_COUNT:
        raise ValueError(f"ONNX output must have shape [1, 150, height, width]; found {logits.shape}")
    if not np.isfinite(logits).all():
        raise ValueError("ONNX model returned non-finite logits")
    logits = logits.astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    sky = probabilities[0, SKY_CLASS_INDEX].astype(np.float32)
    print(f"Raw sky answer shape: {sky.shape}")
    sky = cv2.resize(sky, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(sky, 0.0, 1.0), {
        "model_id": MODEL_ID,
        "model_path": None,
        "sky_class_index": SKY_CLASS_INDEX,
        "class_count": SKY_CLASS_COUNT,
        "input_layout": "NCHW RGB float32",
        "resize": {"height": resized_height, "width": resized_width, "policy": "fixed model size or shortest edge 512"},
        "normalization_mean": list(MODEL_MEAN),
        "normalization_std": list(MODEL_STD),
    }


def guided_probability(display_rgb: Any, probability: Any, radius: int, epsilon: float) -> Any:
    np = _numpy()
    cv2 = _module("cv2", "")
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "guidedFilter"):
        raise RuntimeError("OpenCV contrib with cv2.ximgproc.guidedFilter is required")
    if radius < 1 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Guided-filter radius must be positive and epsilon must be finite and positive")
    guide = np.ascontiguousarray(np.clip(display_rgb, 0.0, 1.0), dtype=np.float32)
    source = np.ascontiguousarray(probability, dtype=np.float32)
    refined = cv2.ximgproc.guidedFilter(guide, source, int(radius), float(epsilon))
    if refined.shape != source.shape or not np.isfinite(refined).all():
        raise ValueError("Guided filter returned invalid probability data")
    return np.clip(refined, 0.0, 1.0).astype(np.float32)


def make_trimap(probability: Any, sky_threshold: float, foreground_threshold: float) -> tuple[Any, Any, dict[str, Any]]:
    np = _numpy()
    if not (0.0 <= foreground_threshold < sky_threshold <= 1.0):
        raise ValueError("Trimap thresholds must satisfy 0 <= foreground < sky <= 1")
    if not np.isfinite(probability).all():
        raise ValueError("Probability map contains non-finite values")
    foreground = probability <= foreground_threshold
    sky = probability >= sky_threshold
    unknown = ~(foreground | sky)
    if not foreground.any() or not sky.any():
        raise ValueError("Trimap must contain at least one definite-sky and one definite-foreground pixel")
    trimap = np.full(probability.shape, 0.5, dtype=np.float32)
    trimap[foreground] = 0.0
    trimap[sky] = 1.0
    diagnostic = np.full(probability.shape, 128, dtype=np.uint8)
    diagnostic[foreground] = 0
    diagnostic[sky] = 255
    total = int(probability.size)
    stats = {
        "unknown_pixels": int(unknown.sum()),
        "unknown_fraction": float(unknown.mean()),
        "definite_sky_pixels": int(sky.sum()),
        "definite_sky_fraction": float(sky.mean()),
        "definite_foreground_pixels": int(foreground.sum()),
        "definite_foreground_fraction": float(foreground.mean()),
        "total_pixels": total,
        "sky_threshold": float(sky_threshold),
        "foreground_threshold": float(foreground_threshold),
    }
    return trimap, diagnostic, stats


def estimate_alpha(guide_rgb: Any, trimap: Any, matting_fn: Callable[..., Any] | None = None) -> tuple[Any, dict[str, int]]:
    np = _numpy()
    if matting_fn is None:
        pymatting = _module("pymatting", "")
        matting_fn = pymatting.estimate_alpha_cf
    alpha = np.asarray(matting_fn(np.ascontiguousarray(guide_rgb, dtype=np.float64), np.ascontiguousarray(trimap, dtype=np.float64)))
    if alpha.shape != trimap.shape or not np.isfinite(alpha).all():
        raise ValueError("Closed-form matting returned invalid alpha data")
    clamp_count = int(np.count_nonzero((alpha < 0.0) | (alpha > 1.0)))
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    alpha[trimap >= 1.0] = 1.0
    alpha[trimap <= 0.0] = 0.0
    return alpha, {"clamped_pixels": clamp_count}


def contrast_mask_approximation(linear_rgb: Any, strength: float, blur_percent: float = 4.0) -> tuple[Any, dict[str, Any]]:
    np = _numpy()
    cv2 = _module("cv2", "")
    if not math.isfinite(strength) or not -0.5 <= strength <= 0.5:
        raise ValueError("Contrast-mask strength must be between -0.5 and 0.5")
    if not math.isfinite(blur_percent) or not 2.0 <= blur_percent <= 6.0:
        raise ValueError("Blur scale must be between 2 and 6 percent of the image short side")
    source = np.asarray(linear_rgb, dtype=np.float64)
    if not np.isfinite(source).all():
        raise ValueError("Linear image contains non-finite values")
    if abs(strength) < 1e-12:
        return source.copy(), {"density_range": 0.0, "sigma_pixels": 0.0, "plane_mean": 0.0}

    luminance = np.tensordot(source, np.asarray(LUMA), axes=([2], [0]))
    pseudo_density = -np.log10(np.maximum(luminance, 1e-6))
    low, high = np.percentile(pseudo_density, (1.0, 99.0))
    density_range = max(float(high - low), 1e-6)
    normalized_density = (pseudo_density - low) / density_range
    sigma = blur_percent * 0.01 * min(source.shape[:2])
    blurred = cv2.GaussianBlur(
        np.ascontiguousarray(normalized_density, dtype=np.float32),
        (0, 0),
        sigmaX=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    plane = blurred.astype(np.float64) - float(blurred.mean())
    stops = plane * (-strength * density_range / math.log10(2.0))
    adjusted = source * np.exp2(stops)[..., None]
    if not np.isfinite(adjusted).all():
        raise ValueError("Contrast-mask calculation produced non-finite values")
    return adjusted, {"density_range": density_range, "sigma_pixels": sigma, "plane_mean": float(plane.mean())}


def blend_with_alpha(original: Any, adjusted: Any, alpha: Any) -> Any:
    np = _numpy()
    if original.shape != adjusted.shape or alpha.shape != original.shape[:2]:
        raise ValueError("Original, adjusted image, and alpha dimensions do not match")
    if not np.isfinite(alpha).all() or np.any(alpha < 0.0) or np.any(alpha > 1.0):
        raise ValueError("Alpha must contain finite values between 0 and 1")
    return (1.0 - alpha[..., None]) * original + alpha[..., None] * adjusted


def clipping_stats(values: Any) -> dict[str, Any]:
    np = _numpy()
    array = np.asarray(values, dtype=np.float64)
    outside = (array < 0.0) | (array > 1.0)
    return {
        "out_of_range_samples": int(outside.sum()),
        "out_of_range_fraction": float(outside.mean()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def encode_output(
    linear_rgb: Any, representation: str, log_min_log2: float | None, log_max_log2: float | None
) -> tuple[Any, dict[str, Any]]:
    np = _numpy()
    if representation == "linear":
        encoded = np.asarray(linear_rgb, dtype=np.float64)
    elif representation == "srgb":
        encoded = linear_to_srgb(linear_rgb)
    elif representation == "a98c":
        encoded = _linear_srgb_to_a98(linear_rgb)
    elif representation == "log":
        if log_min_log2 is None or log_max_log2 is None:
            raise ValueError("Log output requires its input log mapping")
        log_values = np.log2(np.maximum(linear_rgb, 2.0**log_min_log2))
        encoded = (log_values - log_min_log2) / (log_max_log2 - log_min_log2)
    else:
        raise ValueError("Representation must be linear, srgb, or log")
    if not np.isfinite(encoded).all():
        raise ValueError("Output encoding contains non-finite values")
    stats = clipping_stats(encoded)
    return np.rint(np.clip(encoded, 0.0, 1.0) * 65535.0).astype(np.uint16), stats


def _uint8_image(rgb: Any) -> Any:
    np = _numpy()
    return np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_gray_png(path: Path, values: Any) -> None:
    np = _numpy()
    image = _module("PIL.Image", "")
    pixels = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    image.fromarray(pixels, mode="L").save(path)


def save_overlay(path: Path, display_rgb: Any, alpha: Any) -> None:
    np = _numpy()
    cv2 = _module("cv2", "")
    image = _module("PIL.Image", "")
    base = np.clip(display_rgb, 0.0, 1.0)
    tint = np.zeros_like(base)
    tint[..., 0] = 1.0
    tint[..., 1] = 0.25
    overlay = base * (1.0 - alpha[..., None] * 0.35) + tint * (alpha[..., None] * 0.35)
    edge = cv2.Canny(np.rint(alpha * 255.0).astype(np.uint8), 32, 224) > 0
    overlay[edge] = (1.0, 1.0, 1.0)
    image.fromarray(_uint8_image(overlay), mode="RGB").save(path)


def save_comparison(path: Path, original: Any, global_result: Any, localized: Any) -> None:
    image = _module("PIL.Image", "")
    image_draw = _module("PIL.ImageDraw", "")
    panels = [_uint8_image(item) for item in (original, global_result, localized)]
    height, width = panels[0].shape[:2]
    strip = 36
    canvas = image.new("RGB", (width * 3, height + strip), (24, 24, 24))
    draw = image_draw.Draw(canvas)
    for index, (pixels, title) in enumerate(zip(panels, ("Original", "Global contrast mask", "Sky-localized"))):
        canvas.paste(image.fromarray(pixels, mode="RGB"), (index * width, strip))
        draw.text((index * width + 12, 10), title, fill=(245, 245, 245))
    canvas.save(path)


def _write_tiff(path: Path, pixels: Any, profile: bytes | None) -> None:
    tifffile = _module("tifffile", "")
    options: dict[str, Any] = {"photometric": "rgb", "metadata": None}
    if profile:
        options["extratags"] = [(ICC_PROFILE_TAG, 7, len(profile), profile, False)]
    tifffile.imwrite(path, pixels, **options)


def _output_profile(representation: str, source_profile: bytes | None) -> bytes | None:
    if representation not in ("srgb", "a98c"):
        return None
    if source_profile is not None:
        return source_profile
    imagecms = _module("PIL.ImageCms", "")
    return imagecms.ImageCmsProfile(imagecms.createProfile("sRGB")).tobytes()


def resolve_model_path(model_path: Path | None = None) -> Path:
    if model_path is not None:
        local_path = model_path.expanduser()
        if not local_path.is_file():
            raise ValueError(f"ONNX model file does not exist: {local_path}")
        return local_path.resolve()

    hub = _module("huggingface_hub", "")
    try:
        downloaded_path = hub.hf_hub_download(
            repo_id=MODEL_ID,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not fetch the pinned SegFormer ONNX model from Hugging Face: {exc}") from exc
    return Path(downloaded_path)


def _create_session(model_path: Path) -> Any:
    if not model_path.is_file():
        raise ValueError(f"ONNX model file does not exist: {model_path}")
    runtime = _module("onnxruntime", "")
    try:
        session = runtime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ValueError(f"Could not load ONNX model {model_path}: {exc}") from exc
    finally:
        for inp in session.get_inputs():
            print(inp.name, inp.shape, inp.type)

        for out in session.get_outputs():
            print(out.name, out.shape, out.type)
        return session


def _validate_trimap_reliability(stats: dict[str, Any], warnings: list[str]) -> None:
    if stats["unknown_fraction"] >= 0.995:
        warnings.append("Almost the entire image is unknown in the trimap")
    for key, label in (("definite_sky_fraction", "definite-sky"), ("definite_foreground_fraction", "definite-foreground")):
        if stats[key] < 0.001:
            warnings.append(f"The {label} trimap region is very small")


def run_pipeline(
    input_path: Path,
    output_directory: Path,
    representation: str,
    model_path: Path | None = None,
    *,
    strength: float = 0.25,
    blur_percent: float = 4.0,
    guided_radius: int = 12,
    guided_epsilon: float = 1e-3,
    sky_threshold: float = 0.95,
    foreground_threshold: float = 0.05,
    log_min_log2: float | None = None,
    log_max_log2: float | None = None,
    session: Any | None = None,
    matting_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    source_path = input_path.expanduser().resolve()
    if not source_path.is_file():
        raise ValueError(f"Input TIFF does not exist: {input_path}")
    samples, image_metadata, profile = read_rgb_tiff(source_path)
    linear, display, input_info = prepare_input(samples, representation, log_min_log2, log_max_log2, profile)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "output.tif"
    if source_path == output_path.resolve():
        raise ValueError("Output path must not overwrite the input TIFF")

    resolved_model_path = resolve_model_path(model_path) if session is None else model_path
    inference_session = session if session is not None else _create_session(resolved_model_path)
    probability, model_info = infer_sky_probability(display, inference_session)
    model_info["model_path"] = str(resolved_model_path) if resolved_model_path is not None else None
    model_info["model_revision"] = MODEL_REVISION if model_path is None else None
    refined = guided_probability(display, probability, guided_radius, guided_epsilon)
    trimap, trimap_png, trimap_stats = make_trimap(refined, sky_threshold, foreground_threshold)
    warnings: list[str] = []
    _validate_trimap_reliability(trimap_stats, warnings)
    alpha, alpha_stats = estimate_alpha(display, trimap, matting_fn)
    global_result, contrast_info = contrast_mask_approximation(linear, strength, blur_percent)
    localized_result = blend_with_alpha(linear, global_result, alpha)
    encoded, output_clipping = encode_output(localized_result, representation, log_min_log2, log_max_log2)
    if output_clipping["out_of_range_samples"]:
        warnings.append("Some output samples are outside the representable range and will be clipped")
    profile_out = _output_profile(representation, profile)

    _write_tiff(output_path, encoded, profile_out)
    save_gray_png(output_directory / "sky_probability.png", probability)
    save_gray_png(output_directory / "sky_guided_probability.png", refined)
    _module("PIL.Image", "").fromarray(trimap_png, mode="L").save(output_directory / "sky_trimap.png")
    save_gray_png(output_directory / "sky_mask.png", alpha)
    save_overlay(output_directory / "mask_overlay.png", display, alpha)
    save_comparison(output_directory / "comparison.png", display, linear_to_srgb(global_result), linear_to_srgb(localized_result))

    report = {
        "model": {**model_info, "license": MODEL_LICENSE},
        "input": {**image_metadata, **input_info, "path": str(source_path)},
        "output": {"path": str(output_path), "representation": representation, "dtype": "uint16", "channels": 3, "mode": "localized"},
        "parameters": {
            "contrast_strength": strength,
            "blur_percent_short_side": blur_percent,
            "guided_radius": guided_radius,
            "guided_epsilon": guided_epsilon,
            "trimap_sky_threshold": sky_threshold,
            "trimap_foreground_threshold": foreground_threshold,
            "log_min_log2": log_min_log2,
            "log_max_log2": log_max_log2,
        },
        "contrast_mask": {
            "method": "positive-image pseudo-density approximation",
            "reference_formula": "D' = D - g * blur(D) + const",
            "assumption": "pseudo-density is -log10(linear sRGB luminance); its 1st-to-99th percentile range sets exposure strength",
            "working_space": "linear RGB with sRGB primaries",
            **contrast_info,
        },
        "trimap": trimap_stats,
        "alpha": {"minimum": float(alpha.min()), "maximum": float(alpha.max()), **alpha_stats},
        "clipping": {
            "global_linear": clipping_stats(global_result),
            "localized_linear": clipping_stats(localized_result),
            "encoded_output": output_clipping,
            "policy": "report out-of-range values, then clip only when encoding the 16-bit TIFF",
        },
        "warnings": warnings,
    }
    with (output_directory / "report.json").open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, allow_nan=False)
        report_file.write("\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a full-resolution sky matte and apply a localized contrast-mask approximation.")
    parser.add_argument("input_tiff", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--representation", choices=("linear", "srgb", "a98c", "log"), required=True)
    parser.add_argument("--model", type=Path, help="Optional local ONNX model override; defaults to the pinned Hugging Face model")
    parser.add_argument("--strength", type=float, default=0.25, help="Approximation strength from -0.5 to 0.5; zero disables it")
    parser.add_argument("--blur-percent", type=float, default=4.0, help="Gaussian blur sigma as percent of the short image side (2-6)")
    parser.add_argument("--guided-radius", type=int, default=12)
    parser.add_argument("--guided-epsilon", type=float, default=1e-3)
    parser.add_argument("--sky-threshold", type=float, default=0.95)
    parser.add_argument("--foreground-threshold", type=float, default=0.05)
    parser.add_argument("--log-min-log2", type=float, help="Log input value represented by sample 0")
    parser.add_argument("--log-max-log2", type=float, help="Log input value represented by sample 65535")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.representation == "log" and (args.log_min_log2 is None or args.log_max_log2 is None):
            raise ValueError("Log input requires --log-min-log2 and --log-max-log2")
        run_pipeline(
            args.input_tiff,
            args.output_directory,
            args.representation,
            args.model,
            strength=args.strength,
            blur_percent=args.blur_percent,
            guided_radius=args.guided_radius,
            guided_epsilon=args.guided_epsilon,
            sky_threshold=args.sky_threshold,
            foreground_threshold=args.foreground_threshold,
            log_min_log2=args.log_min_log2,
            log_max_log2=args.log_max_log2,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
