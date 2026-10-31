"""Transient sky segmentation and edge refinement for the matte preview."""

from __future__ import annotations

import os
import threading
import urllib.request
import hashlib
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Optional

import cv2
import numpy as np

from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.kernel.system.version import get_app_version

logger = get_logger(__name__)

MODEL_ID = "Xenova/segformer-b0-finetuned-ade-512-512"
MODEL_REVISION = "d3e5499fa8701ff0453ca940a8dfeae39b2f1504"
MODEL_VERSION = "segformer-b0-ade-512-512-v1"
MODEL_SIZE = 512
SKY_CLASS_ID = 2
GUIDED_FILTER_RADIUS = 12
GUIDED_FILTER_EPSILON = 1e-3
TRIMAP_SKY_THRESHOLD = 0.95
TRIMAP_FOREGROUND_THRESHOLD = 0.05

_MODEL_FILE = "model_quantized.onnx"
_MODEL_URL = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/onnx/{_MODEL_FILE}"
_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
_USER_AGENT = f"NegPy/{get_app_version()} (+https://github.com/marcinz606/NegPy)"
_CHUNK_SIZE = 256 * 1024
_DOWNLOAD_LOCK = threading.Lock()
_MATTE_CACHE_VERSION = 1
_MATTE_CACHE_LOCK = threading.Lock()
_MATTE_CACHE_MEMORY: OrderedDict[str, np.ndarray] = OrderedDict()
_MATTE_CACHE_LIMIT = 8
_MATTE_FUTURES: dict[str, tuple[Future[np.ndarray], list[Callable[[str, Optional[Exception]], None]]]] = {}
_MATTE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sky-matte")
_MATTE_MODEL = None


class SkyMatteError(RuntimeError):
    """The sky model could not be fetched or its output could not be processed."""


def _model_path() -> str:
    return os.path.join(APP_CONFIG.cache_dir, "sky_matte", MODEL_VERSION, _MODEL_FILE)


def sky_model_ready() -> bool:
    return os.path.isfile(_model_path())


def download_sky_model(
    on_progress: Optional[Callable[[int, int], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> None:
    """Download the pinned ONNX export atomically into the application cache."""
    with _DOWNLOAD_LOCK:
        destination = _model_path()
        if os.path.isfile(destination):
            return
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        temporary = f"{destination}.part"
        request = urllib.request.Request(_MODEL_URL, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=30) as response, open(temporary, "wb") as output:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    if is_cancelled is not None and is_cancelled():
                        raise SkyMatteError("Sky matte processing was canceled.")
                    chunk = response.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    done += len(chunk)
                    if on_progress is not None:
                        on_progress(done, total)
            os.replace(temporary, destination)
        except SkyMatteError:
            _discard(temporary)
            raise
        except Exception as exc:
            _discard(temporary)
            raise SkyMatteError(f"Could not download the sky model: {exc}") from exc


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except OSError:
        logger.exception("Could not remove partial sky model download: %s", path)


def _preprocess(image_rgb: np.ndarray, size: tuple[int, int] = (MODEL_SIZE, MODEL_SIZE)) -> np.ndarray:
    """Resize RGB float pixels to (height, width) and normalize them."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or not image_rgb.size:
        raise SkyMatteError("The sky model needs a non-empty RGB frame.")
    resized = cv2.resize(
        np.ascontiguousarray(image_rgb, dtype=np.float32),
        (size[1], size[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    normalized = (resized - _MEAN) / _STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def _sky_probability(logits: np.ndarray) -> np.ndarray:
    """Extract the ADE20K sky class from NCHW or NHWC SegFormer logits."""
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 4 or values.shape[0] != 1:
        raise SkyMatteError(f"Unexpected SegFormer output shape: {values.shape}")
    if values.shape[1] == 150:
        classes = np.moveaxis(values[0], 0, -1)
    elif values.shape[-1] == 150:
        classes = values[0]
    else:
        raise SkyMatteError(f"SegFormer output has no sky class: {values.shape}")
    classes = classes - np.max(classes, axis=-1, keepdims=True)
    probabilities = np.exp(classes)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    if not np.isfinite(probabilities).all():
        raise SkyMatteError("SegFormer returned non-finite class probabilities.")
    return np.ascontiguousarray(probabilities[..., SKY_CLASS_ID], dtype=np.float32)


def _infer_sky_probability(image_rgb: np.ndarray, session) -> np.ndarray:
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise SkyMatteError(f"Expected one SegFormer input, found {len(inputs)}.")
    input_info = inputs[0]
    if getattr(input_info, "type", "tensor(float)") != "tensor(float)":
        raise SkyMatteError("SegFormer input must use float32.")
    shape = input_info.shape
    if len(shape) != 4 or shape[1] not in (3, "num_channels"):
        raise SkyMatteError(f"SegFormer input must use NCHW RGB layout, found {shape}.")
    height, width = image_rgb.shape[:2]
    model_height, model_width = shape[2], shape[3]
    if isinstance(model_height, int) and isinstance(model_width, int):
        input_size = model_height, model_width
    elif not isinstance(model_height, int) and not isinstance(model_width, int):
        scale = MODEL_SIZE / min(height, width)
        input_size = max(1, round(height * scale)), max(1, round(width * scale))
    else:
        raise SkyMatteError(f"SegFormer input must have both fixed or both dynamic spatial dimensions, found {shape}.")
    outputs = session.run(None, {input_info.name: _preprocess(image_rgb, input_size)})
    if not outputs:
        raise SkyMatteError("SegFormer returned no output.")
    probability = _sky_probability(outputs[0])
    return cv2.resize(probability, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)


class SkyMatteModel:
    """Lazy ONNX Runtime session that runs only on the CPU provider."""

    def __init__(self) -> None:
        self._session = None

    def _ensure_session(
        self,
        on_progress: Optional[Callable[[int, int], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ):
        if self._session is None:
            if not sky_model_ready():
                download_sky_model(on_progress, is_cancelled)
            import onnxruntime as ort

            try:
                self._session = ort.InferenceSession(_model_path(), providers=["CPUExecutionProvider"])
            except Exception as exc:
                raise SkyMatteError(f"Could not load the sky model: {exc}") from exc
        return self._session

    def sky_probability(
        self,
        image_rgb: np.ndarray,
        on_progress: Optional[Callable[[int, int], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> np.ndarray:
        try:
            if is_cancelled is not None and is_cancelled():
                raise SkyMatteError("Sky matte processing was canceled.")
            return _infer_sky_probability(image_rgb, self._ensure_session(on_progress, is_cancelled))
        except SkyMatteError:
            raise
        except Exception as exc:
            logger.exception("Sky segmentation failed")
            raise SkyMatteError(f"Could not segment the sky: {exc}") from exc


def make_sky_matte(
    image_rgb: np.ndarray,
    model: SkyMatteModel,
    is_cancelled: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Return a full-resolution float32 sky alpha matte for a positive RGB frame."""
    if is_cancelled is not None and is_cancelled():
        raise SkyMatteError("Sky matte processing was canceled.")
    guide = np.ascontiguousarray(np.clip(image_rgb, 0.0, 1.0), dtype=np.float32)
    sky = model.sky_probability(guide, on_progress, is_cancelled)
    if is_cancelled is not None and is_cancelled():
        raise SkyMatteError("Sky matte processing was canceled.")
    try:
        refined = cv2.ximgproc.guidedFilter(guide, sky, GUIDED_FILTER_RADIUS, GUIDED_FILTER_EPSILON, dDepth=-1)
    except AttributeError as exc:
        raise SkyMatteError("OpenCV contrib with ximgproc.guidedFilter is required for sky mattes.") from exc
    except Exception as exc:
        logger.exception("Guided sky matte refinement failed")
        raise SkyMatteError(f"Could not refine the sky matte: {exc}") from exc
    if refined.shape != sky.shape or not np.isfinite(refined).all():
        raise SkyMatteError("Guided filter returned invalid sky probability data.")
    if is_cancelled is not None and is_cancelled():
        raise SkyMatteError("Sky matte processing was canceled.")
    trimap = _make_trimap(np.clip(refined, 0.0, 1.0))
    try:
        alpha = _estimate_alpha(guide, trimap)
    except Exception as exc:
        logger.exception("Closed-form sky matte refinement failed")
        raise SkyMatteError(f"Could not refine the sky matte: {exc}") from exc
    if alpha.shape != trimap.shape or not np.isfinite(alpha).all():
        raise SkyMatteError(f"Closed-form matting returned invalid alpha data: {alpha.shape}")
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha[trimap >= 1.0] = 1.0
    alpha[trimap <= 0.0] = 0.0
    return np.ascontiguousarray(alpha, dtype=np.float32)


def rough_positive_proxy(normalized_log: np.ndarray) -> np.ndarray:
    """Invert the normalized negative into a bounded positive proxy."""
    if normalized_log.ndim != 3 or normalized_log.shape[2] < 3 or not normalized_log.size:
        raise SkyMatteError("The normalized negative must be a non-empty RGB frame.")
    return np.ascontiguousarray(np.clip(1.0 - normalized_log[:, :, :3], 0.0, 1.0), dtype=np.float32)


def sky_matte_cache_key(image_rgb: np.ndarray, identity: object = None) -> str:
    """Key the derived matte by the exact guide pixels and all output-affecting versions."""
    image = np.ascontiguousarray(image_rgb, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str((_MATTE_CACHE_VERSION, MODEL_VERSION, GUIDED_FILTER_RADIUS, GUIDED_FILTER_EPSILON)).encode())
    digest.update(repr(identity).encode())
    digest.update(repr((image.shape, image.dtype.str)).encode())
    digest.update(memoryview(image).cast("B"))
    return digest.hexdigest()


def _matte_path(key: str) -> str:
    return os.path.join(APP_CONFIG.cache_dir, "sky_matte", "mattes", str(_MATTE_CACHE_VERSION), f"{key}.npy")


def _remember_matte(key: str, matte: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(matte, dtype=np.float32)
    with _MATTE_CACHE_LOCK:
        _MATTE_CACHE_MEMORY[key] = result
        _MATTE_CACHE_MEMORY.move_to_end(key)
        while len(_MATTE_CACHE_MEMORY) > _MATTE_CACHE_LIMIT:
            _MATTE_CACHE_MEMORY.popitem(last=False)
    return result


def load_cached_sky_matte(key: str, shape: tuple[int, int]) -> Optional[np.ndarray]:
    """Load a validated matte from memory or the disposable application cache."""
    with _MATTE_CACHE_LOCK:
        matte = _MATTE_CACHE_MEMORY.get(key)
        if matte is not None:
            _MATTE_CACHE_MEMORY.move_to_end(key)
            if matte.shape == shape and np.isfinite(matte).all():
                return matte
            _MATTE_CACHE_MEMORY.pop(key, None)

    path = _matte_path(key)
    try:
        matte = np.load(path, allow_pickle=False)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("Ignoring invalid cached sky matte %s", path, exc_info=True)
        return None
    if matte.dtype != np.float32 or matte.shape != shape or not np.isfinite(matte).all():
        logger.warning("Ignoring incompatible cached sky matte %s", path)
        return None
    return _remember_matte(key, np.clip(matte, 0.0, 1.0))


def _make_cached_sky_matte(key: str, image_rgb: np.ndarray) -> np.ndarray:
    global _MATTE_MODEL
    cached = load_cached_sky_matte(key, image_rgb.shape[:2])
    if cached is not None:
        return cached
    if _MATTE_MODEL is None:
        _MATTE_MODEL = SkyMatteModel()
    matte = make_sky_matte(image_rgb, _MATTE_MODEL)
    matte = _remember_matte(key, matte)
    path = _matte_path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.{threading.get_ident()}.part"
    try:
        with open(temporary, "wb") as output:
            np.save(output, matte, allow_pickle=False)
        os.replace(temporary, path)
    except OSError:
        _discard(temporary)
        logger.exception("Could not save cached sky matte %s", path)
    return matte


def get_sky_matte(
    image_rgb: np.ndarray,
    key: str,
    wait: bool,
    on_status: Optional[Callable[[str, Optional[Exception]], None]] = None,
) -> Optional[np.ndarray]:
    """Return a cached matte, wait for one, or queue one background inference."""
    image = np.ascontiguousarray(np.clip(image_rgb, 0.0, 1.0), dtype=np.float32)
    cached = load_cached_sky_matte(key, image.shape[:2])
    if cached is not None:
        if on_status is not None:
            on_status("done", None)
        return cached
    watch_future = False
    with _MATTE_CACHE_LOCK:
        pending = _MATTE_FUTURES.get(key)
        if pending is None:
            future = _MATTE_EXECUTOR.submit(_make_cached_sky_matte, key, image.copy())
            callbacks = [on_status] if on_status is not None else []
            _MATTE_FUTURES[key] = (future, callbacks)
            pending = _MATTE_FUTURES[key]
            watch_future = True
        elif on_status is not None:
            pending[1].append(on_status)
        if on_status is not None:
            on_status("calculating", None)
    if watch_future:
        pending[0].add_done_callback(lambda completed, cache_key=key: _finish_sky_matte(cache_key, completed))
    if not wait:
        return None
    return pending[0].result()


def _finish_sky_matte(key: str, future: Future[np.ndarray]) -> None:
    with _MATTE_CACHE_LOCK:
        pending = _MATTE_FUTURES.pop(key, None)
    if pending is None:
        return
    try:
        future.result()
    except Exception as exc:
        logger.exception("Background sky matte generation failed")
        for callback in pending[1]:
            callback("failed", exc)
    else:
        for callback in pending[1]:
            callback("done", None)


def _make_trimap(probability: np.ndarray) -> np.ndarray:
    """Mark confident sky, uncertain pixels, and definite foreground."""
    if not np.isfinite(probability).all():
        raise SkyMatteError("Sky probability contains non-finite values.")
    definite_sky = probability >= TRIMAP_SKY_THRESHOLD
    definite_foreground = probability <= TRIMAP_FOREGROUND_THRESHOLD
    if not definite_sky.any() or not definite_foreground.any():
        raise SkyMatteError("Sky probabilities must contain definite sky and foreground regions.")
    trimap = np.full(probability.shape, 0.5, dtype=np.float32)
    trimap[definite_sky] = 1.0
    trimap[definite_foreground] = 0.0
    return trimap


def _estimate_alpha(guide: np.ndarray, trimap: np.ndarray) -> np.ndarray:
    from pymatting import estimate_alpha_cf

    return np.asarray(
        estimate_alpha_cf(
            np.ascontiguousarray(guide, dtype=np.float64),
            np.ascontiguousarray(trimap, dtype=np.float64),
        ),
        dtype=np.float32,
    )
