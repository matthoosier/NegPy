from types import SimpleNamespace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pytest

from negpy.services.assets import sky_matte


class _FakeSession:
    def __init__(self, logits, input_shape=(1, 3, "height", "width")):
        self.logits = logits
        self.feed: dict | None = None
        self.input_shape = input_shape

    def get_inputs(self):
        return [SimpleNamespace(name="pixel_values", shape=self.input_shape, type="tensor(float)")]

    def run(self, _outputs, feed):
        self.feed = feed
        return [self.logits]


def test_preprocess_resizes_and_normalizes_rgb():
    image = np.full((32, 64, 3), 0.5, dtype=np.float32)

    result = sky_matte._preprocess(image)

    assert result.shape == (1, 3, 512, 512)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0, :, 256, 256], (0.5 - sky_matte._MEAN) / sky_matte._STD, atol=1e-6)


def test_inference_extracts_soft_sky_probability_and_resizes():
    logits = np.zeros((1, 150, 8, 8), dtype=np.float32)
    logits[:, 2] = 4.0
    session = _FakeSession(logits)
    image = np.full((24, 40, 3), 0.5, dtype=np.float32)

    result = sky_matte._infer_sky_probability(image, session)

    assert session.feed is not None
    assert session.feed["pixel_values"].shape == (1, 3, 512, 853)
    assert result.shape == (24, 40)
    assert result.dtype == np.float32
    assert float(result.mean()) == pytest.approx(np.exp(4.0) / (np.exp(4.0) + 149.0), rel=1e-5)
    assert np.all((result > 0.0) & (result < 1.0))


class _FakeSkyModel(sky_matte.SkyMatteModel):
    def __init__(self, probability: np.ndarray) -> None:
        super().__init__()
        self.probability = probability

    def sky_probability(
        self,
        image_rgb: np.ndarray,
        on_progress: Callable[[int, int], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> np.ndarray:
        return self.probability


def test_sky_probability_accepts_nhwc_logits():
    logits = np.zeros((1, 4, 5, 150), dtype=np.float32)
    logits[..., 2] = 3.0

    result = sky_matte._sky_probability(logits)

    assert result.shape == (4, 5)
    assert np.all(result > 0.0)


def test_sky_probability_rejects_unexpected_output_shape():
    with pytest.raises(sky_matte.SkyMatteError, match="Unexpected SegFormer output shape"):
        sky_matte._sky_probability(np.zeros((1, 4, 5), dtype=np.float32))


def test_guided_filter_and_alpha_matte_use_trimapped_probability(monkeypatch):
    image = np.full((1024, 512, 3), 0.5, dtype=np.float32)
    sky = np.full((1024, 512), 0.5, dtype=np.float32)
    sky[:, :100] = 0.99
    sky[:, -100:] = 0.01
    model = _FakeSkyModel(sky)
    calls = {}

    def guided_filter(guide, source, radius, epsilon, dDepth):
        calls.update(guide=guide, source=source, radius=radius, epsilon=epsilon, dDepth=dDepth)
        return source

    def estimate_alpha(guide, trimap):
        calls.update(alpha_guide=guide, trimap=trimap)
        return np.full(trimap.shape, 0.6, dtype=np.float32)

    monkeypatch.setattr(cv2, "ximgproc", SimpleNamespace(guidedFilter=guided_filter), raising=False)
    monkeypatch.setattr(sky_matte, "_estimate_alpha", estimate_alpha)

    result = sky_matte.make_sky_matte(image, model)

    assert calls["guide"] is image or np.array_equal(calls["guide"], image)
    assert calls["source"] is sky
    assert calls["radius"] == 12
    assert calls["epsilon"] == sky_matte.GUIDED_FILTER_EPSILON
    assert calls["dDepth"] == -1
    assert calls["alpha_guide"].dtype == np.float32
    np.testing.assert_array_equal(calls["trimap"][:, :100], 1.0)
    np.testing.assert_array_equal(calls["trimap"][:, 100:-100], 0.5)
    np.testing.assert_array_equal(calls["trimap"][:, -100:], 0.0)
    assert result.dtype == np.float32
    assert result.shape == (1024, 512)
    np.testing.assert_array_equal(result[:, :100], 1.0)
    np.testing.assert_allclose(result[:, 100:-100], 0.6)
    np.testing.assert_array_equal(result[:, -100:], 0.0)


def test_make_sky_matte_runs_guided_filter_and_closed_form_alpha():
    height, width = 32, 64
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[:, :32] = (0.1, 0.3, 0.8)
    image[:, 32:] = (0.2, 0.2, 0.2)
    probability = np.full((height, width), 0.5, dtype=np.float32)
    probability[:, :24] = 0.99
    probability[:, 40:] = 0.01

    result = sky_matte.make_sky_matte(image, _FakeSkyModel(probability))

    assert result.shape == (height, width)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert result.min() == 0.0
    assert result.max() == 1.0
    assert np.count_nonzero((result > 0.0) & (result < 1.0)) > 0


def test_rough_positive_proxy_inverts_and_clamps_normalized_log():
    normalized_log = np.array(
        [
            [[-0.2, 0.0, 0.25], [0.5, 0.75, 1.0], [1.1, 0.4, -0.1]],
        ],
        dtype=np.float32,
    )

    result = sky_matte.rough_positive_proxy(normalized_log)

    np.testing.assert_allclose(result, [[[1.0, 1.0, 0.75], [0.5, 0.25, 0.0], [0.0, 0.6, 1.0]]])
    assert result.dtype == np.float32
    assert result.flags.c_contiguous


def test_compare_sky_mattes_maps_rough_to_red_and_final_to_green():
    final = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    rough = np.array([[0.25, 0.5, 0.0]], dtype=np.float32)

    result = sky_matte.compare_sky_mattes(final, rough)

    np.testing.assert_allclose(result, [[[0.25, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0]]])


def test_make_sky_matte_comparison_runs_both_inputs_and_builds_overlap(monkeypatch):
    final = np.full((2, 3, 3), 0.4, dtype=np.float32)
    normalized = np.full((2, 3, 3), 0.75, dtype=np.float32)
    observed = []

    def make_matte(image_rgb, *_args):
        observed.append(image_rgb.copy())
        alpha = np.zeros(image_rgb.shape[:2], dtype=np.float32)
        alpha[:, :2] = 0.8 if image_rgb.mean() > 0.3 else 0.6
        return alpha

    monkeypatch.setattr(sky_matte, "make_sky_matte", make_matte)
    model = _FakeSkyModel(np.zeros((2, 3), dtype=np.float32))
    final_matte, rough_matte, comparison = sky_matte.make_sky_matte_comparison(final, normalized, model)

    np.testing.assert_allclose(observed[0], final)
    np.testing.assert_allclose(observed[1], 0.25)
    np.testing.assert_allclose(final_matte[:, :2], 0.8)
    np.testing.assert_allclose(rough_matte[:, :2], 0.6)
    np.testing.assert_allclose(comparison[:, :2, 0], 0.6)
    np.testing.assert_allclose(comparison[:, :2, 1], 0.8)
    np.testing.assert_array_equal(comparison[:, :, 2], 0.0)


def test_inference_uses_fixed_model_dimensions_when_declared():
    logits = np.zeros((1, 150, 8, 8), dtype=np.float32)
    session = _FakeSession(logits, input_shape=(1, 3, 512, 512))
    image = np.full((24, 40, 3), 0.5, dtype=np.float32)

    sky_matte._infer_sky_probability(image, session)

    assert session.feed is not None
    assert session.feed["pixel_values"].shape == (1, 3, 512, 512)


def test_inference_rejects_partially_dynamic_model_dimensions():
    session = _FakeSession(np.zeros((1, 150, 8, 8), dtype=np.float32), input_shape=(1, 3, 512, "width"))

    with pytest.raises(sky_matte.SkyMatteError, match="both fixed or both dynamic"):
        sky_matte._infer_sky_probability(np.zeros((24, 40, 3), dtype=np.float32), session)


def test_trimap_marks_threshold_regions():
    probabilities = np.array([[0.05, 0.5, 0.95]], dtype=np.float32)

    result = sky_matte._make_trimap(probabilities)

    np.testing.assert_array_equal(result, [[0.0, 0.5, 1.0]])


def test_trimap_rejects_missing_confident_class():
    with pytest.raises(sky_matte.SkyMatteError, match="definite sky and foreground"):
        sky_matte._make_trimap(np.full((4, 4), 0.5, dtype=np.float32))


def test_model_session_uses_cpu_provider(monkeypatch, tmp_path):
    from negpy.kernel.system.config import APP_CONFIG

    monkeypatch.setattr(APP_CONFIG, "cache_dir", str(tmp_path))
    model_file = sky_matte._model_path()
    import os

    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    Path(model_file).touch()
    observed = {}

    class FakeOrt:
        @staticmethod
        def InferenceSession(path, providers):
            observed.update(path=path, providers=providers)
            return object()

    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", FakeOrt)

    session = sky_matte.SkyMatteModel()._ensure_session()

    assert session is not None
    assert observed["path"] == model_file
    assert observed["providers"] == ["CPUExecutionProvider"]


def test_model_download_is_atomic_and_reports_progress(monkeypatch, tmp_path):
    model_file = tmp_path / "sky" / "model.onnx"
    monkeypatch.setattr(sky_matte, "_model_path", lambda: str(model_file))
    progress = []

    class Response:
        headers = {"Content-Length": "4"}

        def __init__(self):
            self.read_once = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size):
            if self.read_once:
                return b""
            self.read_once = True
            return b"data"

    monkeypatch.setattr(sky_matte.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())

    sky_matte.download_sky_model(lambda done, total: progress.append((done, total)))

    assert model_file.read_bytes() == b"data"
    assert progress == [(4, 4)]
    assert not Path(f"{model_file}.part").exists()


def test_cancelled_model_download_removes_partial_file(monkeypatch, tmp_path):
    model_file = tmp_path / "sky" / "model.onnx"
    monkeypatch.setattr(sky_matte, "_model_path", lambda: str(model_file))

    class Response:
        headers = {"Content-Length": "4"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size):
            return b"data"

    monkeypatch.setattr(sky_matte.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())

    with pytest.raises(sky_matte.SkyMatteError, match="canceled"):
        sky_matte.download_sky_model(is_cancelled=lambda: True)

    assert not model_file.exists()
    assert not Path(f"{model_file}.part").exists()
