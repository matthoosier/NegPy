import json

import numpy as np
import pytest
import tifffile

from scripts import sky_contrast_mask as sky


class _ModelInput:
    name = "pixel_values"
    shape = [1, 3, 16, 16]
    type = "tensor(float)"


class _SymbolicModelInput(_ModelInput):
    shape = ["batch_size", "num_channels", "height", "width"]


class _FakeSession:
    def get_inputs(self):
        return [_ModelInput()]

    def run(self, _outputs, feeds):
        assert feeds["pixel_values"].shape == (1, 3, 16, 16)
        logits = np.full((1, 150, 4, 4), -12.0, dtype=np.float32)
        logits[:, 2, :2, :] = 12.0
        logits[:, 4, :2, :] = -12.0
        logits[:, 0, 2:, :] = 12.0
        logits[:, 2, 2:, :] = -12.0
        return [logits]


def test_model_input_size_accepts_symbolic_nchw_dimensions():
    class Session:
        def get_inputs(self):
            return [_SymbolicModelInput()]

    assert sky.model_input_size(Session()) == (None, None)


def test_model_input_size_rejects_fixed_non_rgb_channel_count():
    class Session:
        def get_inputs(self):
            model_input = _ModelInput()
            model_input.shape = [1, 4, 16, 16]
            return [model_input]

    with pytest.raises(ValueError, match="NCHW RGB"):
        sky.model_input_size(Session())


def test_srgb_linear_round_trip():
    values = np.linspace(0.0, 1.0, 101)
    np.testing.assert_allclose(sky.linear_to_srgb(sky.srgb_to_linear(values)), values, atol=1e-12)


def test_log_representation_uses_explicit_log2_mapping():
    samples = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
    linear, display, metadata = sky.prepare_input(samples, "log", -2.0, 2.0)
    np.testing.assert_allclose(linear[0, 0], [0.25, 1.0, 4.0], atol=1e-4)
    assert display.shape == samples.shape
    assert metadata["log_display_mapping"] == {"min_log2": -2.0, "max_log2": 2.0}


def test_log_representation_requires_valid_mapping():
    samples = np.zeros((2, 2, 3), dtype=np.uint16)
    with pytest.raises(ValueError, match="requires --log-min-log2"):
        sky.prepare_input(samples, "log")
    with pytest.raises(ValueError, match="must be below"):
        sky.prepare_input(samples, "log", 1.0, 1.0)


def test_a98c_profile_conversion_and_output_round_trip(monkeypatch):
    samples = np.array(
        [[[32768, 32768, 32768], [65535, 10000, 42000], [1234, 50000, 32100]]],
        dtype=np.uint16,
    )
    profile = bytearray(128)
    profile[48:52] = b"A98C"
    monkeypatch.setattr(sky, "_profile_description", lambda _profile: "Adobe RGB (1998)")

    linear, display, metadata = sky.prepare_input(samples, "a98c", profile=bytes(profile))
    expected_neutral = (32768 / 65535.0) ** sky.A98_GAMMA
    np.testing.assert_allclose(linear[0, 0], expected_neutral, atol=2e-5)
    assert display.shape == samples.shape
    assert np.isfinite(display).all()
    assert metadata["input_primaries"] == "Adobe RGB (1998)"
    assert "converted to linear sRGB" in metadata["icc_conversion"]

    encoded, _ = sky.encode_output(linear, "a98c", None, None)
    np.testing.assert_allclose(encoded, samples, atol=2)


def test_a98c_representation_requires_matching_embedded_profile(monkeypatch):
    samples = np.full((1, 1, 3), 32768, dtype=np.uint16)
    with pytest.raises(ValueError, match="requires an embedded"):
        sky.prepare_input(samples, "a98c")
    monkeypatch.setattr(sky, "_profile_description", lambda _profile: "Display P3")
    with pytest.raises(ValueError, match="not an A98C"):
        sky.prepare_input(samples, "a98c", profile=b"profile")


def test_a98c_profile_signature_is_read_from_icc_manufacturer_or_model():
    profile = bytearray(128)
    profile[48:52] = b"A98C"
    assert sky._a98_profile_matches(bytes(profile), "")
    profile[48:52] = b"\0\0\0\0"
    profile[52:56] = b"A98C"
    assert sky._a98_profile_matches(bytes(profile), "")


def test_trimap_has_three_values_and_reports_regions():
    probability = np.array([[0.01, 0.5, 0.99]], dtype=np.float32)
    trimap, diagnostic, stats = sky.make_trimap(probability, 0.95, 0.05)
    np.testing.assert_array_equal(trimap, [[0.0, 0.5, 1.0]])
    np.testing.assert_array_equal(diagnostic, [[0, 128, 255]])
    assert stats["unknown_pixels"] == 1
    assert stats["definite_sky_fraction"] == pytest.approx(1 / 3)


def test_trimap_rejects_missing_known_class():
    with pytest.raises(ValueError, match="at least one definite-sky"):
        sky.make_trimap(np.full((2, 2), 0.5), 0.95, 0.05)


def test_alpha_is_finite_bounded_and_respects_known_regions():
    trimap = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    image = np.zeros((1, 3, 3), dtype=np.float32)
    alpha, _ = sky.estimate_alpha(image, trimap, lambda *_: np.array([[-0.2, 0.4, 1.2]]))
    np.testing.assert_allclose(alpha, [[0.0, 0.4, 1.0]])


def test_zero_strength_is_identity_and_nonzero_strength_changes_detail():
    ramp = np.linspace(0.1, 0.9, 64, dtype=np.float64)
    image = np.broadcast_to(ramp[None, :, None], (64, 64, 3)).copy()
    identity, _ = sky.contrast_mask_approximation(image, 0.0)
    np.testing.assert_array_equal(identity, image)
    adjusted, details = sky.contrast_mask_approximation(image, 0.25, 4.0)
    assert np.isfinite(adjusted).all()
    assert not np.allclose(adjusted, image)
    assert details["sigma_pixels"] == pytest.approx(2.56)


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_alpha_blend_zero_and_full(alpha):
    original = np.full((3, 4, 3), 0.25)
    adjusted = np.full_like(original, 0.75)
    result = sky.blend_with_alpha(original, adjusted, np.full((3, 4), alpha))
    np.testing.assert_array_equal(result, adjusted if alpha else original)


def test_alpha_blend_preserves_pixels_outside_mask():
    original = np.full((2, 3, 3), 0.2)
    adjusted = np.full_like(original, 0.8)
    alpha = np.array([[0.0, 0.0, 1.0], [0.0, 0.5, 0.0]])
    result = sky.blend_with_alpha(original, adjusted, alpha)
    np.testing.assert_array_equal(result[alpha == 0.0], original[alpha == 0.0])


def test_output_encoding_is_uint16_rgb_and_reports_clipping():
    values = np.array([[[-0.1, 0.5, 1.2]]])
    encoded, stats = sky.encode_output(values, "linear", None, None)
    assert encoded.dtype == np.uint16
    assert encoded.shape == (1, 1, 3)
    np.testing.assert_array_equal(encoded[0, 0], [0, 32768, 65535])
    assert stats["out_of_range_samples"] == 2


def test_reads_and_rejects_non_rgb_tiff(tmp_path):
    path = tmp_path / "gray.tif"
    tifffile.imwrite(path, np.zeros((4, 5), dtype=np.uint16), photometric="minisblack")
    with pytest.raises(ValueError, match="must be RGB"):
        sky.read_rgb_tiff(path)


def test_run_pipeline_writes_localized_tiff_and_diagnostics(tmp_path):
    input_path = tmp_path / "input.tif"
    output_dir = tmp_path / "result"
    values = np.full((8, 8, 3), 32768, dtype=np.uint16)
    tifffile.imwrite(input_path, values, photometric="rgb", metadata=None)

    report = sky.run_pipeline(
        input_path,
        output_dir,
        "srgb",
        tmp_path / "unused-model.onnx",
        strength=0.0,
        sky_threshold=0.52,
        foreground_threshold=0.48,
        session=_FakeSession(),
        matting_fn=lambda _image, trimap: trimap,
    )

    output = tifffile.imread(output_dir / "output.tif")
    assert output.shape == values.shape
    assert output.dtype == np.uint16
    assert report["output"]["mode"] == "localized"
    assert report["input"]["representation"] == "srgb"
    assert report["model"]["sky_class_index"] == 2
    assert (output_dir / "report.json").is_file()
    loaded_report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert loaded_report["output"]["dtype"] == "uint16"
    for name in (
        "sky_probability.png",
        "sky_guided_probability.png",
        "sky_trimap.png",
        "sky_mask.png",
        "mask_overlay.png",
        "comparison.png",
    ):
        assert (output_dir / name).is_file()


def test_model_session_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        sky._create_session(tmp_path / "missing.onnx")


def test_resolve_model_path_downloads_pinned_hugging_face_artifact(tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls = {}
    model_path = tmp_path / "model.onnx"
    model_path.touch()

    def download(**kwargs):
        calls.update(kwargs)
        return str(model_path)

    monkeypatch.setattr(sky, "_module", lambda name, _hint: SimpleNamespace(hf_hub_download=download))
    assert sky.resolve_model_path() == model_path
    assert calls == {
        "repo_id": "nvidia/segformer-b0-finetuned-ade-512-512",
        "filename": "onnx/model.onnx",
        "revision": "b9175de73a0a34f7843135853d27629aa6987b2f",
    }


def test_resolve_model_path_keeps_local_override(tmp_path):
    model_path = tmp_path / "custom.onnx"
    model_path.touch()
    assert sky.resolve_model_path(model_path) == model_path.resolve()


def test_model_session_reports_missing_onnx_runtime(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fixture")
    original_import = sky.importlib.import_module

    def missing_runtime(name):
        if name == "onnxruntime":
            raise ModuleNotFoundError(name)
        return original_import(name)

    monkeypatch.setattr(sky.importlib, "import_module", missing_runtime)
    with pytest.raises(RuntimeError, match="onnxruntime is required"):
        sky._create_session(model_path)
