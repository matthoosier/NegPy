"""The Contrast Mask: an automatic tone mask after the Frontier's HyperTone.

Pins what would break silently: the sign, the fixed strength, resolution invariance,
the crop boundary, the migration and CPU/GPU parity.
"""

import unittest
from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import MASK_AIM_CODE, MASK_CODES_PER_DENSITY, MASK_MID_CODE, contrast_mask_coeffs, contrast_mask_ev
from negpy.features.exposure.normalization import LogNegativeBounds, contrast_mask_plane, contrast_mask_plane_with_input
from negpy.features.exposure.normalization import contrast_mask_matte_plane
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.services.rendering.engine import DarkroomEngine


def _wide_range_negative(h: int = 240, w: int = 360) -> np.ndarray:
    """A negative whose subject range runs far past the paper scale, with fine texture
    at every density so micro-contrast is measurable in each third."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    key = np.exp(-3.2 * (yy / h))
    texture = 1.0 + 0.16 * np.sin(xx / 3.0) * np.sin(yy / 2.5)
    scene = np.clip(key * texture, 1e-5, None)
    scene /= scene.max()
    neg = np.clip(0.03 + 0.85 * (1.0 - scene**0.35), 1e-4, 1.0)
    return np.ascontiguousarray(np.dstack([neg, neg, neg]).astype(np.float32))


def _flat_negative(h: int = 300, w: int = 450) -> np.ndarray:
    """A negative too flat for Grade: its log range puts the straight-line slope on the
    k floor at every ISO-R, so the grade slider has no travel left."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    key = np.exp(-0.9 * (yy / h))
    texture = 1.0 + 0.14 * np.sin(xx / 3.0) * np.sin(yy / 2.5)
    scene = np.clip(key * texture, 1e-5, None)
    scene /= scene.max()
    neg = np.clip(0.25 + 0.30 * (1.0 - scene**0.35), 1e-4, 1.0)
    return np.ascontiguousarray(np.dstack([neg, neg, neg]).astype(np.float32))


def _global_spread(plane: np.ndarray) -> float:
    """Spread of the broad tonal masses, with fine detail blurred away."""
    return float(cv2.GaussianBlur(plane, (0, 0), 12.0).std())


def _bw_settings(**exposure) -> WorkspaceConfig:
    s = WorkspaceConfig()
    return replace(
        s,
        process=replace(s.process, process_mode="B&W"),
        exposure=replace(s.exposure, auto_exposure=False, auto_normalize_contrast=False, **exposure),
    )


_BOUNDS = LogNegativeBounds(floors=(-1.4, -1.4, -1.4), ceils=(-0.05, -0.05, -0.05))
# Green's straight line at a mid grade: code 412 (mid-gray) sits at val 0.5 + _MID_OFFSET.
_SLOPE, _PIVOT, _RANGE = 1.6, 0.5, 1.4
_MID_OFFSET = ((MASK_MID_CODE - MASK_AIM_CODE) / MASK_CODES_PER_DENSITY + 0.75) / _SLOPE


def _coeffs(strength: float = 1.0):
    return contrast_mask_coeffs(strength, 0.5, (_SLOPE,) * 3, (_PIVOT,) * 3, (0.0,) * 3, _RANGE)


def _render(img: np.ndarray, settings: WorkspaceConfig, tag: str) -> np.ndarray:
    return np.asarray(DarkroomEngine().process(img.copy(), settings, tag))[:, :, 1]


class TestContrastMaskPlane(unittest.TestCase):
    def test_plane_is_zero_mean(self):
        """The plane carries only the surround's tone; its mean rides in the centre."""
        plane, _ = contrast_mask_plane(_wide_range_negative(), _BOUNDS, None)
        self.assertAlmostEqual(float(plane.mean()), 0.0, places=5)

    def test_surround_outside_the_crop_stays_out_of_the_mask(self):
        """A rebate or surround blurred into the mask prints as a vignette the negative
        does not have."""
        h, w = 600, 900
        img = np.full((h, w, 3), 0.35, dtype=np.float32)
        border = 60
        img[:border] = img[-border:] = img[:, :border] = img[:, -border:] = 0.98
        roi_norm = (border / h, (h - border) / h, border / w, (w - border) / w)

        cropped, _ = contrast_mask_plane(img, _BOUNDS, None, roi_norm=roi_norm)
        whole, _ = contrast_mask_plane(img, _BOUNDS, None)

        # A uniform picture area has no low frequencies, so its mask must be flat.
        self.assertLess(float(cropped.std()), 0.002, f"crop-respecting plane std {cropped.std():.5f}")
        # Without the crop the border bleeds in and manufactures a gradient.
        self.assertGreater(float(whole.std()), 10.0 * float(cropped.std()) + 0.01)


class TestContrastMaskCoeffs(unittest.TestCase):
    def test_off_at_zero_strength_or_without_a_density_range(self):
        self.assertIsNone(_coeffs(strength=0.0))
        self.assertIsNone(contrast_mask_coeffs(1.0, 0.5, (_SLOPE,) * 3, (_PIVOT,) * 3, (0.0,) * 3, 0.0))

    def test_both_sides_take_the_slider_on_every_frame(self):
        """One slider value sets both sides on every frame."""
        k = _coeffs(0.7)
        assert k is not None
        self.assertEqual(k[4], k[5])
        self.assertAlmostEqual(k[4], 0.7 * _RANGE / (MASK_CODES_PER_DENSITY * _SLOPE * np.log10(2.0)) / 1024.0)

    def test_the_code_follows_the_mean_of_the_channel_lines(self):
        """Cast Removal splits the slopes; the Frontier reads the mean of its channels."""
        k = contrast_mask_coeffs(1.0, 0.5, (2.0, 1.0, 3.0), (_PIVOT,) * 3, (0.0,) * 3, _RANGE)
        assert k is not None
        self.assertAlmostEqual(k[0], MASK_CODES_PER_DENSITY * 2.0)

    def test_strength_scales_both_sides(self):
        full, half = _coeffs(1.0), _coeffs(0.5)
        assert full is not None and half is not None
        self.assertGreater(full[4], 0.0)
        self.assertGreater(full[5], 0.0)
        np.testing.assert_allclose(half[4:6], np.asarray(full[4:6]) / 2.0, rtol=1e-9)
        self.assertEqual(half[:4], full[:4])

    def test_light_surrounds_burn_and_dark_surrounds_dodge(self):
        k = _coeffs()
        assert k is not None
        mid = 0.5 + _MID_OFFSET - 0.5
        plane = np.array([[mid - 0.3, mid, mid + 0.3]], dtype=np.float32)
        ev = contrast_mask_ev(plane, k, (1, 3))
        assert ev is not None
        self.assertGreater(ev[0, 0], 0.0)
        self.assertAlmostEqual(float(ev[0, 1]), 0.0, places=5)
        self.assertLess(ev[0, 2], 0.0)

    def test_ev_places_the_plane_back_at_the_crop(self):
        plane = np.full((4, 6), 0.25, dtype=np.float32)
        ev = contrast_mask_ev(plane, _coeffs(), (20, 30), roi=(5, 15, 6, 24))
        assert ev is not None
        self.assertEqual(ev.shape, (20, 30))
        # Edge-replicated outside the crop, so the crop tool's full-frame view has no seam.
        np.testing.assert_allclose(ev[0, 0], ev[5, 6], rtol=1e-5)

    def test_ev_is_off_without_coefficients_or_plane(self):
        self.assertIsNone(contrast_mask_ev(np.zeros((8, 8), dtype=np.float32), None, (8, 8)))
        self.assertIsNone(contrast_mask_ev(None, _coeffs(), (8, 8)))

    def test_coverage_weights_the_contrast_mask_stops(self):
        plane = np.array([[0.2, 0.4, 0.6, 0.8]], dtype=np.float32)
        coverage = np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        full = contrast_mask_ev(plane, _coeffs(), (1, 4))
        masked = contrast_mask_ev(plane, _coeffs(), (1, 4), coverage=coverage)
        assert full is not None and masked is not None
        np.testing.assert_allclose(masked[0, :2], full[0, :2])
        np.testing.assert_array_equal(masked[0, 2:], 0.0)


class TestContrastMaskMatte(unittest.TestCase):
    def setUp(self):
        self.image = _wide_range_negative(120, 180)

    def test_disabled_sky_restriction_returns_full_coverage_without_inference(self):
        from negpy.services.assets import sky_matte

        with patch.object(sky_matte, "get_sky_matte", side_effect=AssertionError("inference requested")):
            plane = contrast_mask_matte_plane(
                self.image,
                _BOUNDS,
                None,
                False,
                rotation=3,
                roi_norm=(0.25, 0.75, 0.0, 0.5),
            )
        self.assertEqual(plane.shape, (90, 60))
        np.testing.assert_array_equal(plane, 1.0)

    def test_enabled_sky_matte_uses_rough_positive_from_shared_normalized_input(self):
        from negpy.services.assets import sky_matte

        _, _, normalized_log = contrast_mask_plane_with_input(self.image, _BOUNDS, None)
        expected = sky_matte.rough_positive_proxy(normalized_log)
        observed = {}

        def fake_get_matte(image, _key, wait, _on_status):
            observed.update(image=image, wait=wait)
            return np.full(image.shape[:2], 0.25, dtype=np.float32)

        with patch.object(sky_matte, "get_sky_matte", side_effect=fake_get_matte):
            plane = contrast_mask_matte_plane(self.image, _BOUNDS, None, True, wait=True)

        np.testing.assert_array_equal(observed["image"], expected)
        self.assertTrue(observed["wait"])
        np.testing.assert_array_equal(plane, 0.25)


class TestContrastMaskRender(unittest.TestCase):
    def test_mask_opens_the_dark_area_and_holds_the_bright_one(self):
        """Columns of a bright, a mid and a dark area, each wider than the blur."""
        h, w = 240, 360
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        texture = 1.0 + 0.1 * np.sin(xx / 3.0) * np.sin(yy / 2.5)
        neg = np.where(xx < w / 3, 0.04, np.where(xx < 2 * w / 3, 0.2, 0.85)) * texture
        img = np.ascontiguousarray(np.dstack([neg, neg, neg]).astype(np.float32))
        plain = _render(img, _bw_settings(), "cm-cols-plain")
        masked = _render(img, _bw_settings(contrast_mask_strength=1.0), "cm-cols-masked")
        self.assertLess(masked[:, : w // 3].mean(), plain[:, : w // 3].mean())
        self.assertGreater(masked[:, 2 * w // 3 :].mean(), plain[:, 2 * w // 3 :].mean())

    def test_mask_compresses_the_global_range(self):
        img = _wide_range_negative()
        plain = _render(img, _bw_settings(grade=60.0), "cm-plain")
        masked = _render(img, _bw_settings(grade=60.0, contrast_mask_strength=1.0), "cm-masked")
        self.assertLess(_global_spread(masked), _global_spread(plain))

    def test_zero_strength_changes_nothing(self):
        img = _wide_range_negative()
        off = _render(img, _bw_settings(grade=60.0), "cm-off")
        explicit = _render(img, _bw_settings(grade=60.0, contrast_mask_strength=0.0), "cm-off")
        np.testing.assert_allclose(off, explicit, atol=1e-6)

    def test_disabled_sky_restriction_preserves_the_contrast_mask(self):
        img = _wide_range_negative()
        settings = _bw_settings(grade=60.0, contrast_mask_strength=1.0)
        with patch("negpy.services.assets.sky_matte.make_sky_matte", side_effect=AssertionError("model called")):
            unrestricted = _render(img, settings, "cm-no-sky")
        self.assertFalse(settings.exposure.contrast_mask_sky_only)
        self.assertTrue(np.isfinite(unrestricted).all())

    def test_a_slide_builds_no_plane(self):
        """The transfer curve reads no mask, so its plane is not worth building."""
        s = _bw_settings(contrast_mask_strength=1.0)
        engine = DarkroomEngine()
        engine.process(_wide_range_negative(), replace(s, process=replace(s.process, process_mode="Transparency")), "cm-slide")
        self.assertIsNone(engine._mask_plane)

    def test_mask_survives_a_resolution_change(self):
        """Sigma is a fraction of the analysis grid, not of the render."""
        img = _wide_range_negative()
        h, w = img.shape[:2]
        big = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)

        def drift(**kw):
            small = _render(img, _bw_settings(grade=60.0, **kw), "cm-res-s" + str(kw))
            large = _render(big, _bw_settings(grade=60.0, **kw), "cm-res-l" + str(kw))
            return float(np.abs(small - cv2.resize(large, (w, h), interpolation=cv2.INTER_AREA)).mean())

        self.assertLess(drift(contrast_mask_strength=1.0), 2.0 * drift() + 0.002)


class TestContrastMaskMigration(unittest.TestCase):
    def test_the_linear_mask_maps_onto_the_strength(self):
        config = WorkspaceConfig.from_flat_dict({"contrast_mask": 0.1, "mask_spacer": 6.0})
        self.assertAlmostEqual(config.exposure.contrast_mask_strength, 0.2)

    def test_the_contrast_boost_has_no_equivalent(self):
        config = WorkspaceConfig.from_flat_dict({"contrast_mask": -0.5, "mask_blur": 3.0})
        self.assertEqual(config.exposure.contrast_mask_strength, 0.0)

    def test_a_current_save_is_not_mapped_again(self):
        config = WorkspaceConfig.from_flat_dict(WorkspaceConfig.from_flat_dict({"contrast_mask_strength": 0.3}).to_dict())
        self.assertAlmostEqual(config.exposure.contrast_mask_strength, 0.3)

    def test_sky_only_setting_round_trips_and_old_split_settings_are_dropped(self):
        config = WorkspaceConfig.from_flat_dict(
            {"contrast_mask_sky_only": True, "contrast_mask_split_matte": True, "contrast_mask_matte_area": "right"}
        )
        self.assertTrue(config.exposure.contrast_mask_sky_only)
        restored = WorkspaceConfig.from_flat_dict(config.to_dict())
        self.assertTrue(restored.exposure.contrast_mask_sky_only)

    def test_sky_only_defaults_off(self):
        config = WorkspaceConfig.from_flat_dict({})
        self.assertFalse(config.exposure.contrast_mask_sky_only)


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestContrastMaskParity(unittest.TestCase):
    def setUp(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        self.processor = ImageProcessor()
        if self.processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")
        self.img = _wide_range_negative(96, 144)

    def _render(self, settings: WorkspaceConfig, tag: str, prefer_gpu: bool, size_ref: float = 0.0) -> np.ndarray:
        result, _ = self.processor.run_pipeline(
            self.img.copy(),
            settings,
            tag,
            render_size_ref=size_ref or float(max(self.img.shape[:2])),
            prefer_gpu=prefer_gpu,
            readback_metrics=False,
        )
        arr = np.asarray(result.readback()) if hasattr(result, "readback") else np.asarray(result)
        return arr[:, :, :3].astype(np.float64)

    def _assert_match(self, settings: WorkspaceConfig, tag: str):
        cpu = self._render(settings, tag, prefer_gpu=False)
        # render_size_ref sizes the paper and only the GPU honours it, so a cropped frame
        # needs the crop's own long edge to land the two on the same grid.
        gpu = self._render(settings, tag, prefer_gpu=True, size_ref=float(max(cpu.shape[:2])))
        self.assertEqual(cpu.shape, gpu.shape)
        self.assertLess(float(np.mean(np.abs(cpu - gpu))), 0.01)
        self.assertLess(float(np.max(np.abs(cpu - gpu))), 0.04)

    def test_cpu_gpu_match(self):
        self._assert_match(_bw_settings(grade=60.0, contrast_mask_strength=1.0), "contrast-mask-parity")

    def test_cpu_gpu_match_color(self):
        s = _bw_settings(grade=60.0, contrast_mask_strength=0.7)
        self._assert_match(replace(s, process=replace(s.process, process_mode="Color Negative")), "contrast-mask-parity-c41")

    def test_cpu_gpu_match_cropped(self):
        # The only case where the shader's mask rect is not the whole frame.
        s = _bw_settings(grade=60.0, contrast_mask_strength=1.0)
        self._assert_match(
            replace(s, geometry=replace(s.geometry, crop_rect=(0.2, 0.15, 0.65, 0.7))),
            "contrast-mask-parity-crop",
        )

    def test_cpu_gpu_match_sky_only_after_rotation_and_crop(self):
        from unittest.mock import patch
        from negpy.services.assets import sky_matte

        settings = _bw_settings(
            grade=60.0,
            contrast_mask_strength=1.0,
            contrast_mask_sky_only=True,
        )
        settings = replace(
            settings,
            geometry=replace(settings.geometry, rotation=3, crop_rect=(0.1, 0.15, 0.75, 0.65)),
        )
        with patch.object(
            sky_matte,
            "get_sky_matte",
            side_effect=lambda image, _key, _wait, _done: np.full(image.shape[:2], 0.5, dtype=np.float32),
        ):
            self._assert_match(settings, "contrast-mask-sky-only-rotated-crop")

    def test_slider_uploads_no_texture(self):
        base = _bw_settings(grade=60.0, contrast_mask_strength=1.0)
        self._render(base, "contrast-mask-drag", prefer_gpu=True)
        engine = self.processor.engine_gpu
        mask_key, ev_key = engine._mask_tex_key, engine._local_ev_key
        self.assertIsNotNone(mask_key)

        self._render(replace(base, exposure=replace(base.exposure, contrast_mask_strength=0.4)), "contrast-mask-drag", prefer_gpu=True)
        self.assertEqual(engine._mask_tex_key, mask_key)
        self.assertEqual(engine._local_ev_key, ev_key)

    def test_sky_only_toggle_uploads_the_matte_texture(self):
        from unittest.mock import patch
        from negpy.services.assets import sky_matte

        base = _bw_settings(grade=60.0, contrast_mask_strength=1.0)
        with patch.object(
            sky_matte,
            "get_sky_matte",
            side_effect=lambda image, _key, _wait, _done: np.full(image.shape[:2], 0.5, dtype=np.float32),
        ):
            self._render(
                replace(base, exposure=replace(base.exposure, contrast_mask_sky_only=True)), "contrast-mask-matte-cache", prefer_gpu=True
            )
        engine = self.processor.engine_gpu
        matte_key = engine._matte_tex_key
        self.assertIsNotNone(matte_key)
        with patch.object(
            sky_matte,
            "get_sky_matte",
            side_effect=lambda image, _key, _wait, _done: np.full(image.shape[:2], 0.5, dtype=np.float32),
        ):
            self._render(base, "contrast-mask-matte-cache", prefer_gpu=True)
        self.assertNotEqual(engine._matte_tex_key, matte_key)

    def test_gpu_matte_completion_reports_the_original_source_hash(self):
        from unittest.mock import Mock, patch
        from negpy.services.assets import sky_matte

        settings = _bw_settings(contrast_mask_strength=1.0, contrast_mask_sky_only=True)
        callback = Mock()
        self.processor.on_sky_matte_status = callback

        def finish_matte(image, _key, _wait, on_status):
            if on_status is not None:
                on_status("calculating", None)
                on_status("done", None)
            return np.ones(image.shape[:2], dtype=np.float32)

        with patch.object(sky_matte, "get_sky_matte", side_effect=finish_matte):
            self.processor.run_pipeline(
                self.img.copy(),
                settings,
                "source-frame",
                render_size_ref=float(max(self.img.shape[:2])),
                prefer_gpu=True,
                readback_metrics=False,
            )

        self.assertEqual(callback.call_count, 2)
        self.assertEqual(callback.call_args_list[0].args[0], "source-frame")
        self.assertEqual(callback.call_args_list[0].args[2], "calculating")
        self.assertEqual(callback.call_args_list[1].args[2], "done")
        self.assertIsNone(callback.call_args_list[1].args[3])


if __name__ == "__main__":
    unittest.main()
