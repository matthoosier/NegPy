## Plan: Sky Contrast Mask POC

Build a standalone CLI in `scripts/` for sky inference, full-resolution mask reconstruction, contrast adjustment, and diagnostics. Because the input is a positive TIFF rather than NegPy’s calibrated negative-density data, the contrast operation will be explicitly labeled an approximation.

**Steps**
1. **Pin the model contract.** Use NVIDIA’s SegFormer-B0 ADE20K checkpoint: its config maps sky to class 2 and specifies RGB normalization and 512 input size. Verify resize behavior and the ONNX input/output contract during implementation. The model license restricts use to non-commercial research/evaluation; document that restriction. Require a local ONNX path and do not download model files at runtime.
2. **Implement input handling and mask stages.** Require an explicit `linear`, `srgb`, or `log` representation. Validate TIFF layout and metadata, preserve dimensions, and report display-mapping parameters. Convert compatible tagged profiles to sRGB primaries without applying a second transfer conversion; reject ambiguous profile/representation combinations. Run ONNX inference, guided filtering (`radius=12`, `epsilon=1e-3` defaults), configurable trimap thresholds, then `pymatting.estimate_alpha_cf(image, trimap, ...)`. Missing dependencies or invalid inputs should fail clearly, with no algorithm substitution.
3. **Implement the contrast approximation.** Document NegPy’s reference equation, `D' = D - g * blur(D) + const`, and its dependence on normalized negative log-density and print-exposure context. Since those inputs are unavailable here, implement an isolated approximation on a pseudo-density derived from the declared input in linear light. Use a luminance-based scalar adjustment, blend with the matte in linear light, and report the assumptions and clipping statistics.
4. **Write outputs and diagnostics.**  Neever overwrite the input.
    - `output.tif` — final image after application of spatially restricted contrast mask, saved
       as 16-bit RGB TIFF.
    - `sky_probability.png` — raw SegFormer sky probability, upsampled to original resolution.
    - `sky_guided_probability.png` — probability map after guided-filter refinement.
    - `sky_trimap.png` — definite-sky, unknown, and definite-foreground regions.
    - `sky_mask.png` — final alpha matte, encoded as 8-bit grayscale for inspection.
    - `mask_overlay.png` — final mask or boundary overlaid on the original image.
    - `comparison.png` — original, global contrast-mask result, and localized result displayed side by side.
    - `report.json` — model identity, input representation, processing parameters, image metadata, clipping statistics, and warnings.
5. **Add setup, tests, and evaluation notes.** Provide isolated pinned dependencies and a README covering model setup, license, CLI usage, input representations, and the approximation. Add synthetic tests for identity/strength, alpha blending, output encoding, matte bounds, trimap constraints, and failure cases. Real-photo quality evaluation remains pending because no evaluation TIFF has been supplied.

**Relevant files**
- [scripts/sky_contrast_mask.py](scripts/sky_contrast_mask.py) — standalone CLI and processing functions.
- [scripts/sky_contrast_mask_requirements.txt](scripts/sky_contrast_mask_requirements.txt) — isolated dependencies, including OpenCV contrib and PyMatting.
- [scripts/README_sky_contrast_mask.md](scripts/README_sky_contrast_mask.md) — setup, model provenance, and usage.
- [scripts/sky_contrast_mask_evaluation.md](scripts/sky_contrast_mask_evaluation.md) — evaluation record, with visual findings pending a TIFF.
- [tests/test_sky_contrast_mask.py](tests/test_sky_contrast_mask.py) — focused synthetic tests.
- Reference only: [normalization.py](negpy/features/exposure/normalization.py), [logic.py](negpy/features/exposure/logic.py), [PIPELINE.md](docs/PIPELINE.md), and [semantic_model.py](negpy/services/assets/semantic_model.py).

**Verification**
1. Run the focused test module in the isolated tool environment and invoke the CLI with `--help`.
2. Run a synthetic end-to-end case with a generated 16-bit TIFF and a test ONNX fixture.
3. Run existing reference checks: `uv run pytest tests/test_contrast_mask.py tests/test_tiff_loader_encoding.py -q`.
4. With a real TIFF and ONNX model, inspect mask stages, leakage, halos, and clipping, then record the findings.

**Decisions**
- No NegPy runtime, GUI, dependency, or pipeline/user-guide changes.
- The selected NVIDIA model is for non-commercial evaluation only.
- `output.tif` follows the selected mode; the comparison always includes both results.
- The contrast operation is an approximation, not a claim of reproducing NegPy’s calibrated negative workflow.
- Photo-quality assessment is pending an evaluation TIFF.

