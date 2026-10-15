# Proof of concept: ML-guided, full-resolution sky contrast masking

## Objective

Implement a standalone Python command-line tool that takes a 16-bit-per-channel RGB TIFF positive, identifies sky using a pretrained semantic-segmentation model, reconstructs a full-resolution sky alpha matte using OpenCV and alpha matting, and applies a contrast-mask operation selectively to the sky.

The experiment must distinguish three separate questions:

1. Can semantic segmentation identify the sky reliably enough to guide the operation?
2. Can image-guided refinement and alpha matting produce a spatially accurate, fractional-coverage mask at full resolution?
3. Does applying contrast masking only through that mask recover sky detail while preserving foreground tones?

This is a proof of concept, not a NegPy integration. Do not modify NegPy or build a GUI.

**Input flexibility:** the TIFF does not have to be a linear-light positive. It may be a gamma-encoded positive (such as sRGB) or a non-gamma-corrected positive represented in log space. The tool must require the input representation to be specified explicitly and must not assume that 16-bit samples imply linear light. Segmentation and mask reconstruction should work with either representation after appropriate preprocessing. The contrast-mask operation must document which representation it expects; an approximate operation on the supplied representation is acceptable for the first POC if faithfully reproducing NegPy's mathematics is not practical.

## 1. Input, output, and dependencies

### Input

Accept one RGB TIFF and an output directory.

- Expect 16-bit samples per RGB channel.
- Accept either:
  - linear-light RGB;
  - gamma-encoded RGB, such as sRGB; or
  - a non-gamma-corrected positive represented in log space.
- Require the user to specify the input representation with a CLI option. Do not silently infer linearity from bit depth or assume that every TIFF with an ICC profile is correctly described by that profile.
- If the input is log-space data, document what the stored values mean and how they are mapped to the display-referred image used for segmentation. Do not apply an sRGB transfer function directly to log-space values.
- If a color profile is present, account for it where applicable. Distinguish color-space conversion from transfer-function conversion.
- Preserve the original dimensions and never overwrite the input.
- Reject unsupported channel layouts or ambiguous input formats with a clear error message.

Use a suitable image library, such as `tifffile`, for reliable 16-bit TIFF reading and writing.

### Dependencies

Use:

- ONNX Runtime for model inference.
- A pretrained SegFormer-B0 semantic-segmentation model exported to ONNX.
- OpenCV contrib (`cv2.ximgproc`) for guided filtering.
- PyMatting for closed-form alpha matting.
- NumPy for numerical operations.
- `tifffile` for TIFF I/O.
- Pillow or OpenCV for diagnostic images.

CPU execution must be supported. GPU acceleration is optional.

Pin or document compatible dependency versions. Do not silently substitute different algorithms if a required dependency is unavailable.

### Outputs

Write the following files into the output directory:

- `output.tif` — processed 16-bit RGB TIFF.
- `sky_probability.png` — raw SegFormer sky probability, upsampled to original resolution.
- `sky_guided_probability.png` — probability map after guided-filter refinement.
- `sky_trimap.png` — definite-sky, unknown, and definite-foreground regions.
- `sky_mask.png` — final alpha matte, encoded as 8-bit grayscale for inspection.
- `mask_overlay.png` — final mask or boundary overlaid on the original image.
- `comparison.png` — original, global contrast-mask result, and localized result displayed side by side.
- `report.json` — model identity, input representation, processing parameters, image metadata, clipping statistics, and warnings.

Retain the final alpha matte as floating-point data internally. The 8-bit mask is a diagnostic representation, not the processing representation.

## 2. Stage A: semantic sky inference

Use a pretrained SegFormer-B0 model, preferably one trained on ADE20K, with a documented sky class.

Before implementation, identify the exact ONNX model and verify:

- Its provenance and license.
- Its expected input dimensions and tensor layout.
- Its color channel order.
- Its normalization and other preprocessing requirements.
- Its output tensor layout and sky class index.

Do not assume that the sky class index or preprocessing is correct without checking the selected model's configuration.

### Inference preprocessing

Convert the input to the display-referred color representation expected by the model.

- For linear-light RGB, apply the appropriate transfer function to produce display-referred sRGB (and perform any required color-space conversion).
- For gamma-encoded RGB, respect the specified transfer function and convert to the model's expected representation without applying the same encoding twice.
- For log-space positive data, use a documented display mapping from the stored log values to a suitable display-referred RGB image. The mapping may use an explicit scale/offset or user-specified display parameters; do not assume that log values are linear RGB or apply the sRGB encoding function directly to them.
- Apply the model's documented resizing and normalization.
- Run inference with ONNX Runtime.
- Extract the sky-class probability from the model's semantic output, applying softmax if required by the model output.

Save the raw sky probability map after upsampling it to the original image dimensions using `cv2.resize` with `cv2.INTER_LINEAR`.

This map is a semantic prior, not the final mask. It is expected to have uncertain or coarse boundaries because the model operates at reduced resolution.

## 3. Stage B: full-resolution, image-guided mask reconstruction

This stage is essential. Do not simply resize the SegFormer mask and apply Gaussian blur.

The purpose is to combine semantic knowledge from SegFormer with the full-resolution image's local structure, then estimate fractional pixel coverage at uncertain boundaries.

### B1. Guided-filter refinement

Use `cv2.ximgproc.guidedFilter`.

Inputs:

- **Guide:** the full-resolution source image converted into a suitable normalized floating-point RGB representation.
- **Source:** the upsampled SegFormer sky-probability map, represented as `float32` in the range 0–1.

For gamma-encoded input, use a consistently normalized display-referred guide. For linear or log-space input, create a documented, stable guide representation appropriate for edge guidance. The guide is used to locate image structure; it does not determine the contrast-mask math.

Start with these configurable parameters:

- Radius: 8–16 full-resolution pixels.
- Epsilon: approximately `1e-3` for a guide normalized to 0–1.

These are initial experimental values, not universal optima. Expose the radius and epsilon as command-line parameters.

The guided filter should refine the probability map using local image structure, encouraging mask transitions to align with edges in the full-resolution image. Keep the output in floating-point form and constrain it to the range 0–1.

Save this result as `sky_guided_probability.png`.

Do not treat every image edge as a sky boundary: buildings, branches, and clouds contain edges too. The SegFormer probability map must remain the semantic prior.

### B2. Construct a three-state trimap

Create a trimap from the refined probability map `P_r`.

- Definite sky: `P_r >= 0.95`.
- Definite foreground: `P_r <= 0.05`.
- Unknown: all remaining pixels.

Make both thresholds configurable.

Encode the trimap for diagnostics using three distinct grayscale values, such as 255 for definite sky, 0 for definite foreground, and 128 for unknown.

Report the number and percentage of unknown pixels. If almost the entire image is unknown, or either known class is implausibly small, emit a warning rather than silently proceeding as if the trimap were reliable.

Do not blindly erode or dilate the entire sky region to move the boundary. The unknown band is intended to let the matting algorithm estimate the transition.

### B3. Closed-form alpha matting

Use PyMatting's `pymatting.estimate_alpha_cf` to estimate a full-resolution alpha matte from:

- The original-resolution RGB image, converted into the color representation expected by the selected implementation.
- The three-state trimap.

Verify the selected PyMatting API and input conventions against its installed version. Ensure that definite-sky and definite-foreground constraints are respected in the final matte.

The desired output is a floating-point alpha image `A(x,y)` in the range 0–1:

- 0 means no sky contribution.
- 1 means full sky coverage.
- Intermediate values represent estimated fractional coverage.

This is intended to improve mask transitions around tree branches, wires, irregular rooflines, and other fine boundaries. Alpha matting uses local color relationships to estimate mixed foreground/background pixels rather than merely blurring a binary boundary.

Save the result as `sky_mask.png` for visual inspection, but retain floating-point precision internally for image processing.

If the trimap or source image is unsuitable for the selected matting algorithm, fail with a clear diagnostic or warning. Do not silently replace alpha matting with Gaussian blur, `cv2.grabCut`, or another hard segmentation method.

### B4. Diagnostic inspection and limitations

Alpha matting cannot guarantee correct recovery of features below the source resolution, and it may struggle when sky and foreground colors are similar.

Create `mask_overlay.png` so the user can inspect whether the reconstructed mask follows the actual skyline. The overlay should make it possible to identify foreground leakage, missing sky regions, and boundary halos.

Keep the semantic prediction, guided-filter result, trimap, and final alpha matte separately available. This makes failures attributable to the appropriate stage.

## 4. Stage C: contrast-mask operation

The mask determines where an adjustment applies. It does not itself recover cloud detail.

Implement the contrast-mask operation as a separate, testable function.

### Preferred implementation

Reproduce the contrast-mask mathematics used by NegPy as closely as practical, without importing or modifying NegPy for this initial experiment.

Before coding, identify the actual mathematical operation used by NegPy's Contrast Mask feature and document the formula being reproduced.

If implementing that operation outside NegPy proves disproportionately difficult, use a clearly documented experimental approximation. Keep it isolated behind a function interface so it can later be replaced with the actual NegPy operation.

Do not substitute a generic local-tone-mapping operator and claim that it demonstrates the effectiveness of NegPy's contrast-mask operation.

### Working-space requirements

The contrast-mask calculation must explicitly state which representation it operates on:

- If reproducing NegPy's operation, match its actual input domain and mathematical conventions as closely as possible.
- If the operation requires linear intensity or density/log values, convert to that representation using a documented transformation when the input supports it.
- If the log-space input's precise calibration or mapping is unknown, do not claim that an arbitrary conversion is physically exact. Label the operation as an approximation and record its assumptions.
- Do not perform blurring, luminance calculations, or contrast-mask math directly on gamma-encoded RGB values unless that is intentionally the chosen approximation.

Keep the spatial alpha matte separate from the contrast-mask calculation.

### Global and localized modes

Compute the contrast-adjusted result `C(I)` from the selected working representation.

Provide two comparison modes:

1. **Global:** apply the contrast-mask operation to the whole image.
2. **Localized:** blend the contrast-adjusted image with the original using the reconstructed sky alpha matte.

The localized blend is:

`output = (1 - alpha) * original + alpha * contrast_adjusted`

Here, `alpha` is the final full-resolution matte. The expression defines the spatial blend, not the contrast-mask mathematics itself. Perform the blend in a documented, appropriate working space. If the contrast operation is calculated in a different representation from the original image, convert both results to a common space before blending.

Do not apply arbitrary per-channel RGB adjustments and call them contrast masking.

### Numerical handling

- Keep intermediate calculations in floating-point form.
- Do not silently clip intermediate values.
- Detect and report values outside the representable output range.
- Convert back to 16-bit RGB at the final output stage using a documented range mapping or clipping policy.
- Preserve the original image dimensions and channel layout.

Avoid obvious halos at the skyline. Inspect both the mask and the rendered result rather than relying solely on numerical metrics.

## 5. Command-line interface

Provide a minimal CLI with:

- Input TIFF path.
- Output directory.
- Required input representation option, such as `linear`, `srgb`, another explicitly supported transfer function, or `log`.
- For log-space input, any required display-mapping parameters (or a documented default that can be overridden).
- Model path or documented model configuration.
- Contrast-mask strength, including zero.
- Global or localized comparison mode.
- Guided-filter radius and epsilon.
- Sky-probability thresholds for the trimap.
- Optional contrast-mask blur scale if required by the selected formula.

Use sensible defaults so the experiment can run with one command once the input representation and model path are supplied. Do not add a GUI, interactive polygon editor, training workflow, or extensive collection of tuning parameters.

If the model must be downloaded separately, document how to obtain it, where to place it, and how to specify its path. Do not silently download arbitrary model files at runtime.

## 6. Required diagnostics and automated tests

Produce a side-by-side comparison of:

1. Original image.
2. Global contrast-mask result.
3. Sky-localized contrast-mask result.

Also save all mask-stage diagnostics.

Implement these automated checks:

- Zero contrast-mask strength reproduces the input within expected numerical and encoding tolerance.
- A zero alpha mask leaves the image unchanged.
- A full-one alpha mask matches the global contrast-mask result.
- Output dimensions and 16-bit RGB encoding are preserved.
- Pixels outside the mask remain unchanged, except for explicitly documented processing effects.
- The final alpha matte contains finite values in the range 0–1.
- Definite-sky and definite-foreground trimap constraints are respected.
- Missing required dependencies or invalid model configuration produce explicit errors.
- The declared input representation is recorded in the report and is used consistently by preprocessing.

Use synthetic tests for the blending and numerical behavior. Use real photographs to assess semantic segmentation and mask quality.

## 7. Evaluation procedure

First test one challenging image with a nearly white sky and recoverable cloud detail. A gamma-encoded 16-bit positive or a non-gamma-corrected log-space positive is acceptable; the input representation must be declared and the corresponding preprocessing documented. Then test two or three images with different sky/foreground boundaries if the initial result is promising.

Inspect the raw SegFormer probability, guided-filter result, trimap, and final alpha matte independently.

Evaluate:

- Whether guided filtering improves boundary alignment over simple upsampling.
- Whether alpha matting produces plausible fractional coverage around branches, wires, and rooflines.
- Whether the sky is correctly identified without foreground leakage.
- Whether the localized operation reveals sky detail while protecting foreground tones.
- Whether halos or abrupt tonal transitions appear at the boundary.
- Whether the global and localized outputs demonstrate the benefit of spatial selectivity.

A successful proof of concept should show that the localized operation retains the desired sky improvement of the global operation while substantially reducing unwanted changes to the foreground.

If the global contrast-mask operation fails to recover the desired detail, investigate the contrast-mask formula, source clipping, and working-space assumptions before tuning the segmentation model.

## 8. Implementation deliverables

Provide:

- Source code organized into small functions for input/output, representation conversion, inference, guided filtering, trimap generation, alpha matting, contrast masking, and diagnostics.
- A dependency list with compatible versions.
- Model provenance and setup instructions.
- A README with installation instructions and a reproducible example command.
- Automated tests for the numerical pipeline.
- A short report of the test images, declared input representation, observed mask failures, clipping statistics, and visual results.

Before implementation, briefly report the chosen ONNX model, its sky class mapping, the exact guided-filter API and initial parameters, the PyMatting API, and the contrast-mask formula to be used. Resolve any uncertainty about the model or library interfaces before coding.
