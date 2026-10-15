# Sky Contrast Mask POC

This standalone command-line tool tests semantic sky inference, full-resolution matte reconstruction, and a sky-localized contrast adjustment on 16-bit RGB TIFF positives. It does not modify or import the NegPy processing pipeline.

The contrast adjustment is an approximation. NegPy's Contrast Mask operates on normalized negative log-density and adds an exposure-stop plane inside the print pipeline. A positive TIFF does not provide the negative-density bounds or print context needed to reproduce that operation.

## Setup

Create an isolated environment with Python 3.13 and install the pinned dependencies:

```sh
uv venv .venv-sky-poc
uv pip install --python .venv-sky-poc/bin/python -r scripts/sky_contrast_mask_requirements.txt
```

On Windows, use `.venv-sky-poc/Scripts/python.exe` as the `--python` path.

## Model

The default model contract is NVIDIA's `nvidia/segformer-b0-finetuned-ade-512-512` ADE20K checkpoint. Sky is class 2 of 150. The image input is RGB, resized to a 512-pixel shortest side for dynamic spatial input, then normalized with means `(0.485, 0.456, 0.406)` and standard deviations `(0.229, 0.224, 0.225)`. Logits are treated as unnormalized class scores and softmax is applied before reading the sky probability.

The model card points to the NVIDIA SegFormer license. It limits the model and derivatives to non-commercial research or evaluation. Review the license before use.

The script fetches `onnx/model.onnx` from the pinned repository revision on first use and stores it in the Hugging Face cache. A network connection is needed only when the model is not already cached. No manual export or model path is required.

The model revision is `b9175de73a0a34f7843135853d27629aa6987b2f`. You can pass `--model PATH` to use a local ONNX file instead. The script validates the model input and logits shape when inference starts.

## Input Representation

The TIFF must be single-page, contiguous, unsigned 16-bit RGB with three samples per pixel. Planar RGB, alpha/extra samples, grayscale, and other bit depths are rejected. The source is never overwritten.

Pass `--representation` for every input:

- `linear`: samples are linear sRGB-primary values. No embedded ICC profile is accepted.
- `srgb`: samples use the sRGB transfer function and sRGB primaries. An embedded profile is accepted only when its description identifies sRGB; other profiles fail rather than being silently interpreted. Untagged inputs are assumed to use sRGB primaries.
- `a98c`: samples use the Adobe RGB (1998) transfer curve and primaries. An embedded profile identified as `A98C` or Adobe RGB (1998) is required. The script converts the samples to linear sRGB for processing and converts the output back to Adobe RGB (1998), retaining the embedded profile.
- `log`: normalized samples map linearly between explicit base-2 log values. For example, `--log-min-log2 -8 --log-max-log2 2` maps sample zero to $2^{-8}$ and full scale to $2^2$ linear intensity. Log TIFFs must not contain an ICC profile.

Segmentation uses a display-referred sRGB image. Linear input is sRGB-encoded first; log input is mapped to linear intensity using its declared bounds and then sRGB-encoded. The selected representation and mapping are saved in `report.json`.

For TIFF files with an A98C Adobe RGB (1998) profile, pass `--representation a98c`.

## Contrast Approximation

The NegPy reference is `D' = D - g * blur(D) + const`, where `D` is normalized, unmixed negative log-density. NegPy builds a zero-mean blurred luminance plane and converts it to print-exposure stops with `-gamma * density_range / log10(2)`.

This POC has positive-image input. It defines pseudo-density as `-log10(max(linear sRGB luminance, 1e-6))`, scales it by its 1st-to-99th percentile range, blurs it with a Gaussian sigma of 2-6% of the image's short side, and applies the resulting stops equally to RGB. This is an experimental approximation, not a reproduction of NegPy's negative workflow. Strength defaults to `0.25`, accepts `-0.5` to `0.5`, and zero is an identity operation. The localized blend and output encoding occur in the selected representation's corresponding linear-light pipeline. Out-of-range values are reported and clipped only during final 16-bit encoding.

## Run

```sh
.venv-sky-poc/bin/python scripts/sky_contrast_mask.py input.tif output-dir \
  --representation srgb
```

Use `--model PATH` only to override the cached model with a local ONNX file.

For log data, include both `--log-min-log2` and `--log-max-log2`. Guided-filter radius/epsilon, trimap thresholds, blur scale, and contrast strength have CLI options; run `--help` for the full list.

`output.tif` is the sky-localized result. `comparison.png` always shows the original, global result, and localized result. The other output files retain each probability, trimap, and alpha stage for inspection. `report.json` records model/input metadata, parameters, warnings, and clipping statistics.

## Validation

Run synthetic numerical and file-output tests without adding dependencies to NegPy:

```sh
uv run --no-project --with-requirements scripts/sky_contrast_mask_requirements.txt \
  python -m pytest tests/test_sky_contrast_mask.py -q
```

Synthetic tests do not assess semantic accuracy or halo quality. Test real photographs before relying on the matte.