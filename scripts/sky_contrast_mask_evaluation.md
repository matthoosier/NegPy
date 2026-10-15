# Sky Contrast Mask Evaluation

Status: pending a 16-bit RGB TIFF and a local ONNX model export.

| Item | Result |
| --- | --- |
| Input file | Not supplied |
| Declared representation | Not run |
| Model artifact and license | NVIDIA SegFormer-B0 ADE20K; non-commercial evaluation only |
| Sky probability | Not assessed |
| Guided-filter boundary alignment | Not assessed |
| Trimap known/unknown coverage | Not assessed |
| Alpha around branches, wires, and rooflines | Not assessed |
| Foreground leakage | Not assessed |
| Global sky detail | Not assessed |
| Localized foreground protection | Not assessed |
| Visible halos or abrupt transitions | Not assessed |
| Clipping | Not assessed |

The required photo review compares the raw probability, guided probability, trimap, alpha matte, overlay, and original/global/localized comparison. Record observed errors here after a real-image run. Synthetic test results do not establish segmentation or edge quality.