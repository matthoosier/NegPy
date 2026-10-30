"""OpenCV is allowed only in the temporary sky-matte preview service."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1] / "negpy"


def test_negpy_imports_no_cv2():
    pattern = re.compile(r"^\s*(import cv2|from cv2\b)", re.MULTILINE)
    allowed = {"services/assets/sky_matte.py"}
    offenders = [
        str(relative)
        for p in ROOT.rglob("*.py")
        if pattern.search(p.read_text(encoding="utf-8")) and (relative := p.relative_to(ROOT).as_posix()) not in allowed
    ]
    assert not offenders, f"cv2 imported in {offenders}; port the call to negpy_core"
