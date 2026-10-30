"""ASCII-path-safe wrapper around the official evaluator.

The official ``tools/evaluate.py`` uses ``cv2.imread``.  On Windows, OpenCV
cannot open a path containing non-ASCII characters (for example a submission
folder named ``班级_学号_姓名``), so every mask is reported as missing and both
segmentation scores become 0.  This wrapper only replaces the *file reading*
function with ``np.fromfile + cv2.imdecode``; the scoring formulas in
``evaluate.py`` are unchanged.

Usage (same arguments as the official evaluator)::

    python tools/evaluate_safe.py --data data --pred predictions --split val \
        --output evaluation_val.json

Keep the official ``tools/evaluate.py`` untouched; use this file only for a
local/defense demonstration on Windows + Chinese paths.
"""
from __future__ import annotations

import runpy
from pathlib import Path

import cv2
import numpy as np


def _imread_unicode(path, flags=cv2.IMREAD_COLOR):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, flags) if buf.size else None


# evaluate.py calls cv2.imread through the module-level cv2 object.
cv2.imread = _imread_unicode

if __name__ == "__main__":
    # Execute the official evaluator as __main__, so its argparse block runs
    # unchanged; only cv2.imread has been redirected above.
    runpy.run_path(str(Path(__file__).with_name("evaluate.py")), run_name="__main__")
