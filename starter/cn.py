"""在图上画中文。

OpenCV 的 `putText` 画不了 CJK 字符。机器上能找到 CJK 字体时走 PIL 渲染；
找不到就静默退回 OpenCV（只能画 ASCII），保证在没有字体的环境里程序仍然能跑。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]

_font_path = next((p for p in _FONT_CANDIDATES if Path(p).exists()), None)
try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:                                   # noqa: BLE001 - 没有 PIL 时静默降级
    _HAS_PIL = False


def available():
    return _HAS_PIL and _font_path is not None


def text(img, s, org, color=(255, 255, 255), size=22, anchor="lt", plate=False):
    """在 `org`（x, y）处画 `s`。`anchor` 取 'lt'（左上）或 'lm'（左中）。"""
    x, y = int(org[0]), int(org[1])
    if not available():
        cv2.putText(img, s, (x, y + size), cv2.FONT_HERSHEY_SIMPLEX, size / 32.0,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, s, (x, y + size), cv2.FONT_HERSHEY_SIMPLEX, size / 32.0,
                    tuple(int(c) for c in color), 1, cv2.LINE_AA)
        return img
    font = ImageFont.truetype(_font_path, size)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    bbox = dr.textbbox((0, 0), s, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ty = y - h // 2 if anchor == "lm" else y
    if plate:
        dr.rectangle([x - 6, ty - 5, x + w + 6, ty + h + 7], fill=(20, 20, 20))
    rgb = tuple(int(c) for c in color)[::-1]        # BGR 转 RGB
    dr.text((x, ty - bbox[1]), s, font=font, fill=rgb)
    out = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    img[:, :] = out
    return img


def banner(img, s, color=(0, 255, 255), size=24):
    cv2.rectangle(img, (0, 0), (img.shape[1], size + 24), (25, 25, 25), -1)
    return text(img, s, (14, 12), color, size)
