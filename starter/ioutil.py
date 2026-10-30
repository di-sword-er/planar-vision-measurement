"""让脚本在任何控制台编码下都不会因为打印中文而崩溃。

Windows 默认控制台是 GBK(cp936)。只要 print 里出现一个 GBK 编不了的字符
（比如数学减号 U+2212、或 emoji），就会抛 UnicodeEncodeError 直接结束进程。

做法：把 stdout/stderr 重设为 UTF-8 + errors="replace"。
最坏情况是显示成乱码，绝不会中断程序。
"""
from __future__ import annotations

import sys


def make_stdout_safe():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:          # noqa: BLE001 - 老版本或已被重定向
            pass
