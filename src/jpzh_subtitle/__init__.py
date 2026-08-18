"""jpzh-subtitle: 离线日中字幕翻译工具

程序化用法:
    from jpzh_subtitle import generate_subtitle
    out = generate_subtitle("video.mkv", translate=True)
"""
from __future__ import annotations

from .pipeline import run as generate_subtitle
from .srt import Segment, write_srt, format_timestamp
from .translate import Translator, SakuraTranslator, StubTranslator

__version__ = "0.1.0"

__all__ = [
    "generate_subtitle",
    "Segment",
    "write_srt",
    "format_timestamp",
    "Translator",
    "SakuraTranslator",
    "StubTranslator",
]
