"""SRT 字幕读写。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    """一条字幕：起止时间（秒）与文本。"""
    start: float
    end: float
    text: str


def format_timestamp(seconds: float) -> str:
    """秒数 -> SRT 时间戳 'HH:MM:SS,mmm'。"""
    if seconds < 0:
        seconds = 0.0
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:  # 四舍五入进位
        seconds += 1
        ms -= 1000
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[Segment], out_path: str | Path) -> Path:
    """将字幕段列表写为 .srt 文件（UTF-8 with BOM，兼容性更好）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        text = seg.text.strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}")
        lines.append(text)
        lines.append("")  # 段间空行
    # 写入 UTF-8 BOM，避免部分播放器把中文显示为乱码
    out_path.write_text("\n".join(lines), encoding="utf-8-sig")
    return out_path
