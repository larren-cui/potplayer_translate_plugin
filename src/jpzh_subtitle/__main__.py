"""命令行入口。

用法:
    python -m jpzh_subtitle <视频文件>                # 产出日文字幕
    python -m jpzh_subtitle <视频文件> --translate   # 产出中文字幕（阶段B）
    python -m jpzh_subtitle <视频文件> -o out.srt
"""
from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .config import ASR_MODEL, ensure_dirs
from .pipeline import run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jpzh-subtitle",
        description="离线日中字幕翻译工具：视频 → 日语ASR → 中文字幕",
    )
    p.add_argument("video", help="输入视频文件路径")
    p.add_argument("-o", "--output", help="输出 .srt 路径（默认与视频同名同目录）")
    p.add_argument(
        "--translate", action="store_true",
        help="执行日→中翻译（阶段 B；不传则只产出日文字幕）",
    )
    p.add_argument("--asr-model", default=ASR_MODEL, help="Whisper 模型名或本地路径")
    p.add_argument("--language", default="ja", help="ASR 源语言（默认 ja）")
    p.add_argument("--device", default="cuda", help="ASR 设备（cuda/cpu）")
    p.add_argument("--batch-size", type=int, default=20, help="翻译每批行数")
    p.add_argument("--keep-audio", action="store_true", help="保留中间音频 wav")
    p.add_argument("-v", "--verbose", action="count", default=0, help="详细日志（-v/-vv）")
    p.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ensure_dirs()

    out = run(
        args.video,
        out_srt=args.output,
        language=args.language,
        asr_model=args.asr_model,
        translate=args.translate,
        batch_size=args.batch_size,
        keep_temp_audio=args.keep_audio,
    )
    print(f"\n✓ 字幕已生成: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
