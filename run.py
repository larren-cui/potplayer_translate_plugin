"""项目根目录入口脚本：无需设置 PYTHONPATH / 无需 pip install，直接运行。

默认行为：翻译 + 输出与视频同名同目录的 .srt（video.mp4 → video.srt）。
即 `python run.py "video.mp4"` 一条命令出中文字幕。

用法:
    python run.py "video.mp4"                  # 默认：翻译，输出 video.srt
    python run.py "video.mp4" -o out.srt        # 指定输出路径
    python run.py "video.mp4" --device cpu     # 无 GPU 时
    python run.py "video.mp4" -vv               # 详细日志
    python run.py "video.mp4" --no-translate    # 只转写不翻译（输出日文 video.srt）

说明:
    - 在项目根目录运行即可，脚本会自动把 src/ 加入模块搜索路径。
    - 首次运行翻译时自动下载缺失权重到 models/，需联网，之后离线。
    - 默认开启翻译；--no-translate 改为只转写（仅日语 ASR，不下载翻译模型）。
    - 输出文件名与视频完全一致，仅后缀改为 .srt（除非用 -o 另行指定）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 1) 把 src/ 加入模块搜索路径（无需 PYTHONPATH，无需 pip install）
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# 2) 强制 UTF-8 输出，避免 Windows 中文/日文输出乱码
os.environ.setdefault("PYTHONUTF8", "1")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 3) 导入包内配置（src/ 已在 sys.path，可直接 import）
from jpzh_subtitle.config import ASR_MODEL  # noqa: E402


def _parse(argv: list[str]) -> argparse.Namespace:
    """解析 run.py 参数：默认翻译、默认输出与视频同名的 .srt。"""
    import argparse

    p = argparse.ArgumentParser(
        prog="run.py",
        description="离线日中字幕翻译工具：视频 → 中文字幕（默认翻译）",
    )
    p.add_argument("video", help="输入视频文件路径")
    p.add_argument(
        "-o", "--output",
        help="输出 .srt 路径（默认与视频同名同目录，仅改后缀为 .srt）",
    )
    p.add_argument(
        "--no-translate", action="store_true",
        help="只转写不翻译（仅日语 ASR，输出日文 .srt）",
    )
    p.add_argument("--asr-model", default=ASR_MODEL, help="Whisper 模型名或本地路径")
    p.add_argument("--language", default="ja", help="ASR 源语言（默认 ja）")
    p.add_argument("--device", default="cuda", help="ASR 设备（cuda/cpu）")
    p.add_argument("--batch-size", type=int, default=20, help="翻译每批行数")
    p.add_argument("--port", type=int, default=8080, help="翻译服务端口")
    p.add_argument("--n-ctx", type=int, default=8192, help="LLM 上下文长度")
    p.add_argument("--keep-audio", action="store_true", help="保留中间音频 wav")
    p.add_argument("-v", "--verbose", action="count", default=0, help="详细日志（-v/-vv）")
    return p.parse_args(argv)


def main() -> int:
    import logging

    from jpzh_subtitle import __version__  # noqa: F401（顺带校验包可导入）
    from jpzh_subtitle.config import ensure_dirs
    from jpzh_subtitle.pipeline import run

    args = _parse(sys.argv[1:])

    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ensure_dirs()

    translate = not args.no_translate
    # 默认输出与视频同名同目录，仅改后缀为 .srt（用户未指定 -o 时）
    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    out_srt = args.output or str(video_path.with_suffix(".srt"))

    # 翻译模式下，预设 llama-server 参数（经环境变量传给 server 模块）
    if translate:
        os.environ["JPZH_LLM_PORT"] = str(args.port)
        os.environ["JPZH_LLM_N_CTX"] = str(args.n_ctx)

    out = run(
        args.video,
        out_srt=out_srt,
        language=args.language,
        asr_model=args.asr_model,
        device=args.device,
        translate=translate,
        batch_size=args.batch_size,
        keep_temp_audio=args.keep_audio,
    )
    print(f"\n✓ 字幕已生成: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
