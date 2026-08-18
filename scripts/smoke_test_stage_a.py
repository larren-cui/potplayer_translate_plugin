"""阶段 A 冒烟测试：对 samples/ja_example.m4a 跑完整管线，产出日文 .srt。

前置：models/whisper-large-v3/model.bin 已下载完成。
用法: python scripts/smoke_test_stage_a.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 用本地 src（src layout，未安装到环境时）
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONUTF8", "1")

from jpzh_subtitle.config import MODELS_DIR  # noqa: E402
from jpzh_subtitle.pipeline import run  # noqa: E402

SAMPLE = ROOT / "samples" / "ja_example.m4a"
ASR_MODEL = str(MODELS_DIR / "whisper-large-v3")
OUT = ROOT / "samples" / "ja_example.ja.srt"


def main() -> int:
    if not SAMPLE.is_file():
        print(f"测试样本不存在: {SAMPLE}（先运行 scripts/fetch_ja_sample.py）", flush=True)
        return 1
    model_bin = MODELS_DIR / "whisper-large-v3" / "model.bin"
    if not model_bin.is_file():
        print(f"Whisper 模型未就绪: {model_bin}", flush=True)
        return 1

    print(f"运行阶段 A: {SAMPLE.name} -> 日文 srt", flush=True)
    out = run(SAMPLE, out_srt=OUT, language="ja", asr_model=ASR_MODEL, translate=False)
    print(f"\n✓ 字幕已生成: {out}", flush=True)
    print("--- 内容 ---", flush=True)
    print(out.read_text(encoding="utf-8-sig"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
