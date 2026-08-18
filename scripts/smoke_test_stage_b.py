"""阶段 B 冒烟测试：对 samples/ja_example.m4a 跑完整管线含翻译，产出中文 .srt。

前置：
  - models/whisper-large-v3/model.bin 已下载
  - models/sakura/sakura-14b-qwen2beta-v0.9.2-q4km.gguf 已下载
  - models/llamacpp/llama-server.exe 已解压
用法: python scripts/smoke_test_stage_b.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONUTF8", "1")

from jpzh_subtitle.config import MODELS_DIR  # noqa: E402
from jpzh_subtitle.pipeline import run  # noqa: E402

SAMPLE = ROOT / "samples" / "ja_example.m4a"
ASR_MODEL = str(MODELS_DIR / "whisper-large-v3")
SAKURA_GGUF = str(MODELS_DIR / "sakura" / "sakura-14b-qwen2beta-v0.9.2-q4km.gguf")
OUT = ROOT / "samples" / "ja_example.zh.srt"


def main() -> int:
    if not SAMPLE.is_file():
        print(f"测试样本不存在: {SAMPLE}", flush=True); return 1
    if not Path(SAKURA_GGUF).is_file():
        print(f"Sakura 模型未就绪: {SAKURA_GGUF}", flush=True); return 1

    # 告诉 server 模块用哪个 GGUF
    os.environ["JPZH_LLM_MODEL"] = SAKURA_GGUF

    print(f"运行阶段 B: {SAMPLE.name} -> 中文 srt", flush=True)
    out = run(SAMPLE, out_srt=OUT, language="ja", asr_model=ASR_MODEL, translate=True)
    print(f"\n✓ 中文字幕已生成: {out}", flush=True)
    print("--- 内容 ---", flush=True)
    print(out.read_text(encoding="utf-8-sig"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
