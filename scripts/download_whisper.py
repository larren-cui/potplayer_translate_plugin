"""下载 Whisper large-v3 模型到本地（走 hf-mirror 国内镜像）。

用法:
    set HF_ENDPOINT=https://hf-mirror.com
    python scripts/download_whisper.py
"""
from __future__ import annotations

import os
import sys

# 国内镜像，避免直连 huggingface.co 被阻断
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download

REPO = "Systran/faster-whisper-large-v3"
LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "whisper-large-v3",
)


def main() -> int:
    print(f"下载 {REPO} -> {LOCAL_DIR} (镜像: {os.environ['HF_ENDPOINT']})", flush=True)
    os.makedirs(LOCAL_DIR, exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        local_dir=LOCAL_DIR,
        # 只下模型权重与配置，跳过原始权重以省空间
        allow_patterns=["*.bin", "*.json", "*.txt", "tokenizer/*", "vad/*"],
    )
    print("下载完成。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
