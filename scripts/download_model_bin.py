"""直接流式下载 faster-whisper-large-v3 的 model.bin（带断点续传 + 进度）。

经 hf-mirror 镜像。若中途中断，重新运行会从已下载位置续传。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

MIRROR = "https://hf-mirror.com"
REPO = "Systran/faster-whisper-large-v3"
DEST = Path(__file__).resolve().parents[1] / "models" / "whisper-large-v3" / "model.bin"
URL = f"{MIRROR}/{REPO}/resolve/main/model.bin"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".bin.part")

    # 查询远端总大小
    head = requests.head(URL, timeout=30, allow_redirects=True,
                        headers={"User-Agent": "Mozilla/5.0"})
    total = int(head.headers.get("Content-Length", 0))
    print(f"远端 model.bin 大小: {total/1024/1024:.0f} MB", flush=True)

    # 断点续传
    have = tmp.stat().st_size if tmp.exists() else 0
    if have and have >= total and total:
        print("已下载完整，重命名。", flush=True)
        tmp.replace(DEST)
        return 0
    if DEST.exists() and total and DEST.stat().st_size >= total:
        print("model.bin 已存在且完整。", flush=True)
        return 0

    headers = {"User-Agent": "Mozilla/5.0"}
    mode = "wb"
    if have:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
        print(f"从 {have/1024/1024:.0f} MB 处续传…", flush=True)
    else:
        print("从头下载…", flush=True)

    with requests.get(URL, stream=True, timeout=60, headers=headers) as r:
        r.raise_for_status()
        # 服务器可能忽略 Range 头返回 200 而非 206，需从头开始
        if have and r.status_code == 200:
            print("服务器忽略 Range 请求(status 200)，从头下载…", flush=True)
            have = 0
            mode = "wb"
        downloaded = have
        last_report = downloaded
        with open(tmp, mode) as f:
            for chunk in r.iter_content(1 << 20):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded - last_report >= 50 * 1024 * 1024:  # 每 50MB 报告
                        pct = (downloaded / total * 100) if total else 0
                        print(f"  {downloaded/1024/1024:.0f} / {total/1024/1024:.0f} MB ({pct:.0f}%)", flush=True)
                        last_report = downloaded
        print(f"下载完成: {downloaded/1024/1024:.0f} MB", flush=True)

    tmp.replace(DEST)
    print(f"已保存: {DEST}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
