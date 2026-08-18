"""下载 llama.cpp 预编译 CUDA 二进制（经 gh-proxy.com 代理 github releases）。

包含：
  - llama-bXXXX-bin-win-cuda-12.4-x64.zip : 主程序（含 llama-server.exe）
  - cudart-llama-bin-win-cuda-12.4-x64.zip : CUDA 运行时 DLL（系统无 CUDA toolkit 时必需）
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

PROXY = "https://gh-proxy.com/https://github.com"
RELEASE_TAG = "b10472"
ASSETS = [
    f"ggml-org/llama.cpp/releases/download/{RELEASE_TAG}/llama-{RELEASE_TAG}-bin-win-cuda-12.4-x64.zip",
    f"ggml-org/llama.cpp/releases/download/{RELEASE_TAG}/cudart-llama-bin-win-cuda-12.4-x64.zip",
]
OUT = Path(__file__).resolve().parents[1] / "models" / "llamacpp"
OUT.mkdir(parents=True, exist_ok=True)

s = requests.Session()
s.headers["User-Agent"] = "Mozilla/5.0"


def download(url: str, dest: Path) -> Path:
    have = dest.stat().st_size if dest.exists() else 0
    headers = {}
    mode = "wb"
    if have:
        # 查总大小，若已完整则跳过
        head = s.head(url, timeout=30, allow_redirects=True)
        total = int(head.headers.get("Content-Length", 0))
        if total and have >= total:
            print(f"  已存在且完整: {dest.name}", flush=True)
            return dest
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
        print(f"  续传 {dest.name} 从 {have/1024/1024:.0f}MB…", flush=True)
    else:
        print(f"  下载 {dest.name} …", flush=True)

    with s.get(url, stream=True, timeout=120, headers=headers) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + have
        done = have
        last = done
        with open(dest, mode) as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if done - last >= 80 * 1024 * 1024:
                        pct = done / total * 100 if total else 0
                        print(f"    {dest.name}: {done/1024/1024:.0f}/{total/1024/1024:.0f}MB ({pct:.0f}%)", flush=True)
                        last = done
    print(f"  完成: {dest.name} ({done/1024/1024:.0f}MB)", flush=True)
    return dest


def main() -> int:
    for asset in ASSETS:
        url = f"{PROXY}/{asset}"
        dest = OUT / asset.split("/")[-1]
        download(url, dest)
    print("全部下载完成。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
