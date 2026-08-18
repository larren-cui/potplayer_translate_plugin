"""在 HF（经 hf-mirror 镜像）上查找可直接下载的日语音频样本。

用 requests 直连镜像，规避 huggingface_hub 的 endpoint 解析问题。
"""
from __future__ import annotations

import os
import sys

MIRROR = "https://hf-mirror.com"
import requests

s = requests.Session()
s.headers["User-Agent"] = "Mozilla/5.0"


def search_datasets(query: str, limit: int = 20) -> list[str]:
    url = f"{MIRROR}/api/datasets?search={requests.utils.quote(query)}&limit={limit}&full=false"
    r = s.get(url, timeout=20)
    r.raise_for_status()
    return [it["id"] for it in r.json()]


def list_files(repo_id: str, repo_type: str = "dataset") -> list[str]:
    url = f"{MIRROR}/api/{repo_type}s/{repo_id}/tree/main?recursive=false&limit=1000"
    r = s.get(url, timeout=25)
    r.raise_for_status()
    return [it["path"] for it in r.json()]


def main() -> int:
    audio_exts = (".wav", ".mp3", ".flac", ".m4a")
    queries = ["japanese", "japanese speech", "ja speech", "reazon"]
    found_ids: list[str] = []
    for q in queries:
        try:
            ids = search_datasets(q)
            print(f"=== search '{q}': {len(ids)} results ===", flush=True)
            for i in ids[:12]:
                print("  -", i, flush=True)
                found_ids.append(i)
        except Exception as e:
            print(f"search '{q}' ERR: {type(e).__name__}: {e}", flush=True)

    # 对若干候选数据集列出顶层文件，找含音频文件的
    seen = set()
    for rid in found_ids[:25]:
        if rid in seen:
            continue
        seen.add(rid)
        try:
            files = list_files(rid)
            audio = [f for f in files if f.lower().endswith(audio_exts)]
            if audio:
                print(f"\n>>> {rid}: {len(audio)} audio files (top-level)", flush=True)
                for f in audio[:10]:
                    print("    ", f, flush=True)
                return 0
            # 没有顶层音频，打印结构线索
            print(f"    [{rid}] {len(files)} files, sample: {files[:4]}", flush=True)
        except Exception as e:
            print(f"    [{rid}] list ERR: {type(e).__name__}: {str(e)[:70]}", flush=True)

    print("\n未找到含顶层音频文件的日语数据集。", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
