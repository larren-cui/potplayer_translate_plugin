"""下载 Sakura-14B-Qwen2beta-v0.9.2 GGUF（q4km 量化，约 8.7GB）经 hf-mirror。

Sakura 是专为轻小说/动漫/galgame 日→中微调的模型，对 NSFW 内容不做安全拒绝，
适合动漫、日剧及成人向视频的翻译。

用法:
    python scripts/download_sakura.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

MIRROR = "https://hf-mirror.com"
REPO = "SakuraLLM/Sakura-14B-Qwen2beta-v0.9.2-GGUF"
# q4km: 质量/体积平衡最佳；如需更小可改 iq4xs(7.5GB)，更高质量改 q6k(11.7GB)
FILENAME = "sakura-14b-qwen2beta-v0.9.2-q4km.gguf"
OUT = Path(__file__).resolve().parents[1] / "models" / "sakura" / FILENAME
URL = f"{MIRROR}/{REPO}/resolve/main/{FILENAME}"

s = requests.Session()
s.headers["User-Agent"] = "Mozilla/5.0"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".gguf.part")

    head = s.head(URL, timeout=30, allow_redirects=True)
    total = int(head.headers.get("Content-Length", 0))
    print(f"Sakura GGUF (q4km) 总大小: {total/1024/1024:.0f} MB", flush=True)

    have = tmp.stat().st_size if tmp.exists() else 0
    if OUT.exists() and total and OUT.stat().st_size >= total:
        print("已存在且完整。", flush=True)
        return 0

    headers = {}
    mode = "wb"
    if have:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
        print(f"从 {have/1024/1024:.0f}MB 续传…", flush=True)
    else:
        print("从头下载…", flush=True)

    with s.get(URL, stream=True, timeout=120, headers=headers) as r:
        r.raise_for_status()
        # 服务器可能忽略 Range 头返回 200（完整内容）而非 206（部分内容）
        # 此时必须从头开始写，否则 append 会产生损坏的文件
        if have and r.status_code == 200:
            print(f"服务器忽略 Range 请求(status 200)，从头下载…", flush=True)
            have = 0
            mode = "wb"
        done = have
        last = done
        with open(tmp, mode) as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if done - last >= 100 * 1024 * 1024:
                        pct = done / total * 100 if total else 0
                        print(f"  {done/1024/1024:.0f}/{total/1024/1024:.0f}MB ({pct:.0f}%)", flush=True)
                        last = done
    print(f"下载完成: {done/1024/1024:.0f}MB", flush=True)
    # 完整性校验：确认下载大小与远端一致
    if total and done < total:
        print(f"警告: 下载不完整 ({done}/{total} 字节)，保留 .part 文件以便续传。", flush=True)
        return 1
    tmp.replace(OUT)
    print(f"已保存: {OUT} ({done/1024/1024:.0f}MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
