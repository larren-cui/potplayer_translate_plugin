"""下载日语测试音频样本（经 hf-mirror）。"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

MIRROR = "https://hf-mirror.com"
OUT = Path(__file__).resolve().parents[1] / "samples"
OUT.mkdir(parents=True, exist_ok=True)

s = requests.Session()
s.headers["User-Agent"] = "Mozilla/5.0"


def resolve_url(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    return f"{MIRROR}/{repo_type}s/{repo_id}/resolve/main/{filename}"


def download(url: str, dest: Path) -> Path:
    print(f"下载 {url} -> {dest}", flush=True)
    with s.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
    print(f"  完成: {dest.stat().st_size/1024:.0f} KB", flush=True)
    return dest


def list_dir(repo_id: str, path: str = "") -> list[str]:
    url = f"{MIRROR}/api/datasets/{repo_id}/tree/main/{path}?recursive=false&limit=200"
    r = s.get(url, timeout=25)
    r.raise_for_status()
    return [it["path"] for it in r.json()]


def main() -> int:
    # 1) 确认可用的日语示例
    download(
        resolve_url("NadiaHolmlund/Japanese_Speech_Examples", "Example_1.m4a"),
        OUT / "ja_example.m4a",
    )

    # 2) 探查动漫语音数据集结构，下载首个音频
    audio_exts = (".wav", ".mp3", ".flac", ".m4a")
    for sub in ["", "data"]:
        try:
            files = list_dir("joujiboi/japanese-anime-speech", sub)
            print(f"[anime-speech/{sub or 'root'}] {len(files)} entries", flush=True)
            auds = [f for f in files if f.lower().endswith(audio_exts)]
            if auds:
                # 文件路径是完整 path
                fn = auds[0].split("/")[-1]
                download(
                    resolve_url("joujiboi/japanese-anime-speech", auds[0]),
                    OUT / f"ja_anime_{fn}",
                )
                return 0
            for f in files[:6]:
                print("    ", f, flush=True)
        except Exception as e:
            print(f"  anime-speech list ERR: {type(e).__name__}: {str(e)[:80]}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
