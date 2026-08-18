"""Stage B 可行性侦察：Sakura 模型文件 + 构建工具 + 可装包。

决定 Stage B 走哪条路：
  - llama.cpp 路线：需 llama-cpp-python 的 CUDA 构建（需 nvcc 或预编译 wheel）
  - transformers 路线：需 transformers + accelerate + bitsandbytes（4bit 量化）
"""
from __future__ import annotations

import json
import shutil
import subprocess

import requests

MIRROR = "https://hf-mirror.com"
s = requests.Session()
s.headers["User-Agent"] = "Mozilla/5.0"


def have(cmd: str) -> str:
    p = shutil.which(cmd)
    return p or "(not found)"


def run(cmd: list[str], timeout: int = 10) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()[:200]
    except Exception as e:
        return f"ERR {type(e).__name__}: {e}"


print("=== 构建工具 ===", flush=True)
print("  cmake:", have("cmake"))
print("  cl.exe (MSVC):", have("cl"))
print("  nvcc:", have("nvcc"))
print("  gcc:", have("gcc"))
print("  make/ninja:", have("make"), have("ninja"))

print("\n=== CUDA toolkit 目录探测 ===", flush=True)
import os, glob
for base in [r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
             r"C:\Program Files\NVIDIA Corporation"]:
    if os.path.isdir(base):
        print("  found:", base)
        for d in glob.glob(os.path.join(base, "v*"))[:5]:
            print("    ", d)

print("\n=== Sakura GGUF 文件 (Qwen2beta v0.9.2) ===", flush=True)
try:
    r = s.get(f"{MIRROR}/api/models/SakuraLLM/Sakura-14B-Qwen2beta-v0.9.2-GGUF/tree/main", timeout=25)
    r.raise_for_status()
    for it in r.json():
        sz = it.get("size")
        szstr = f"{sz/1024/1024:.0f} MB" if sz else ""
        print(f"  {it['path']:<55} {szstr}", flush=True)
except Exception as e:
    print(f"  ERR {type(e).__name__}: {e}", flush=True)

print("\n=== 可装包探测 (清华镜像) ===", flush=True)
for pkg in ["llama-cpp-python", "transformers", "accelerate", "bitsandbytes"]:
    try:
        r = s.get(f"https://pypi.tuna.tsinghua.edu.cn/simple/{pkg}/", timeout=20)
        r.raise_for_status()
        # 取最后几个版本号
        lines = [l for l in r.text.split("\n") if "whl" in l.lower() or "tar.gz" in l.lower()]
        # 找 win_amd64 / cp311 的 wheel
        win_cu = [l for l in lines if "win_amd64" in l and "cp311" in l]
        last_win = win_cu[-3:] if win_cu else lines[-2:]
        print(f"  {pkg}: 最新 win/cp311 wheel 片段:", flush=True)
        for l in last_win:
            # 提取文件名
            import re
            m = re.search(r'href="([^"]+)"', l)
            if m:
                print(f"      {m.group(1).split('/')[-1]}", flush=True)
    except Exception as e:
        print(f"  {pkg}: ERR {type(e).__name__}: {str(e)[:80]}", flush=True)
