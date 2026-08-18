"""路径与运行时配置：ffmpeg 定位、模型目录、临时目录。"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# 工程根目录（src 的上两级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 模型权重目录（不进版本库）。可用环境变量 JPZH_MODELS_DIR 覆盖。
MODELS_DIR = Path(os.environ.get("JPZH_MODELS_DIR", PROJECT_ROOT / "models"))

# Whisper ASR 模型：名称（自动下载）或本地目录路径。
ASR_MODEL = os.environ.get("JPZH_ASR_MODEL", "large-v3")

# 翻译 LLM 的 GGUF 权重路径（阶段 B 填入）。
LLM_MODEL_PATH = os.environ.get("JPZH_LLM_MODEL", str(MODELS_DIR / "sakura-14b.gguf"))

# HuggingFace 镜像端点（国内网络下避免下载被阻断）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# llama.cpp 预编译二进制目录（含 cudart64_12.dll、cublas64_12.dll 等 CUDA 运行时库）。
# faster-whisper/ctranslate2 加载 GPU 时需要 cuBLAS，但 speech 环境的 torch 是 CPU 版
# 不附带 CUDA 库；此处把 llamacpp 目录加入 PATH 使其可被发现。
LLAMACPP_DIR = MODELS_DIR / "llamacpp"
if LLAMACPP_DIR.is_dir():
    os.environ["PATH"] = str(LLAMACPP_DIR) + os.pathsep + os.environ.get("PATH", "")

# 常见 ffmpeg 安装位置（按优先级探测）。
_FFMPEG_CANDIDATES = [
    os.environ.get("JPZH_FFMPEG"),
    shutil.which("ffmpeg"),
    r"D:\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
]


def find_ffmpeg() -> str:
    """返回 ffmpeg 可执行文件路径，找不到则抛出友好错误。"""
    for cand in _FFMPEG_CANDIDATES:
        if cand and Path(cand).is_file():
            return cand
    raise FileNotFoundError(
        "未找到 ffmpeg。请将其加入 PATH，或设置环境变量 JPZH_FFMPEG 指向 ffmpeg.exe。"
    )


def ensure_dirs() -> None:
    """创建所需的运行时目录。"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
