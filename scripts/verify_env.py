"""验证所有运行时依赖是否就绪：ffmpeg, CUDA, 模型文件, llama-server。

用法: python scripts/verify_env.py [--translate]
  不带 --translate: 只检查阶段 A 依赖（ASR）
  带 --translate: 同时检查阶段 B 依赖（Sakura + llama-server）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONUTF8", "1")

from jpzh_subtitle.config import ASR_MODEL, LLM_MODEL_PATH, MODELS_DIR, find_ffmpeg  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "✓" if ok else "✗"
    msg = f"  {status} {name}"
    if detail:
        msg += f": {detail}"
    print(msg, flush=True)
    return ok


def main() -> int:
    check_translate = "--translate" in sys.argv
    print("=== 环境验证 ===\n", flush=True)
    all_ok = True

    # 1. Python
    print("[Python]", flush=True)
    all_ok &= check("Python 版本", sys.version_info >= (3, 10), f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # 2. ffmpeg
    print("\n[ffmpeg]", flush=True)
    try:
        ff = find_ffmpeg()
        all_ok &= check("ffmpeg 可用", True, ff)
    except FileNotFoundError as e:
        all_ok &= check("ffmpeg 可用", False, str(e))

    # 3. CUDA (for faster-whisper/ctranslate2)
    print("\n[CUDA / ctranslate2]", flush=True)
    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
        all_ok &= check("CUDA 设备数", n > 0, f"{n} 个 GPU")
        if n > 0:
            types = ctranslate2.get_supported_compute_types("cuda")
            all_ok &= check("支持的计算类型", "float16" in types, ", ".join(sorted(types)))
    except ImportError:
        all_ok &= check("ctranslate2", False, "未安装")
    except Exception as e:
        all_ok &= check("CUDA 检测", False, str(e))

    # 4. faster-whisper
    print("\n[faster-whisper]", flush=True)
    try:
        import faster_whisper
        all_ok &= check("faster-whisper", True, faster_whisper.__version__)
    except ImportError:
        all_ok &= check("faster-whisper", False, "未安装")

    # 5. Whisper 模型
    print("\n[Whisper ASR 模型]", flush=True)
    asr_path = Path(ASR_MODEL)
    all_ok &= check("ASR 模型路径", True, str(asr_path))
    if asr_path.is_dir():
        model_bin = asr_path / "model.bin"
        if model_bin.is_file():
            sz = model_bin.stat().st_size / 1024 / 1024
            all_ok &= check("model.bin", sz > 1000, f"{sz:.0f} MB")
        else:
            all_ok &= check("model.bin", False, "文件不存在")
    else:
        all_ok &= check("模型目录", False, "目录不存在（将自动从 HF 下载）")

    if not check_translate:
        print("\n" + ("=" * 40), flush=True)
        if all_ok:
            print("✓ 阶段 A 环境就绪！可用: python -m jpzh_subtitle video.mkv", flush=True)
        else:
            print("✗ 有依赖未就绪，请检查上方标记 ✗ 的项目", flush=True)
        return 0 if all_ok else 1

    # === 阶段 B 检查 ===

    # 6. Sakura GGUF 模型
    print("\n[Sakura LLM 模型]", flush=True)
    llm_path = Path(LLM_MODEL_PATH)
    all_ok &= check("LLM 模型路径", True, str(llm_path))
    if llm_path.is_file():
        sz = llm_path.stat().st_size / 1024 / 1024
        all_ok &= check("GGUF 文件", sz > 1000, f"{sz:.0f} MB")
    else:
        # 检查是否有 .part 文件正在下载
        part = llm_path.with_suffix(".gguf.part")
        if part.is_file():
            sz = part.stat().st_size / 1024 / 1024
            all_ok &= check("GGUF 文件", False, f"下载中: {sz:.0f} MB (.part)")
        else:
            all_ok &= check("GGUF 文件", False, "文件不存在")

    # 7. llama-server
    print("\n[llama-server]", flush=True)
    from jpzh_subtitle.server import LLAMACPP_DIR, _find_server_exe
    all_ok &= check("llamacpp 目录", LLAMACPP_DIR.is_dir(), str(LLAMACPP_DIR))
    try:
        exe = _find_server_exe()
        all_ok &= check("llama-server.exe", True, exe)
    except FileNotFoundError as e:
        all_ok &= check("llama-server.exe", False, str(e))

    # 检查 CUDA DLL
    cublas = LLAMACPP_DIR / "cublas64_12.dll"
    cudart = LLAMACPP_DIR / "cudart64_12.dll"
    all_ok &= check("cublas64_12.dll", cublas.is_file(), str(cublas) if cublas.is_file() else "缺失")
    all_ok &= check("cudart64_12.dll", cudart.is_file(), str(cudart) if cudart.is_file() else "缺失")

    # 8. requests
    print("\n[requests]", flush=True)
    try:
        import requests
        all_ok &= check("requests", True, requests.__version__)
    except ImportError:
        all_ok &= check("requests", False, "未安装")

    print("\n" + ("=" * 40), flush=True)
    if all_ok:
        print("✓ 阶段 B 环境就绪！可用: python -m jpzh_subtitle video.mkv --translate", flush=True)
    else:
        print("✗ 有依赖未就绪，请检查上方标记 ✗ 的项目", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
