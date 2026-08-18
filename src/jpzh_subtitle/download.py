"""运行时自动检测并下载模型权重 / 推理引擎。

设计目标：换电脑从零运行 `python -m jpzh_subtitle video --translate` 时，
缺失的权重会在运行时自动下载到项目目录下的 models/，无需手动预下载。

三类资产：
  - Whisper large-v3 ASR 模型（faster-whisper 自带 CTranslate2 引擎，只需权重）
  - Sakura-14B GGUF 翻译模型权重
  - llama.cpp 预编译 CUDA 二进制（含 llama-server.exe + CUDA DLL）
    — 这是加载 GGUF 的推理引擎，与 Sakura 权重配套使用（缺一不可）

全部走 requests 流式下载 + 断点续传 + 完整性校验。
默认国内镜像（hf-mirror / gh-proxy），可用环境变量覆盖。
"""
from __future__ import annotations

import logging
import os
import shutil
import zipfile
from pathlib import Path

from .config import MODELS_DIR

logger = logging.getLogger(__name__)

# --- 镜像源（可环境变量覆盖）---
# 国内默认：HF 走 hf-mirror，GitHub release 走 gh-proxy
HF_MIRROR = os.environ.get("JPZH_HF_MIRROR", "https://hf-mirror.com")
GH_PROXY = os.environ.get("JPZH_GH_PROXY", "https://gh-proxy.com/https://github.com")

# --- Whisper large-v3 (faster-whisper 格式) ---
WHISPER_REPO = "Systran/faster-whisper-large-v3"
WHISPER_DIR = MODELS_DIR / "whisper-large-v3"
# faster-whisper 加载所需的全部文件（与仓库实际清单一致）
WHISPER_FILES = [
    "model.bin",
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
]

# --- Sakura-14B 翻译 GGUF ---
SAKURA_REPO = "SakuraLLM/Sakura-14B-Qwen2beta-v0.9.2-GGUF"
SAKURA_FILENAME = "sakura-14b-qwen2beta-v0.9.2-q4km.gguf"
SAKURA_DIR = MODELS_DIR / "sakura"
SAKURA_PATH = SAKURA_DIR / SAKURA_FILENAME

# --- llama.cpp 预编译 CUDA 二进制 ---
LLAMA_RELEASE_TAG = "b10472"
LLAMA_ASSETS = [
    f"ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE_TAG}/"
    f"llama-{LLAMA_RELEASE_TAG}-bin-win-cuda-12.4-x64.zip",
    f"ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE_TAG}/"
    f"cudart-llama-bin-win-cuda-12.4-x64.zip",
]
LLAMACPP_DIR = MODELS_DIR / "llamacpp"

_CHUNK = 1 << 20  # 1MB


def _session() -> "requests.Session":
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    return s


def _stream_download(
    sess,
    url: str,
    dest: Path,
    *,
    label: str = "",
    report_every_mb: int = 100,
) -> Path:
    """流式下载到 dest，支持断点续传 + 完整性校验。

    下载先写 .part 文件，完整后原子重命名为 dest。中途中断可重跑续传。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    label = label or dest.name

    # 已存在最终文件且完整则跳过
    if dest.is_file():
        logger.info("%s 已存在，跳过下载。", label)
        return dest

    have = part.stat().st_size if part.exists() else 0
    headers: dict[str, str] = {}
    mode = "wb"
    if have:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
        logger.info("%s 续传：从 %.0f MB 起…", label, have / 1024 / 1024)
    else:
        logger.info("%s 开始下载…", label)

    with sess.get(url, stream=True, timeout=120, headers=headers) as r:
        r.raise_for_status()
        # 服务器忽略 Range 头返回 200（完整内容）而非 206（部分内容）
        # 此时必须从头写，否则 append 会产生损坏文件
        if have and r.status_code == 200:
            logger.warning("%s 服务器忽略 Range 请求，从头下载…", label)
            have = 0
            mode = "wb"
        total = int(r.headers.get("Content-Length", 0)) + have
        done = have
        last = done
        with open(part, mode) as f:
            for chunk in r.iter_content(_CHUNK):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if done - last >= report_every_mb * 1024 * 1024:
                        pct = done / total * 100 if total else 0
                        logger.info("  %s: %.0f/%.0f MB (%.0f%%)",
                                   label, done / 1024 / 1024,
                                   total / 1024 / 1024, pct)
                        last = done
        logger.info("%s 下载完成 (%.0f MB)。", label, done / 1024 / 1024)

    # 完整性校验
    if total and done < total:
        raise RuntimeError(
            f"{label} 下载不完整 ({done}/{total} 字节)，保留 .part 文件以便续传。"
        )
    part.replace(dest)
    return dest


def _ensure_whisper() -> Path:
    """确保 Whisper large-v3 权重齐全，缺失则下载。返回权重目录。"""
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    sess = _session()
    for fname in WHISPER_FILES:
        dest = WHISPER_DIR / fname
        if dest.is_file():
            continue
        # model.bin 较大，走续传；配置文件小文件直接下
        url = f"{HF_MIRROR}/{WHISPER_REPO}/resolve/main/{fname}"
        _stream_download(sess, url, dest, label=f"whisper/{fname}", report_every_mb=50)
    logger.info("Whisper 模型就绪: %s", WHISPER_DIR)
    return WHISPER_DIR


def _ensure_sakura() -> Path:
    """确保 Sakura GGUF 存在，缺失则下载。返回 GGUF 路径。"""
    SAKURA_DIR.mkdir(parents=True, exist_ok=True)
    if SAKURA_PATH.is_file():
        logger.info("Sakura GGUF 已存在，跳过下载。")
        return SAKURA_PATH
    url = f"{HF_MIRROR}/{SAKURA_REPO}/resolve/main/{SAKURA_FILENAME}"
    _stream_download(_session(), url, SAKURA_PATH, label="sakura-14b GGUF")
    logger.info("Sakura 模型就绪: %s", SAKURA_PATH)
    return SAKURA_PATH


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    """解压 zip 到 out_dir（直接平铺，不保留压缩包内的顶层目录结构影响）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def _ensure_llamacpp() -> Path:
    """确保 llama.cpp 二进制（含 llama-server.exe + CUDA DLL）就绪。

    下载 zip -> 解压到 LLAMACPP_DIR -> 删除 zip。
    检测标志：LLAMACPP_DIR 下存在 llama-server.exe。
    """
    LLAMACPP_DIR.mkdir(parents=True, exist_ok=True)
    # 已就绪则跳过
    from .server import _find_server_exe  # 延迟导入避免循环
    try:
        _find_server_exe()
        logger.info("llama.cpp 二进制已存在，跳过下载。")
        return LLAMACPP_DIR
    except FileNotFoundError:
        pass

    sess = _session()
    tmp_archives: list[Path] = []
    for asset in LLAMA_ASSETS:
        url = f"{GH_PROXY}/{asset}"
        fname = asset.split("/")[-1]
        dest = LLAMACPP_DIR / fname
        # zip 下载完即解压，不需要保留；用临时位置
        _stream_download(sess, url, dest, label=fname, report_every_mb=80)
        tmp_archives.append(dest)

    for arc in tmp_archives:
        logger.info("解压 %s …", arc.name)
        _extract_zip(arc, LLAMACPP_DIR)
        arc.unlink()  # 解压后删除 zip

    # 校验关键产物
    try:
        _find_server_exe()
    except FileNotFoundError as e:
        raise RuntimeError(
            f"llama.cpp 解压后仍未找到 llama-server.exe: {e}"
        ) from e
    logger.info("llama.cpp 就绪: %s", LLAMACPP_DIR)
    return LLAMACPP_DIR


def ensure_whisper() -> Path:
    """供 ASR 模块调用：确保 Whisper 权重就绪。返回权重目录。"""
    return _ensure_whisper()


def ensure_sakura() -> Path:
    """供 server 模块调用：确保 Sakura GGUF 就绪。返回 GGUF 路径。"""
    return _ensure_sakura()


def ensure_llamacpp() -> Path:
    """供 server 模块调用：确保 llama.cpp 引擎就绪。返回二进制目录。"""
    return _ensure_llamacpp()


def ensure_all_for_translate() -> None:
    """翻译模式所需全部资产：whisper + sakura + llamacpp。"""
    _ensure_whisper()
    _ensure_sakura()
    _ensure_llamacpp()
