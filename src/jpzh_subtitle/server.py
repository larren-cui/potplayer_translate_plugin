"""管理 llama.cpp 的 llama-server 进程：启动/就绪探测/关闭。

llama-server 是 llama.cpp 自带的 OpenAI 兼容 HTTP 服务器，加载 GGUF 后
以本地 HTTP 服务暴露推理。本模块负责拉起它并等待就绪。

架构上这与 PotPlayer 扩展设想中"本地翻译服务"是同一个东西——可复用。
"""
from __future__ import annotations

import logging
import os
import socket
import subprocess
import time
from pathlib import Path

from .config import LLM_MODEL_PATH, MODELS_DIR

logger = logging.getLogger(__name__)

# llama-server 二进制所在目录（下载解压后）
LLAMACPP_DIR = MODELS_DIR / "llamacpp"
DEFAULT_PORT = 8080


def _find_server_exe() -> str:
    """在解压目录中查找 llama-server.exe。"""
    # 优先精确名
    exe = LLAMACPP_DIR / "llama-server.exe"
    if exe.is_file():
        return str(exe)
    # 退而在子目录搜索
    for p in LLAMACPP_DIR.rglob("llama-server.exe"):
        return str(p)
    raise FileNotFoundError(
        f"未找到 llama-server.exe，请先运行 scripts/download_llamacpp.py 并解压到 {LLAMACPP_DIR}"
    )


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_ready(port: int = DEFAULT_PORT, timeout: float = 180.0) -> bool:
    """等待 llama-server 的 HTTP 端口可连。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_in_use(port):
            return True
        time.sleep(1.0)
    return False


def start_server(
    *,
    model_path: str | None = None,
    port: int = DEFAULT_PORT,
    n_gpu_layers: int = -1,
    n_ctx: int = 8192,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """启动 llama-server（若端口已占用则视为已运行，直接返回占位进程）。

    Args:
        model_path: GGUF 路径；默认 config.LLM_MODEL_PATH。
        port: 监听端口。
        n_gpu_layers: GPU 层数，-1 = 全部卸载到 GPU。
        n_ctx: 上下文长度（token）。字幕批量翻译需足够大。
        extra_args: 透传给 llama-server 的额外参数。
    """
    if _port_in_use(port):
        logger.info("端口 %d 已被占用，假定 llama-server 已在运行。", port)
        # 返回一个占位 Popen（不可关闭真实进程）；用属性标记
        ph = subprocess.Popen(["cmd", "/c", "echo", "placeholder"])
        ph._placeholder = True  # type: ignore[attr-defined]
        return ph

    model_path = model_path or LLM_MODEL_PATH
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"GGUF 模型不存在: {model_path}")

    exe = _find_server_exe()
    # 将解压目录加入 PATH，使 CUDA 运行时 DLL（cudart64_12.dll 等）可被找到
    env = os.environ.copy()
    env["PATH"] = str(LLAMACPP_DIR) + os.pathsep + env.get("PATH", "")

    cmd = [
        exe,
        "-m", str(model_path),
        "--port", str(port),
        "--host", "127.0.0.1",
        "-ngl", str(n_gpu_layers),
        "-c", str(n_ctx),
        "-t", "4",          # CPU 线程数（GPU 推理时影响小）
        "--metrics",
        "-na",              # 非交互模式（避免等 stdin）
    ]
    if extra_args:
        cmd += extra_args

    logger.info("启动 llama-server: %s", " ".join(cmd[:1] + ["…"] + cmd[1:]))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    if not wait_ready(port, timeout=240.0):
        proc.terminate()
        raise RuntimeError(f"llama-server 启动超时（端口 {port} 未就绪）。检查模型/显存。")
    logger.info("llama-server 就绪 (port=%d, pid=%d)", port, proc.pid)
    return proc


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if getattr(proc, "_placeholder", False):
        return  # 占位进程，不关真实服务
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
