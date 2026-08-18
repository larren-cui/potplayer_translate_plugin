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
DEFAULT_PORT = int(os.environ.get("JPZH_LLM_PORT", "8080"))
DEFAULT_N_CTX = int(os.environ.get("JPZH_LLM_N_CTX", "8192"))


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
        f"未找到 llama-server.exe（目录 {LLAMACPP_DIR}）。运行翻译时本应自动下载，"
        f"若失败请检查网络或手动运行下载。"
    )


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _http_health_ok(port: int) -> bool:
    """向 llama-server 发 HTTP GET /health，确认 HTTP 层就绪（而非仅 TCP 端口）。"""
    import urllib.request
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health", method="GET")
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def wait_ready(port: int = DEFAULT_PORT, timeout: float = 300.0) -> bool:
    """等待 llama-server 的 HTTP /health 端点返回 200。

    比 TCP 端口检查更可靠：llama-server 可能在模型加载完成前就接受 TCP 连接，
    但 HTTP /health 在服务完全就绪后才返回 200。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _http_health_ok(port):
            return True
        time.sleep(1.5)
    return False


def start_server(
    *,
    model_path: str | None = None,
    port: int = DEFAULT_PORT,
    n_gpu_layers: int = -1,
    n_ctx: int = DEFAULT_N_CTX,
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

    # 运行时自动下载缺失资产：Sakura GGUF 权重 + llama.cpp 引擎二进制
    from .download import ensure_llamacpp, ensure_sakura
    logger.info("确保翻译资产就绪（Sakura GGUF + llama.cpp）…")
    ensure_llamacpp()
    model_path = str(ensure_sakura())

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
        "-a", "sakura",     # API 返回的模型别名（SakuraTranslator 用）
        "--jinja",          # 启用 Jinja chat template（Qwen2 模板从 GGUF 自动读取）
        "--metrics",
    ]
    if extra_args:
        cmd += extra_args

    logger.info("启动 llama-server: %s", " ".join(cmd[:1] + ["…"] + cmd[1:]))

    # 重定向输出到日志文件（而非 DEVNULL），便于诊断启动失败
    log_file = LLAMACPP_DIR.parent / "llama-server.log"
    log_fh = open(log_file, "w", encoding="utf-8", buffering=1)

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    proc._log_fh = log_fh  # type: ignore[attr-defined]

    if not wait_ready(port, timeout=300.0):
        proc.terminate()
        # 读取日志帮助诊断
        try:
            log_fh.flush()
            diag = log_file.read_text(encoding="utf-8", errors="replace")[-800:]
        except Exception:
            diag = "(无法读取日志)"
        raise RuntimeError(
            f"llama-server 启动超时（端口 {port} 未就绪）。\n--- llama-server 日志尾部 ---\n{diag}"
        )
    logger.info("llama-server 就绪 (port=%d, pid=%d, log=%s)", port, proc.pid, log_file)
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
    # 关闭日志文件句柄
    log_fh = getattr(proc, "_log_fh", None)
    if log_fh:
        log_fh.close()
