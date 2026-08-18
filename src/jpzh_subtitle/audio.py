"""音频提取：用 ffmpeg 把视频抽成 16kHz 单声道 WAV（Whisper 所需格式）。"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .config import find_ffmpeg


def extract_audio(
    video_path: str | Path,
    out_wav: str | Path,
    *,
    sample_rate: int = 16000,
    overwrite: bool = True,
) -> Path:
    """从视频抽取音轨为 16kHz 单声道 PCM WAV。

    Args:
        video_path: 输入视频文件。
        out_wav: 输出 wav 路径。
        sample_rate: 采样率，Whisper 要求 16000。
        overwrite: 是否覆盖已有输出。

    Returns:
        输出 wav 的 Path。
    """
    video_path = Path(video_path)
    out_wav = Path(out_wav)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-i", str(video_path),
        "-vn",                 # 不要视频
        "-ac", "1",            # 单声道
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",   # 16bit PCM
        "-loglevel", "error",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)
    return out_wav
