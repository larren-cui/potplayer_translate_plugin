"""端到端编排：视频 -> 音频 -> 日语ASR -> (翻译) -> 字幕。"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from .audio import extract_audio
from .asr import ASR
from .config import PROJECT_ROOT
from .srt import Segment, write_srt
from .translate import SakuraTranslator, StubTranslator, Translator

logger = logging.getLogger(__name__)


class _LocalTempDir:
    """工作区内的临时目录（系统 temp 在沙箱中可能不可写）。"""

    def __init__(self, prefix: str = "jpzh_"):
        self._base = PROJECT_ROOT / ".tmp"
        self._base.mkdir(parents=True, exist_ok=True)
        self.name = self._base / f"{prefix}{uuid.uuid4().hex[:8]}"

    def __enter__(self):
        self.name.mkdir(parents=True, exist_ok=True)
        return self.name

    def __exit__(self, *exc):
        import shutil as _sh
        _sh.rmtree(self.name, ignore_errors=True)


def _batched(items: list, size: int):
    """把列表切成固定大小的批次。"""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run(
    video_path: str | Path,
    out_srt: str | Path | None = None,
    *,
    language: str = "ja",
    asr_model: str = "large-v3",
    translator: Translator | None = None,
    translate: bool = False,
    batch_size: int = 20,
    keep_temp_audio: bool = False,
) -> Path:
    """处理一个视频，生成字幕文件。

    Args:
        video_path: 输入视频。
        out_srt: 输出 .srt 路径；None 则与视频同名、同目录，后缀 .zh.srt。
        language: ASR 源语言（ja=日语）。
        asr_model: Whisper 模型名或本地路径。
        translator: 翻译后端；不传则用占位（原样日文）。
        translate: 是否执行翻译（False 则只产出源语言字幕）。
        batch_size: 翻译时每批合并的行数（减少 LLM 调用次数）。
        keep_temp_audio: 是否保留中间 wav。

    Returns:
        生成的 .srt 路径。
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    if out_srt is None:
        suffix = ".zh.srt" if translate else f".{language}.srt"
        out_srt = video_path.with_suffix(suffix)
    out_srt = Path(out_srt)

    asr = ASR(model=asr_model)
    tr = translator or StubTranslator()

    # 若需要翻译但未显式提供翻译器，自动拉起 llama-server + Sakura
    started_server = None
    if translate and translator is None:
        from .server import start_server
        logger.info("拉起本地 llama-server（Sakura 翻译）…")
        started_server = start_server()
        tr = SakuraTranslator()

    try:
        with _LocalTempDir() as tmp:
            wav = tmp / "audio.wav"
            logger.info("提取音频 -> %s", wav)
            extract_audio(video_path, wav)

            segments = asr.transcribe(wav, language=language)
            if not segments:
                logger.warning("未识别到任何语音，输出空字幕。")
                write_srt([], out_srt)
                return out_srt

            if translate:
                logger.info("开始翻译，每批 %d 行…", batch_size)
                translated_texts: list[str] = []
                for bi, batch in enumerate(_batched(segments, batch_size)):
                    ja_lines = [s.text for s in batch]
                    zh_lines = tr.translate_lines(ja_lines)
                    # 对齐长度：翻译返回行数应与输入一致，否则按原文本兜底
                    if len(zh_lines) != len(ja_lines):
                        logger.warning(
                            "第 %d 批翻译行数不匹配(%d!=%d)，按原文本兜底。",
                            bi, len(zh_lines), len(ja_lines),
                        )
                        zh_lines = ja_lines
                    translated_texts.extend(zh_lines)
                    logger.info("  已翻译 %d/%d 段", len(translated_texts), len(segments))
                segments = [
                    Segment(start=s.start, end=s.end, text=t)
                    for s, t in zip(segments, translated_texts)
                ]

            write_srt(segments, out_srt)
            logger.info("字幕已写入: %s", out_srt)

            if keep_temp_audio:
                keep = Path.cwd() / (video_path.stem + ".wav")
                import shutil as _sh
                _sh.copy2(wav, keep)
                logger.info("中间音频保留: %s", keep)
    finally:
        if started_server is not None:
            from .server import stop_server
            stop_server(started_server)

    return out_srt
