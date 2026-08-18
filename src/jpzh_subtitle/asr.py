"""日语 ASR：基于 faster-whisper（CTranslate2，GPU）做语音转写 + 逐句时间戳。"""
from __future__ import annotations

import logging
from pathlib import Path

from .srt import Segment

logger = logging.getLogger(__name__)


class ASR:
    """封装 faster-whisper 转写器。

    语音 -> 文本。默认走 GPU（float16），日语，带 VAD 过滤静音。
    """

    def __init__(
        self,
        model: str = "large-v3",
        *,
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        from faster_whisper import WhisperModel

        logger.info("加载 ASR 模型 %r (device=%s, compute_type=%s)…", model, device, compute_type)
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str = "ja",
        vad_filter: bool = True,
        beam_size: int = 5,
    ) -> list[Segment]:
        """转写音频，返回带时间戳的字幕段。

        Args:
            audio_path: 16kHz WAV 路径。
            language: 源语言代码，ja=日语。
            vad_filter: 是否用 Silero VAD 过滤静音段（提升质量、加速）。
            beam_size: 束搜索宽度。
        """
        audio_path = str(audio_path)
        logger.info("开始转写 %s (语言=%s)…", audio_path, language)
        segments_gen, info = self._model.transcribe(
            audio_path,
            language=language,
            vad_filter=vad_filter,
            beam_size=beam_size,
            word_timestamps=True,
        )
        logger.info(
            "音频时长 %.1fs，检测语言=%s(概率 %.2f)",
            getattr(info, "duration", 0.0),
            getattr(info, "language", language),
            getattr(info, "language_probability", 0.0),
        )

        results: list[Segment] = []
        for seg in segments_gen:
            text = (seg.text or "").strip()
            if not text:
                continue
            results.append(Segment(start=seg.start, end=seg.end, text=text))
            logger.debug("[%.1fs -> %.1fs] %s", seg.start, seg.end, text)

        logger.info("转写完成，共 %d 段。", len(results))
        return results
