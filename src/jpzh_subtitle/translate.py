"""日→中翻译后端。

设计为可切换接口（Translator 协议）：
  - StubTranslator：阶段 A 占位，原样返回日文（不调模型，便于先打通管线）。
  - SakuraTranslator：阶段 B，基于 llama-cpp-python 跑 Sakura-14B GGUF。

切换后端只需换一个类，管线其余部分不变。
"""
from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class Translator(Protocol):
    """翻译器接口：把一批日文行翻译成中文行。"""

    def translate_lines(self, ja_lines: list[str]) -> list[str]:
        ...


class StubTranslator:
    """占位翻译器：原样返回（阶段 A 用，产出日文字幕）。"""

    def translate_lines(self, ja_lines: list[str]) -> list[str]:
        return list(ja_lines)


class SakuraTranslator:
    """基于 Sakura-14B（llama.cpp）的日→中翻译后端（阶段 B 实现）。

    Sakura 专为轻小说/动漫/galgame 文本微调，对 NSFW 内容不做安全拒绝，
    适合动漫、日剧及成人向视频的日→中翻译。

    TODO(阶段B): 接入 llama-cpp-python，加载 GGUF，按 Sakura 提示词格式批量翻译。
    """

    def __init__(self, model_path: str, *, n_gpu_layers: int = -1, n_ctx: int = 4096):
        raise NotImplementedError("SakuraTranslator 在阶段 B 实现。")

    def translate_lines(self, ja_lines: list[str]) -> list[str]:
        raise NotImplementedError
