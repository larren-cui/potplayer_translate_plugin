"""日→中翻译后端。

设计为可切换接口（Translator 协议）：
  - StubTranslator：阶段 A 占位，原样返回日文（不调模型，便于先打通管线）。
  - SakuraTranslator：阶段 B，经 llama-server（OpenAI 兼容 HTTP）调 Sakura-14B GGUF。

切换后端只需换一个类，管线其余部分不变。
"""
from __future__ import annotations

import logging
import time
from typing import Protocol

logger = logging.getLogger(__name__)

# Sakura 翻译提示词（v0.9 系列，Qwen2 chat 格式经 llama-server 的 OpenAI 接口）
SAKURA_SYSTEM = (
    "你是一个轻小说翻译模型，可以流畅通顺地使用日本轻小说的文风将日文翻译成简体中文。"
)


class Translator(Protocol):
    """翻译器接口：把一批日文行翻译成中文行。"""

    def translate_lines(self, ja_lines: list[str]) -> list[str]:
        ...


class StubTranslator:
    """占位翻译器：原样返回（阶段 A 用，产出日文字幕）。"""

    def translate_lines(self, ja_lines: list[str]) -> list[str]:
        return list(ja_lines)


class SakuraTranslator:
    """基于 Sakura-14B（经 llama-server HTTP）的日→中翻译后端。

    Sakura 专为轻小说/动漫/galgame 文本微调，对 NSFW 内容不做安全拒绝，
    适合动漫、日剧及成人向视频的日→中翻译。

    通过本地 llama-server 暴露的 OpenAI 兼容 /v1/chat/completions 接口调用，
    与 ASR 进程解耦，GPU 由 llama-server 管理（无需 Python CUDA 构建）。
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8080",
        model_name: str = "sakura",
        max_tokens: int = 4096,
        temperature: float = 0.1,
        top_p: float = 0.3,
        timeout: float = 300.0,
        max_retries: int = 2,
    ):
        import requests  # 延迟导入，阶段 A 无需 requests

        self._base_url = base_url.rstrip("/")
        self._model = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._timeout = timeout
        self._max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def _chat(self, messages: list[dict]) -> str:
        url = f"{self._base_url}/v1/chat/completions"
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                r = self._session.post(url, json=payload, timeout=self._timeout)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("翻译请求失败(第%d次): %s", attempt + 1, e)
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"翻译请求多次失败: {last_err}")

    def translate_lines(self, ja_lines: list[str]) -> list[str]:
        if not ja_lines:
            return []
        # 批量翻译：把多行用换行拼接，要求模型保持换行结构
        joined = "\n".join(ja_lines)
        user = (
            "将下面的日文文本翻译成中文，保持原有的换行结构（每行对应一句），"
            "直接输出译文，不要添加任何解释或编号：\n" + joined
        )
        zh = self._chat([
            {"role": "system", "content": SAKURA_SYSTEM},
            {"role": "user", "content": user},
        ])
        zh_lines = [ln.strip() for ln in zh.splitlines() if ln.strip()]
        # 对齐：行数不一致时按行兜底（pipeline 也会再做一次兜底）
        if len(zh_lines) < len(ja_lines):
            zh_lines += ja_lines[len(zh_lines):]
        return zh_lines[: len(ja_lines)]
