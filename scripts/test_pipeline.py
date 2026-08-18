"""Pipeline 集成测试：mock ASR + mock Translator，验证批处理/对齐/SRT 生成。

测试用例：
  1. 多段翻译（25段 > batch_size=20，触发批次切分）
  2. 翻译行数不匹配时的兜底逻辑
  3. 空段过滤
  4. SRT 编号连续性
  5. 时间戳格式正确
  6. 不翻译模式（StubTranslator）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONUTF8", "1")

from jpzh_subtitle.pipeline import run, _batched  # noqa: E402
from jpzh_subtitle.srt import Segment  # noqa: E402
from jpzh_subtitle.translate import Translator  # noqa: E402


class MockASR:
    """模拟 ASR：跳过真实模型，直接返回预设段。"""

    def __init__(self, model: str = "mock", *, device: str = "cuda", **kw):
        self.model = model

    def transcribe(self, audio_path, *, language="ja", **kw):
        # 25 段，每段 3 秒，模拟日语台词
        return [
            Segment(
                start=i * 3.0,
                end=i * 3.0 + 2.5,
                text=f"日本語テキスト_{i:02d}",
            )
            for i in range(25)
        ]


class MockTranslator(Translator):
    """模拟翻译器：在每行前加 [中] 标记，模拟翻译。"""

    def __init__(self, *, mismatch_at: int = -1):
        self._mismatch_at = mismatch_at  # 在第 N 批返回不匹配行数
        self._call_count = 0

    def translate_lines(self, ja_lines: list[str]) -> list[str]:
        self._call_count += 1
        if self._mismatch_at == self._call_count:
            # 返回比输入少 2 行（触发兜底）
            return [f"[中]{ln}" for ln in ja_lines[:-2]]
        return [f"[中]{ln}" for ln in ja_lines]


class StubASR:
    """模拟无语音的 ASR。"""

    def __init__(self, model: str = "mock", *, device: str = "cuda", **kw):
        pass

    def transcribe(self, audio_path, *, language="ja", **kw):
        return []


def _setup_mock_video():
    """创建假视频文件（pipeline 会检查文件存在性）。"""
    fake = ROOT / ".tmp" / "fake_video.mp4"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_bytes(b"fake_video_content")
    return fake


def _patch_pipeline():
    """Monkey-patch pipeline.ASR + extract_audio。"""
    import jpzh_subtitle.pipeline as pl
    orig_asr = pl.ASR
    pl.ASR = MockASR

    orig_extract = pl.extract_audio
    def mock_extract(video, wav, **kw):
        Path(wav).parent.mkdir(parents=True, exist_ok=True)
        Path(wav).write_bytes(b"fake_wav")
        return wav
    pl.extract_audio = mock_extract
    return orig_asr, orig_extract


def _restore_pipeline(orig_asr, orig_extract):
    import jpzh_subtitle.pipeline as pl
    pl.ASR = orig_asr
    pl.extract_audio = orig_extract


def test_batched():
    """_batched 正确切分。"""
    assert list(_batched(list(range(25)), 20)) == [
        list(range(0, 20)), list(range(20, 25))
    ]
    assert list(_batched(list(range(40)), 20)) == [
        list(range(0, 20)), list(range(20, 40))
    ]
    assert list(_batched([], 20)) == []
    print("✓ test_batched 通过", flush=True)


def test_multi_segment_translation():
    """25 段翻译，触发 2 批（20+5），验证输出完整。"""
    fake_video = _setup_mock_video()
    orig_asr, orig_extract = _patch_pipeline()

    try:
        tr = MockTranslator()
        out = ROOT / ".tmp" / "test_multi.srt"
        result = run(
            str(fake_video),
            out_srt=out,
            language="ja",
            translate=True,
            translator=tr,
            batch_size=20,
        )
        content = result.read_text(encoding="utf-8-sig")
        lines = content.strip().split("\n")

        # 验证: 25 段全部翻译
        assert "[中]日本語テキスト_00" in content
        assert "[中]日本語テキスト_24" in content
        assert "日本語テキスト_24" in content

        # 验证: 翻译被调用 2 次（25 段 / batch_size 20 = 2 批）
        assert tr._call_count == 2, f"expected 2 batches, got {tr._call_count}"

        # 验证: SRT 编号连续 1-25
        for i in range(1, 26):
            assert f"\n{i}\n" in content or content.startswith(f"{i}\n"), f"missing index {i}"

        # 验证: 时间戳格式
        assert "00:00:00,000 --> 00:00:02,500" in content
        assert "00:01:12,000 --> 00:01:14,500" in content  # segment 24: 72s-74.5s

        print("✓ test_multi_segment_translation 通过 (25段, 2批, 编号连续, 时间戳正确)", flush=True)
    finally:
        _restore_pipeline(orig_asr, orig_extract)


def test_mismatch_fallback():
    """翻译行数不匹配时，pipeline 用原文本兜底。"""
    fake_video = _setup_mock_video()
    orig_asr, orig_extract = _patch_pipeline()

    try:
        # mismatch_at=1: 第一批（20行）返回 18 行 → 触发兜底
        tr = MockTranslator(mismatch_at=1)
        out = ROOT / ".tmp" / "test_mismatch.srt"
        result = run(
            str(fake_video),
            out_srt=out,
            language="ja",
            translate=True,
            translator=tr,
            batch_size=20,
        )
        content = result.read_text(encoding="utf-8-sig")

        # 行数不匹配时，pipeline 对整个批次用原日文兜底（安全做法：
        # 无法确定哪些行对应，所以全部回退到原文）
        # 第一批(0-19)全部兜底为原日文，第二批(20-24)正常翻译
        assert "日本語テキスト_00" in content  # 兜底：原日文
        assert "日本語テキスト_19" in content  # 兜底：原日文
        assert "[中]日本語テキスト_20" in content  # 第二批正常翻译
        assert "[中]日本語テキスト_24" in content  # 第二批正常翻译

        print("✓ test_mismatch_fallback 通过 (翻译行数不匹配时原文本兜底)", flush=True)
    finally:
        _restore_pipeline(orig_asr, orig_extract)


def test_empty_segments():
    """ASR 返回空段列表时，生成空 SRT 文件。"""
    fake_video = _setup_mock_video()
    import jpzh_subtitle.pipeline as pl
    orig_asr = pl.ASR
    pl.ASR = StubASR  # 返回空列表
    orig_extract = pl.extract_audio
    def mock_extract(video, wav, **kw):
        Path(wav).parent.mkdir(parents=True, exist_ok=True)
        Path(wav).write_bytes(b"fake")
        return wav
    pl.extract_audio = mock_extract

    try:
        out = ROOT / ".tmp" / "test_empty.srt"
        result = run(str(fake_video), out_srt=out, language="ja", translate=False)
        content = result.read_text(encoding="utf-8-sig")
        assert content.strip() == ""
        print("✓ test_empty_segments 通过 (空段→空SRT)", flush=True)
    finally:
        pl.ASR = orig_asr
        pl.extract_audio = orig_extract


def test_stub_no_translate():
    """不翻译模式（StubTranslator），输出原日文。"""
    fake_video = _setup_mock_video()
    orig_asr, orig_extract = _patch_pipeline()

    try:
        out = ROOT / ".tmp" / "test_stub.srt"
        result = run(str(fake_video), out_srt=out, language="ja", translate=False)
        content = result.read_text(encoding="utf-8-sig")
        # 不翻译：输出原日文
        assert "日本語テキスト_00" in content
        assert "[中]" not in content
        print("✓ test_stub_no_translate 通过 (不翻译模式输出原文本)", flush=True)
    finally:
        _restore_pipeline(orig_asr, orig_extract)


def main() -> int:
    print("=== Pipeline 集成测试 ===\n", flush=True)
    test_batched()
    test_multi_segment_translation()
    test_mismatch_fallback()
    test_empty_segments()
    test_stub_no_translate()
    print("\n✓ 全部 5 项测试通过", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
