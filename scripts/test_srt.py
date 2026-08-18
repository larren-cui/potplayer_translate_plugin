"""SRT 模块单元测试：时间戳格式化边界情况。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jpzh_subtitle.srt import format_timestamp, Segment, write_srt  # noqa: E402


def test_timestamps():
    tests = [
        (0.0, "00:00:00,000"),
        (1.5, "00:00:01,500"),
        (59.999, "00:00:59,999"),    # 59s 999ms — no rounding needed
        (61.234, "00:01:01,234"),
        (3661.789, "01:01:01,789"),
        (-1.0, "00:00:00,000"),      # negative clamped to 0
        (3599.999, "00:59:59,999"),  # 59m 59.999s — correct, not 1h
    ]
    all_ok = True
    for sec, expected in tests:
        got = format_timestamp(sec)
        ok = got == expected
        if not ok:
            all_ok = False
        print(f"  {'OK' if ok else 'FAIL'} format_timestamp({sec}) = {got} (expected {expected})")
    return all_ok


if __name__ == "__main__":
    ok = test_timestamps()
    print(f"\n{'PASS' if ok else 'FAIL'}: SRT timestamp tests")
    sys.exit(0 if ok else 1)
