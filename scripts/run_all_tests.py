"""一键运行所有测试。

用法: python scripts/run_all_tests.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
ENV = {
    "PYTHONPATH": str(ROOT / "src"),
    "PYTHONUTF8": "1",
    **os.environ,
}

TESTS = [
    ("SRT 时间戳", ROOT / "scripts" / "test_srt.py"),
    ("Pipeline 集成", ROOT / "scripts" / "test_pipeline.py"),
    ("Translator HTTP", ROOT / "scripts" / "test_translator_http.py"),
]


def main() -> int:
    print("=" * 50)
    print("  jpzh-subtitle 全量测试")
    print("=" * 50)
    total = len(TESTS)
    passed = 0
    failed: list[str] = []
    t0 = time.time()

    for i, (name, script) in enumerate(TESTS, 1):
        print(f"\n[{i}/{total}] {name}")
        print("-" * 40)
        result = subprocess.run(
            [PYTHON, str(script)],
            env=ENV,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        # Print stdout (test results)
        if result.stdout:
            print(result.stdout.rstrip())
        # Print stderr only if it contains relevant info (not CRLF warnings)
        stderr_lines = [
            ln for ln in result.stderr.splitlines()
            if ln.strip() and "warning:" not in ln.lower() and "CRLF" not in ln
        ]
        if stderr_lines:
            for ln in stderr_lines[-5:]:  # last 5 relevant lines
                print(f"  [stderr] {ln}")

        if result.returncode == 0:
            passed += 1
            print(f"  => PASS")
        else:
            failed.append(name)
            print(f"  => FAIL (exit {result.returncode})")

    elapsed = time.time() - t0
    print(f"\n{'=' * 50}")
    print(f"  结果: {passed}/{total} 通过, {len(failed)} 失败 ({elapsed:.1f}s)")
    if failed:
        print(f"  失败: {', '.join(failed)}")
    print("=" * 50)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
