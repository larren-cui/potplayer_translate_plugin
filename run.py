"""项目根目录入口脚本：无需设置 PYTHONPATH / 无需 pip install，直接运行。

用法:
    python run.py "video.mkv"                  # 产出日文字幕 (video.ja.srt)
    python run.py "video.mkv" translate        # 产出中文字幕 (video.zh.srt)，自动下载权重+拉起翻译
    python run.py "video.mkv" translate -o out.srt
    python run.py "video.mkv" translate --device cpu     # 无 GPU 时
    python run.py "video.mkv" translate -vv               # 详细日志

说明:
    - 在项目根目录运行即可，脚本会自动把 src/ 加入模块搜索路径。
    - 首次运行 translate 时自动下载缺失权重到 models/，需联网，之后离线。
    - 不带 translate 则只做日语转写，不下载翻译模型。
    - "translate" 可写成 "--translate"，二者等价；其余参数同标准 CLI。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 1) 把 src/ 加入模块搜索路径（无需 PYTHONPATH，无需 pip install）
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# 2) 强制 UTF-8 输出，避免 Windows 中文/日文输出乱码
os.environ.setdefault("PYTHONUTF8", "1")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _normalize_argv(argv: list[str]) -> list[str]:
    """允许裸 'translate' 作为翻译开关（等价于 '--translate'）。"""
    out = []
    for a in argv:
        if a == "translate":
            out.append("--translate")
        else:
            out.append(a)
    return out


def main() -> int:
    from jpzh_subtitle.__main__ import main as cli_main
    return cli_main(_normalize_argv(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
