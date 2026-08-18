# jpzh-subtitle

离线日中字幕翻译工具：给定一个视频文件，自动生成日译中字幕（`.srt`）。全程离线，使用本地 GPU 完成语音识别与翻译。

模型权重会在首次运行时**自动下载**到项目目录下的 `models/`，无需手动下载。

## 环境要求

- **Python ≥ 3.10**（开发使用 3.11）
- **ffmpeg**：在 PATH 中，或设置环境变量 `JPZH_FFMPEG` 指向 `ffmpeg.exe`
- **NVIDIA GPU** + 驱动（CUDA 12.x；RTX 4090D 24GB 测试通过）

## 安装

```bash
# 1. 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 2. 安装依赖（国内镜像加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 运行

**在项目根目录下用 `run.py` 调用**（无需设置环境变量、无需安装为包）。

默认即翻译，并输出与视频同名同目录的 `.srt`（`video.mp4` → `video.srt`）。首次运行会自动下载 Whisper（~3GB）、Sakura GGUF（~8.7GB）、llama.cpp CUDA 二进制（~612MB）到 `models/`，需联网，之后全程离线。

```bash
# 默认：翻译，输出 video.srt
python run.py "video.mp4"

# 指定输出路径
python run.py "video.mp4" -o out.srt

# 只产出日文字幕（不翻译，不下载翻译模型）
python run.py "video.mp4" --no-translate

# 使用 CPU（无 GPU 时）
python run.py "video.mp4" --device cpu

# 详细日志（-v / -vv）
python run.py "video.mp4" -vv
```

其余参数同标准 CLI（`python run.py -h` 查看全部）。

## 配置（环境变量）

| 变量 | 说明 | 默认 |
|---|---|---|
| `JPZH_FFMPEG` | ffmpeg.exe 路径 | PATH 中查找 |
| `JPZH_MODELS_DIR` | 权重保存目录 | 项目下 `models/` |
| `JPZH_ASR_MODEL` | Whisper 模型名/路径 | `models/whisper-large-v3/` |
| `JPZH_LLM_MODEL` | 翻译 GGUF 权重路径 | `models/sakura/sakura-14b-...-q4km.gguf` |
| `JPZH_LLM_PORT` | 本地翻译服务端口 | `8080` |
| `JPZH_LLM_N_CTX` | LLM 上下文长度 | `8192` |
| `JPZH_HF_MIRROR` | HuggingFace 镜像 | `https://hf-mirror.com` |
| `JPZH_GH_PROXY` | GitHub release 代理 | `https://gh-proxy.com/https://github.com` |

## 验证环境

```bash
python scripts/verify_env.py --translate
```

检查 Python / ffmpeg / CUDA / 依赖包 / 权重文件是否就绪。
