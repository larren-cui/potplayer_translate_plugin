# jpzh-subtitle

离线日中字幕翻译工具：给定一个视频文件，自动生成日文→中文翻译的字幕（`.srt`）。

全程离线，使用本地 GPU（NVIDIA CUDA）完成日语语音识别与翻译。

## 工作流程

```
视频 ──ffmpeg抽音轨──► 16kHz WAV
   │
   ▼
faster-whisper (large-v3, GPU) ──► 日文转写 + 逐句时间戳
   │
   ▼
本地 LLM (Sakura-14B, GPU) ──► 日文 → 中文
   │
   ▼
.srt 字幕（中文，时间戳对齐原语音）
```

## 环境

- Python ≥ 3.10
- ffmpeg（在 PATH 中，或设置环境变量 `JPZH_FFMPEG`）
- NVIDIA GPU + 驱动（CUDA 12.x）

## 安装

```bash
# 创建虚拟环境并安装依赖
uv venv .venv
.venv\Scripts\activate
uv pip install -e .

# 翻译模型（阶段 B）需要 llama-cpp-python（CUDA 版），见下方说明
```

## 使用

```bash
# 产出日文字幕（视频同目录，文件名. ja.srt）
python -m jpzh_subtitle video.mkv

# 产出中文字幕（阶段 B，文件名.zh.srt）
python -m jpzh_subtitle video.mkv --translate

# 指定输出路径
python -m jpzh_subtitle video.mkv -o out.srt

# 详细日志
python -m jpzh_subtitle video.mkv -vv
```

## 配置（环境变量）

| 变量 | 说明 | 默认 |
|---|---|---|
| `JPZH_FFMPEG` | ffmpeg.exe 路径 | PATH 中查找 |
| `JPZH_ASR_MODEL` | Whisper 模型名/路径 | `large-v3` |
| `JPZH_LLM_MODEL` | 翻译 GGUF 权重路径 | `models/sakura-14b.gguf` |
| `JPZH_MODELS_DIR` | 模型权重目录 | `models/` |
| `HF_ENDPOINT` | HF 镜像（国内下载加速） | `https://hf-mirror.com` |

## 状态

- [x] 阶段 A：音频提取 + 日语 ASR → 日文 `.srt`
- [ ] 阶段 B：接入 Sakura-14B → 中文 `.srt`
