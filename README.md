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

## 技术栈

| 环节 | 技术 | 说明 |
|---|---|---|
| 音频提取 | ffmpeg | 视频 → 16kHz mono WAV |
| 日语 ASR | faster-whisper (CTranslate2) | Whisper large-v3，float16，GPU |
| 日中翻译 | Sakura-14B-Qwen2beta-v0.9.2 | 轻小说/Galgame 领域日中翻译模型，NSFW 安全 |
| LLM 推理 | llama.cpp (llama-server) | 预编译 CUDA 二进制，OpenAI 兼容 HTTP API |
| 字幕输出 | 自写 SRT 模块 | UTF-8 BOM，标准 SRT 格式 |

## 环境

- Python ≥ 3.10（开发使用 3.11）
- ffmpeg（在 PATH 中，或设置环境变量 `JPZH_FFMPEG`）
- NVIDIA GPU + 驱动（CUDA 12.x，RTX 4090D 24GB 测试通过）
- 依赖包：`faster-whisper`、`ctranslate2`、`requests`

## 安装

```bash
# 1. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 2. 安装依赖（使用国内镜像加速）
pip install faster-whisper ctranslate2 requests -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 下载模型权重和 llama.cpp 二进制
python scripts/download_whisper.py      # Whisper large-v3 (~3GB)
python scripts/download_sakura.py       # Sakura-14B GGUF q4km (~8.7GB)
python scripts/download_llamacpp.py     # llama.cpp CUDA 二进制 (~612MB)
```

## 使用

```bash
# 设置环境（如使用 conda 环境而非安装包）
set PYTHONPATH=src
set PYTHONUTF8=1

# 产出日文字幕（视频同目录，文件名.ja.srt）
python -m jpzh_subtitle video.mkv

# 产出中文字幕（自动启动 llama-server + Sakura，文件名.zh.srt）
python -m jpzh_subtitle video.mkv --translate

# 指定输出路径
python -m jpzh_subtitle video.mkv -o out.srt

# 详细日志
python -m jpzh_subtitle video.mkv -vv

# 使用 CPU（无 GPU 时）
python -m jpzh_subtitle video.mkv --device cpu
```

## 配置（环境变量）

| 变量 | 说明 | 默认 |
|---|---|---|
| `JPZH_FFMPEG` | ffmpeg.exe 路径 | PATH 中查找 |
| `JPZH_ASR_MODEL` | Whisper 模型名/路径 | 本地 `models/whisper-large-v3/`，不存在则 `large-v3` |
| `JPZH_LLM_MODEL` | 翻译 GGUF 权重路径 | `models/sakura/sakura-14b-qwen2beta-v0.9.2-q4km.gguf` |
| `JPZH_MODELS_DIR` | 模型权重目录 | `models/` |
| `HF_ENDPOINT` | HF 镜像（国内下载加速） | `https://hf-mirror.com` |

## 状态

- [x] 阶段 A：音频提取 + 日语 ASR → 日文 `.srt`（✅ 端到端测试通过）
- [x] 阶段 B 代码：Sakura 翻译后端 + llama-server 管理 + pipeline 接入（✅ HTTP 集成测试通过）
- [ ] 阶段 B 端到端测试：视频 → 中文 `.srt`（⏳ 等待 Sakura 模型下载）
- [ ] Phase 3：PotPlayer 扩展集成（AngelScript `.as`，调用本地翻译服务）
