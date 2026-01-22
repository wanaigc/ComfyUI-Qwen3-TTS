# ComfyUI-Qwen3-TTS

一个功能强大的 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 插件，基于阿里云通义实验室开源的 **Qwen3-TTS** 模型，提供高质量的语音合成、音色设计和声音克隆功能。

本插件深度集成了 Qwen3-TTS 的所有核心能力，支持 **批量推理 (Batch Inference)**、**Prompt 复用**、**高级生成参数控制** 以及 **ModelScope/HuggingFace 双源下载**。

## ✨ 核心特性

* **四大核心节点**：
* 🎭 **Custom Voice (预设音色)**：内置 Vivian, Uncle_Fu 等 9 种高质量预设角色，支持多情感控制。
* 🎨 **Voice Design (音色设计)**：通过自然语言描述（如“撒娇的萝莉音”）设计独一无二的声音。
* 🦜 **Voice Clone (声音克隆)**：通过 3秒+ 参考音频复刻目标声音，支持跨语言克隆。
* ⚡ **Pre-Compute Prompt (预计算)**：分离声纹提取与语音生成，极大加速重复生成任务。


* **批量推理 (Batch Inference)**：支持通过换行符输入多行文本，自动并行生成多段音频，速度飞快。
* **智能参数广播**：在批量模式下，自动处理语言、说话人或指令的广播逻辑（一对多或一对一）。
* **高级生成控制**：完整暴露 `max_new_tokens`, `temperature`, `top_p`, `repetition_penalty` 等参数，精准控制生成效果。
* **纯声纹模式 (X-Vector)**：支持仅通过音频进行克隆（无需参考文本），适合处理未知内容的音频。
* **双源自动下载**：支持从 **ModelScope (国内加速)** 或 Hugging Face 自动下载模型，无需手动搬运文件。
* **硬件自适应**：自动检测运行设备，包含 CPU 保护机制（自动降低生成长度防止死机）和显存强制清理。

## 📦 安装指南

### 1. 插件安装

进入您的 ComfyUI `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YourName/ComfyUI-Qwen3-TTS.git
cd ComfyUI-Qwen3-TTS

```

### 2. 依赖安装

请确保您的环境已安装必要的 Python 库：

```bash
pip install -r requirements.txt

```

*依赖列表通常包含：`qwen-tts`, `modelscope`, `soundfile`, `numpy`, `torch` 等。*

### 3. (可选) Flash Attention 加速

如果您使用的是 NVIDIA 显卡，**强烈建议**安装 Flash Attention 2 以获得最佳推理速度：

```bash
pip install flash-attn --no-build-isolation

```

*注：如果不安装，插件会自动降级使用 PyTorch 原生加速 (SDPA)，速度稍慢但兼容性更好。*

## 📥 模型下载与管理

插件会在首次运行时**自动下载**所需模型。默认下载源为 **ModelScope**（适合国内用户）。

模型文件将存储在：
`ComfyUI/models/TTS/Qwen/`

**支持的模型列表：**

* `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (推荐用于克隆)
* `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (推荐用于预设音色)
* `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (推荐用于音色设计)
* *(同时支持 0.6B 版本，显存较小时可选)*

---

## 🚀 节点使用说明

### 1. Qwen3 TTS Loader

这是所有工作流的起点，用于加载模型。

* **Model Repo**: 选择要加载的模型类型。
* 做克隆选 `Base`。
* 用预设音色选 `CustomVoice`。
* 设计音色选 `VoiceDesign`。


* **Download Source**: 选择 `ModelScope` (默认) 或 `HuggingFace`。
* **Precision**: 推荐 `bf16` (30系/40系显卡) 或 `fp16`。
* **Attn Mode**: 推荐 `flash_attention_2`。如果报错，请尝试切换为 `sdpa`。

### 2. Qwen3 Custom Voice (预设音色)

使用官方微调好的高质量角色进行语音合成。

* **Inputs**:
* `text`: 输入文本。**支持多行输入**，每行生成一段音频（批量模式）。
* `speaker`: 选择预设角色（如 Vivian, Uncle_Fu 等）。
* `instruct`: (可选) 情感指导，如 "用悲伤的语气"。


* **Batch Logic**: 如果输入 3 行文本，插件会自动生成 3 段音频并合并输出。

### 3. Qwen3 Voice Design (音色设计)

通过 Prompt 捏造声音。

* **Inputs**:
* `voice_instruction`: 声音描述。例如："体现撒娇稚嫩的萝莉女声，音调偏高"。


* **高级用法**:
* **广播模式**：输入 3 行文本 + 1 行指令 -> 所有文本都用这 1 个声音。
* **对话模式**：输入 3 行文本 + 3 行指令 -> 第一句用第一个声音，第二句用第二个...



### 4. Qwen3 Voice Clone (声音克隆)

核心功能。支持两种工作流：

#### A. 快速模式 (简单)

直接连接 `ref_audio` (参考音频) 和填写 `ref_text` (参考文本内容)。

* **优点**：连线简单。
* **缺点**：每次生成都会重新分析参考音频，速度较慢。

#### B. 高效模式 (推荐)

配合 **Qwen3 Pre-Compute Prompt** 节点使用。

1. 先将音频连入 `Pre-Compute Prompt` 节点，生成 `voice_clone_prompt`。
2. 将输出连入 `Voice Clone` 节点。

* **优点**：参考音频只分析一次。后续修改目标文本进行生成时，**速度极快**。

### 5. 高级参数说明

所有生成节点均包含以下高级参数，点击节点底部的 `Generate` 即可看到：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| **max_new_tokens** | 1024 | 生成的最大长度。文本越长需要设得越大。CPU模式下会被强制限制。 |
| **temperature** | 0.7 | 采样温度。越高越富有情感变化，越低越稳定。 |
| **top_p** | 0.8 | 核采样概率。控制候选词范围。 |
| **repetition_penalty** | 1.05 | 重复惩罚。如果发现复读机现象，请调高此值 (如 1.1)。 |

---

## 🛠️ 常见问题 (FAQ)

**Q: 报错 `AttributeError: 'Qwen3TTSModel' object has no attribute 'dtype'`?**
A: 这是早期版本的 Bug，请确保已更新到最新版插件代码。

**Q: 生成速度极慢，控制台显示 CPU Warning?**
A: 请检查 `Qwen3 TTS Loader` 中的日志输出。如果显示 Device 为 CPU，说明您的 PyTorch 未正确识别 CUDA，或者显存不足导致强制回退。请检查 CUDA 环境安装。

**Q: 批量生成时报错 "Batch Mismatch"?**
A: 请检查 `Voice Design` 节点。如果您输入了 N 行文本，那么 `voice_instruction` 要么只有 1 行（应用到所有），要么必须正好有 N 行（一一对应）。

**Q: 如何进行纯声纹克隆（不知道参考文本）？**
A: 在 `Qwen3 Pre-Compute Prompt` 节点中，勾选 `x_vector_only`。或者在 `Voice Clone` 节点中勾选 `enable_x_vector_instant`。注意这可能会稍微降低音色相似度。

---

## 🙏 致谢

* [Qwen3-TTS Official Repo](https://github.com/QwenLM/Qwen3-TTS)
* [ModelScope](https://www.google.com/search?q=https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base)

---

*Disclaimer: This is a community-developed plugin and is not affiliated with the official Qwen team.*
