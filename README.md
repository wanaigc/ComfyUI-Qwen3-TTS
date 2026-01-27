# 🚀 ComfyUI-Qwen3-TTS

基于阿里巴巴 Qwen 团队开源的 **Qwen3-TTS** 模型，为 ComfyUI 打造的工业级语音合成方案。

本插件基于Qwen3-TTS语音克隆和多角色配音。具备脚本驱动的无限多角色配音功能、情感感知的语音转语音（ASR）技术，支持变速不变调以及高效的配音资产管理能力。

![Workflow Preview](docs/image/workflow.png)

---

## 📋 更新日志

* **2026-01-27**：**升级功能**：节点增强情绪， Voice Clone、 Custom Voice、Voice Design支持[pause:0.5]停顿，注：支持保存音色，无限角色配音。
* **2026-01-25**：**升级功能**：Qwen3TTS AudioSpeed节点新增 channel_mode 参数、Method选项新增 FFmpeg (atempo)：人声变速首选（推荐）、Time Stretch (Librosa)：变速不变调、Resampling (Pitch Shift)：变速变调，并增加 FFT参数。
* **2026-01-24**：**核心升级**：新增 `RoleBank`角色 与 `AdvancedDialogue` 多角色配音节点，新增 `ScriptProcessor`，支持 `[情感标签]` 与 `[pause:停顿]` 自动解析，并增加**Seed 随机种子**控制、**输出模式(合并/分段)**、**音频转文本**功能。
* **2026-01-23**：🔥 **基础功能**：为所有生成节点添加完整采样控制（`top_p`, `top_k`, `temperature`, `repetition_penalty`）。

---

## ✨ 核心特性

* **四大基础能力**：
    * 🎭 **Custom Voice (预设音色)**：内置 Vivian, Uncle_Fu 等 9 种高质量预设角色。
    * 🎨 **Voice Design (音色设计)**：通过自然语言描述（如“撒娇的萝莉音”）设计独一无二的声音。
    * 🦜 **Voice Clone (声音克隆)**：支持 3秒+ 参考音频复刻，包含 **X-Vector 纯声纹模式**。
    * ⚡ **Pre-Compute (预计算)**：分离声纹提取与生成，修改文本时无需重复分析音频。

* **进阶多任务设计 (New)**：
    * 🧠 **智能情感解析**：识别文本中的 `[开心]`、`[冷酷]` 等情感标签。
    * ⏱️ **高精停顿控制**：支持脚本内 `[pause:1.2]` 毫秒级停顿解析，精准把握节奏。
    * 👥 **多角色编排**：通过角色库 (RoleBank) 实现单节点生成多角色对话，自动拼接音频。
    * 💾 **资产持久化**：支持将克隆好的 Prompt 保存为文件，随时复用。

* **工程级优化**：
    * **完全可复现**：所有生成节点支持 **Seed** 控制，锁定随机种子即可复现完美结果。
    * **批量推理**：支持换行符输入多行文本并行生成。
    * **双源下载**：支持 ModelScope (国内) / HuggingFace 自动切换。
    * **稳健运行**：内置显存强制清理与 CPU 保护机制。

---

## 📦 安装指南

### 1. 插件安装
进入您的 ComfyUI `custom_nodes` 目录：
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/wanaigc/ComfyUI-Qwen3-TTS.git


```

### 2. 依赖安装

请确保您的环境已安装必要的 Python 库：

```bash
cd ComfyUI-Qwen3-TTS
pip install -r requirements.txt

```

*主要依赖：`qwen-tts`, `transformers`, `modelscope`, `huggingface-hub`，`funasr`，`soundfile`。*

### 3. (可选) Flash Attention 加速

如果您使用的是 NVIDIA 显卡，**强烈建议**安装 Flash Attention 2 以获得最佳推理速度：

```bash
pip install flash-attn --no-build-isolation

```

*注：如果不安装，插件会自动降级使用 PyTorch 原生加速 (SDPA)，兼容性更好但速度略慢。*

---

## 📥 模型下载与管理

插件会在首次运行时**自动下载**所需模型。默认下载源为 **ModelScope**（适合国内用户）。

模型文件将存储在：`ComfyUI/models/TTS/Qwen/`

**推荐下载项：**

* `Qwen3-TTS-12Hz-1.7B-Base` (万能模型，用于克隆和对话)
* `Qwen3-TTS-12Hz-1.7B-CustomVoice` (用于预设音色)

---

## 🧩 新功能设计与使用说明

### 1. 智能脚本系统 (Smart Script System)

这是本插件最强大的进阶功能，由三个节点协同工作：

* **ScriptProcessor (脚本处理器)**：
* **功能**：解析剧本。
* **格式支持**：
```text
角色A: [开心] 这里的风景真好啊！
[pause:1.5]
角色B: [冷酷] 是吗？我倒觉得很一般。

```


* **输出**：自动分离出文本列表、情感指令列表、角色列表和停顿时间。


* **RoleBank (角色库)**：
* **功能**：将多个 `Voice Clone Prompt` 注册为具名角色（如 "角色A", "角色B"）。
* **作用**：让对话引擎知道 "角色A" 对应哪个声音特征。


* **AdvancedDialogue (高级对话引擎)**：
* **功能**：接收处理器的列表和角色库，全自动生成混排音频。
* **优势**：一次生成整段对话，自动处理不同角色的音色切换和中间的停顿。



### 2. 音频工作室 (Audio Studio)

针对 TTS 生成的常见问题提供的专业处理节点：

* **Audio Post-Process (音频后处理)**：
* **消除爆音**：设置 `fade_in_ms` (建议 10-20ms) 可完美消除开头“咔哒”声。
* **重采样**：支持将音频统一转换为 44.1kHz 或 48kHz，适配视频剪辑标准。


* **Prompt Manager (资产管理)**：
* **功能**：将克隆好的 `Voice Clone Prompt` 保存为 `.qwen3tts` 文件。
* **场景**：下次使用该声音时直接 Load，无需再找参考音频，实现“声音资产化”。



---

## 🚀 基础操作模式指南

### A. 快速模式 (简单)

直接在 **Voice Clone** 节点连接 `ref_audio` (参考音频) 和填写 `ref_text`。

* **优点**：连线简单。
* **缺点**：每次生成都会重新分析参考音频，速度较慢。

### B. 高效模式 (推荐)

配合 **Pre-Compute Prompt** 节点使用。

1. 先将音频连入 `Pre-Compute Prompt` 节点，生成 `voice_clone_prompt`。
2. 将输出连入 `Voice Clone` 节点。

* **优点**：参考音频只分析一次。后续修改目标文本进行生成时，**速度极快**。

---

## ⚙️ 高级生成参数说明

所有生成节点均包含以下参数，点击 `Generate` 即可调整：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| **seed** | 0 | **随机种子**。固定此值可确保每次生成的结果完全一致。 |
| **max_new_tokens** | 2048 | 生成最大长度。CPU模式下会被强制限制以防卡死。 |
| **temperature** | 0.90 | 采样温度。越高情感越丰富（但也越不可控），越低越稳定。 |
| **top_p** | 1.00 | 核采样概率。控制候选词范围。 |
| **repetition_penalty** | 1.05 | 重复惩罚。如果发现复读机现象，请调高此值 (如 1.1)。 |

---

## 🛠️ 常见问题 (FAQ)

**Q: 报错 `Out of Memory` (显存溢出)？**
A: 插件已内置 `clear_memory`，但如果显存仍不足，请在加载器中选择 `bf16` 精度，并减小 `batch_size`（如果是对话节点）。

**Q: 批量生成时报错 "Batch Mismatch"?**
A: 在 **Voice Design** 节点中，如果您输入了 N 行文本，`voice_instruction` 要么只有 1 行（广播给所有），要么必须正好有 N 行（一一对应）。

**Q: 如何进行纯声纹克隆（不知道参考文本）？**
A: 在 `Pre-Compute Prompt` 节点中勾选 `x_vector_only`。注意这可能会稍微降低音色相似度。

**Q: 生成的音频开头有噪音？**
A: 请连接 **Audio Post-Process** 节点，并设置 `fade_in_ms` 为 10 或 20。

---

## 🙏 致谢与许可

* **模型归属**：阿里巴巴 Qwen 团队官方开源仓库。
* **许可证**：本项目采用 Apache License 2.0 许可证。
