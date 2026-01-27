import os
import torch
import numpy as np
import logging
import folder_paths
import sys
import gc
import re
from typing import Optional, List, Tuple, Dict, Any
import torch.nn.functional as F
import tempfile
import soundfile as sf
import types

from comfy.utils import ProgressBar

# --- 依赖检查 ---
HAS_MODELSCOPE = False
HAS_HUGGINGFACE = False
HAS_FUNASR = False
HAS_LIBROSA = False
HAS_FFMPEG_PYTHON = False

try:
    from funasr import AutoModel

    HAS_FUNASR = True
except ImportError:
    pass

try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    pass

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    raise ImportError("Please install the core dependency: pip install qwen-tts")

try:
    from modelscope.hub.snapshot_download import (
        snapshot_download as ms_snapshot_download,
    )

    HAS_MODELSCOPE = True
except ImportError:
    pass

try:
    from huggingface_hub import snapshot_download as hf_snapshot_download

    HAS_HUGGINGFACE = True
except ImportError:
    pass

try:
    import ffmpeg
    HAS_FFMPEG_PYTHON = True
except ImportError:
    pass

logger = logging.getLogger("ComfyUI-Qwen3-TTS")

EMOTION_MAP = {
    # 中文情绪
    "开心": "用愉快、开心的语气说话，充满活力和阳光。",
    "激动": "语气非常激动，语速稍快，充满兴奋感。",
    "生气": "用愤怒、严厉的语气说话，语调生硬且带有攻击性。",
    "难过": "语气低沉、忧伤，带有明显的哀伤和哭腔感。",
    "温柔": "声音轻柔、温婉，充满爱意和关怀。",
    "恐惧": "声音颤抖，语气惊恐不安，呼吸感加重。",
    "冷酷": "语气冰冷、没有任何情感波动，显得疏离而机械。",
    "低语": "用极小的声音说话，像是在耳边轻声细语，充满神秘感。",
    "惊讶": "语气充满震惊和不可思议，音调上扬，带有明显的意外感。",
    "厌恶": "语气充满嫌弃和反感，带有不屑和排斥的情感。",
    "平静": "语气平稳自然，没有明显的情绪波动，清晰且沉稳。",
    # English Emotions (Mapping to detailed instructs)
    "happy": "Speak in a very happy and cheerful tone, full of energy.",
    "angry": "Speak with an angry and stern tone, aggressive and sharp.",
    "sad": "Speak in a sad, low-spirited voice with a hint of sorrow.",
    "whisper": "Speak in a very soft whisper, as if sharing a secret.",
    "fearful": "Speak with a trembling, terrified voice, sounding panic-stricken.",
    "surprised": "Speak with a shocked and incredulous tone, pitch raised.",
    "disgusted": "Speak with a tone of revulsion and strong dislike.",
    "neutral": "Speak in a natural, calm, and steady tone.",
}

if "TTS" not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("TTS", os.path.join(folder_paths.models_dir, "TTS"))


def safe_get_device(model):
    try:
        if hasattr(model, "device"):
            return model.device
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except:
            pass

def apply_qwen3_patches(model, attn_mode="sdpa"):
    if model is None:
        return

    def _safe_normalize(self, audios):
        if isinstance(audios, list):
            items = audios
        elif (
            isinstance(audios, tuple)
            and len(audios) == 2
            and isinstance(audios[0], np.ndarray)
        ):
            items = [audios]
        else:
            items = [audios]

        out = []
        for a in items:
            if a is None:
                continue
            if isinstance(a, str):
                try:
                    wav, sr = self._load_audio_to_np(a)
                    out.append([wav.astype(np.float32), int(sr)])
                except Exception as e:
                    logger.error(f"Failed to load audio file {a}: {e}")
            elif (
                isinstance(a, (tuple, list))
                and len(a) == 2
                and isinstance(a[0], np.ndarray)
            ):
                out.append([a[0].astype(np.float32), int(a[1])])

        for i in range(len(out)):
            wav, sr = out[i][0], out[i][1]
            if wav.ndim > 1:
                out[i][0] = np.mean(wav, axis=-1).astype(np.float32)
        return out

    model._normalize_audio_inputs = types.MethodType(_safe_normalize, model)

    if attn_mode == "sage_attention":
        try:
            from sageattention import sageattn

            def make_sage_forward(orig_forward):
                def sage_forward(*args, **kwargs):
                    if len(args) >= 3:
                        q, k, v = args[0], args[1], args[2]
                        return sageattn(
                            q,
                            k,
                            v,
                            is_causal=False,
                            attn_mask=kwargs.get("attention_mask"),
                        )
                    return orig_forward(*args, **kwargs)

                return sage_forward

            patched_count = 0
            for name, m in model.model.named_modules():
                if "Attention" in type(m).__name__ or "attn" in name.lower():
                    if hasattr(m, "forward"):
                        m.forward = make_sage_forward(m.forward)
                        patched_count += 1
            logger.info(f"Qwen3-TTS: SageAttention patched on {patched_count} modules.")
        except ImportError:
            logger.warning(
                "Qwen3-TTS: 'sageattention' not installed. Running in SDPA mode."
            )
        except Exception as e:
            logger.warning(f"Qwen3-TTS: SageAttention patch failed: {e}")

    logger.info("Qwen3-TTS: Industrial stability patches applied.")


def set_random_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))


class Qwen3TTSLoader:
    def __init__(self):
        self.model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_repo": (
                    [
                        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                    ],
                ),
                "download_source": (
                    ["ModelScope", "HuggingFace"],
                    {"default": "ModelScope"},
                ),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "attn_mode": (
                    ["flash_attention_2", "sage_attention", "sdpa", "eager"],
                    {"default": "sdpa"},
                ),
                "auto_download": (
                    "BOOLEAN",
                    {"default": True, "label": "Auto Download if missing"},
                ),
            }
        }

    RETURN_TYPES = ("QWEN3_MODEL",)
    RETURN_NAMES = ("model_obj",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen3-TTS"

    def load_model(
        self, model_repo, download_source, precision, attn_mode, auto_download
    ):
        clear_memory()
        logger.info(
            f"Loader: Loading {model_repo} | Attn: {attn_mode} | Precision: {precision}"
        )

        tts_path_root = folder_paths.get_folder_paths("TTS")[0]
        local_model_path = os.path.join(tts_path_root, model_repo)

        if not os.path.exists(local_model_path) or not os.listdir(local_model_path):
            if auto_download:
                logger.info(f"Loader: Downloading model to {local_model_path}")
                if download_source == "ModelScope":
                    if not HAS_MODELSCOPE:
                        raise ImportError("Need modelscope installed.")
                    ms_snapshot_download(
                        model_id=model_repo, local_dir=local_model_path
                    )
                elif download_source == "HuggingFace":
                    if not HAS_HUGGINGFACE:
                        raise ImportError("Need huggingface_hub installed.")
                    hf_snapshot_download(
                        repo_id=model_repo,
                        local_dir=local_model_path,
                        local_dir_use_symlinks=False,
                    )
            else:
                raise FileNotFoundError(f"Model not found at {local_model_path}")

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }

        try:
            from comfy.model_management import get_torch_device
            device = get_torch_device()
        except ImportError:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )

        if device.type == "mps" and precision == "fp32":
            logger.warning(
                "MPS device detected: 'fp32' is extremely slow. Forcing 'fp16'."
            )
            torch_dtype = torch.float16
        else:
            torch_dtype = dtype_map.get(precision, torch.bfloat16)

        attn_impl = attn_mode
        if attn_impl == "flash_attention_2":
            if device.type == "cpu" or torch_dtype == torch.float32:
                logger.warning(
                    "Loader: Flash Attention 2 req GPU+Half. Fallback to 'sdpa'."
                )
                attn_impl = "sdpa"
        elif attn_impl == "sage_attention":
            attn_impl = "sdpa"

        try:
            if "Qwen3TTSModel" not in globals():
                raise ImportError("Qwen3TTSModel not imported. Is qwen-tts installed?")

            model = Qwen3TTSModel.from_pretrained(
                local_model_path,
                device_map=device,
                dtype=torch_dtype,
                attn_implementation=attn_impl,
            )
            apply_qwen3_patches(model, attn_mode)

            model.model_type_str = model_repo
            dev = safe_get_device(model)
            logger.info(f"Loader: Success. Device: {dev} | Dtype: {torch_dtype}")

            return (model,)
        except Exception as e:
            logger.error(f"Loader: Failed to load model: {e}")
            raise RuntimeError(f"Error loading Qwen3-TTS model: {e}")


class Qwen3TTSBaseNode:
    def _convert_audio_to_comfy(
        self, wavs: List[np.ndarray], sr: int, concat: bool = False
    ):
        if concat:
            valid_wavs = []
            for w in wavs:
                if w.size > 0:
                    if w.ndim > 1:
                        w = w.squeeze()
                    valid_wavs.append(w)

            if not valid_wavs:
                return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": sr},)

            full_wav = np.concatenate(valid_wavs)
            wavs = [full_wav]

        max_len = max([w.shape[0] if w.ndim == 1 else w.shape[1] for w in wavs])
        batch_size = len(wavs)

        batch_waveform = torch.zeros(batch_size, 1, max_len)

        for i, w in enumerate(wavs):
            tensor_w = torch.from_numpy(w).float()
            if tensor_w.dim() == 1:
                length = tensor_w.shape[0]
                batch_waveform[i, 0, :length] = tensor_w
            else:
                length = tensor_w.shape[1]
                if tensor_w.shape[0] > 1:
                    tensor_w = torch.mean(tensor_w, dim=0)
                batch_waveform[i, 0, :length] = tensor_w

        return ({"waveform": batch_waveform, "sample_rate": sr},)

    def _parse_text_with_pauses(self, text_input):
        input_lines = [t.strip() for t in text_input.split("\n") if t.strip()]
        segments = []
        pause_pattern = re.compile(r"\[(?:pause|p):(\d+(?:\.\d+)?)\]", re.IGNORECASE)

        for line in input_lines:
            last_idx = 0
            for match in pause_pattern.finditer(line):
                text_part = line[last_idx : match.start()].strip()
                if text_part:
                    segments.append(("text", text_part))

                try:
                    duration = float(match.group(1))
                    segments.append(("pause", duration))
                except ValueError:
                    pass

                last_idx = match.end()

            remaining_text = line[last_idx:].strip()
            if remaining_text:
                segments.append(("text", remaining_text))

        return segments

    def _audio_tensor_to_numpy_tuple(
        self, audio_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, int]:
        waveform = audio_data["waveform"]
        sr = audio_data["sample_rate"]

        if waveform.dim() > 1:
            waveform = waveform[0]
            if waveform.dim() > 1:
                waveform = torch.mean(waveform, dim=0)

        wav_np = waveform.cpu().numpy().astype(np.float32)
        if wav_np.size < 1024:
            wav_np = np.pad(wav_np, (0, 1024 - wav_np.size))

        return (wav_np, int(sr))

    def _check_model_compatibility(self, model, required_keywords):
        name = getattr(model, "model_type_str", "")
        if not any(k in name for k in required_keywords):
            error_msg = f"Model Mismatch! Current: '{name}', Need: {required_keywords}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    @classmethod
    def get_generation_config(cls):
        return {
            "max_new_tokens": (
                "INT",
                {
                    "default": 2048,
                    "min": 64,
                    "max": 8192,
                    "step": 64,
                    "display": "number",
                },
            ),
            "temperature": (
                "FLOAT",
                {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "number",
                },
            ),
            "top_p": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "number",
                },
            ),
            "top_k": (
                "INT",
                {"default": 50, "min": 0, "max": 200, "display": "number"},
            ),
            "repetition_penalty": (
                "FLOAT",
                {
                    "default": 1.05,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "number",
                },
            ),
            "subtalker_temperature": (
                "FLOAT",
                {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "number",
                },
            ),
            "subtalker_top_p": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "number",
                },
            ),
            "subtalker_top_k": (
                "INT",
                {"default": 50, "min": 0, "max": 200, "display": "number"},
            ),
        }

    def _pack_gen_kwargs(self, kwargs):
        keys = [
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "subtalker_temperature",
            "subtalker_top_p",
            "subtalker_top_k",
        ]
        res = {k: kwargs.get(k) for k in keys if k in kwargs}

        res["subtalker_dosample"] = True
        res["do_sample"] = True
        return res


class Qwen3TTSCustomVoice(Qwen3TTSBaseNode):
    SPEAKER_MAPPING = {
        "Vivian (Chinese - Bright, Sharp, Young Female)": "Vivian",
        "Serena (Chinese - Warm, Soft, Young Female)": "Serena",
        "Uncle_Fu (Chinese - Deep, Mellow, Mature Male)": "Uncle_Fu",
        "Dylan (Chinese Beijing - Clear, Natural Young Male)": "Dylan",
        "Eric (Chinese Sichuan - Lively, Husky Male)": "Eric",
        "Ryan (English - Rhythmic, Dynamic Male)": "Ryan",
        "Aiden (English - Sunny, Clear American Male)": "Aiden",
        "Ono_Anna (Japanese - Light, Playful Female)": "Ono_Anna",
        "Sohee (Korean - Emotional, Warm Female)": "Sohee",
    }

    @classmethod
    def INPUT_TYPES(cls):
        speaker_options = list(cls.SPEAKER_MAPPING.keys())
        inputs = {
            "required": {
                "model_obj": ("QWEN3_MODEL",),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "你好，我是Vivian。[pause:0.8] 很高兴为你服务！",
                    },
                ),
                "speaker": (speaker_options, {"default": speaker_options[0]}),
                "language": (
                    [
                        "Auto",
                        "Chinese",
                        "English",
                        "Japanese",
                        "Korean",
                        "German",
                        "French",
                        "Russian",
                        "Portuguese",
                        "Spanish",
                        "Italian",
                    ],
                    {"default": "Auto"},
                ),
                "output_mode": (
                    ["Batch (Separate)", "Concatenate (Merge)"],
                    {"default": "Concatenate (Merge)"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "instruct": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "可选：例如 '开心'",
                    },
                ),
            },
        }
        inputs["optional"].update(cls.get_generation_config())
        return inputs

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(
        self,
        model_obj,
        text,
        speaker,
        language,
        output_mode,
        seed,
        instruct="",
        **kwargs,
    ):
        self._check_model_compatibility(model_obj, ["CustomVoice"])
        clear_memory()
        set_random_seed(seed)

        real_speaker_id = self.SPEAKER_MAPPING.get(speaker, "Vivian")
        dev = safe_get_device(model_obj)
        gen_kwargs = self._pack_gen_kwargs(kwargs)

        final_instruct = instruct.strip()
        if final_instruct in EMOTION_MAP:
            final_instruct = EMOTION_MAP[final_instruct]
        if not final_instruct:
            final_instruct = None

        segments = self._parse_text_with_pauses(text)
        if not segments:
            raise ValueError("Input text cannot be empty.")

        logger.info(f"CustomVoice: Generating {len(segments)} segments on {dev}")

        audio_results = []
        sr = 24000
        lang_input = "auto" if language == "Auto" else language

        pbar = ProgressBar(len(segments))

        for i, (seg_type, content) in enumerate(segments):
            pbar.update(i)
            if seg_type == "pause":
                if content > 0:
                    silence_samples = int(content * sr)
                    audio_results.append(np.zeros(silence_samples, dtype=np.float32))
            else:
                try:
                    wavs, current_sr = model_obj.generate_custom_voice(
                        text=[content],
                        language=[lang_input],
                        speaker=[real_speaker_id],
                        instruct=[final_instruct] if final_instruct else None,
                        **gen_kwargs,
                    )
                    sr = current_sr
                    if wavs[0].ndim > 1:
                        audio_results.append(wavs[0].squeeze())
                    else:
                        audio_results.append(wavs[0])
                except Exception as e:
                    logger.error(f"Error segment '{content}': {e}")

        pbar.update(len(segments))

        if not audio_results:
            raise RuntimeError("No audio generated.")

        do_concat = output_mode == "Concatenate (Merge)"
        if do_concat:
            full_audio = np.concatenate(audio_results)
            return self._convert_audio_to_comfy([full_audio], sr, concat=True)
        else:
            return self._convert_audio_to_comfy(audio_results, sr, concat=False)


class Qwen3TTSVoiceDesign(Qwen3TTSBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model_obj": ("QWEN3_MODEL",),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "哥哥，你回来啦。[pause:0.5] 人家等了你好久好久了！，要抱抱！\nIt's in the top drawer... wait, it's empty?",
                    },
                ),
                "voice_instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。\nSpeak in an incredulous tone, but with a hint of panic.",
                        "placeholder": "描述声音特征 (语气/音色)。",
                    },
                ),
                "language": (
                    [
                        "Chinese",
                        "English",
                        "Japanese",
                        "Korean",
                        "German",
                        "French",
                        "Russian",
                        "Portuguese",
                        "Spanish",
                        "Italian",
                    ],
                    {"default": "Chinese"},
                ),
                "output_mode": (
                    ["Batch (Separate)", "Concatenate (Merge)"],
                    {"default": "Concatenate (Merge)"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {},
        }
        inputs["optional"].update(cls.get_generation_config())
        return inputs

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(
        self, model_obj, text, voice_instruction, language, output_mode, seed, **kwargs
    ):
        self._check_model_compatibility(model_obj, ["VoiceDesign"])
        clear_memory()
        set_random_seed(seed)

        dev = safe_get_device(model_obj)
        gen_kwargs = self._pack_gen_kwargs(kwargs)

        instruct_str = voice_instruction.strip()
        if not instruct_str:
            raise ValueError("Voice instruction cannot be empty.")

        segments = self._parse_text_with_pauses(text)
        if not segments:
            raise ValueError("Input text cannot be empty.")

        logger.info(f"VoiceDesign: Generating {len(segments)} segments on {dev}")

        audio_results = []
        sr = 24000
        pbar = ProgressBar(len(segments))

        for i, (seg_type, content) in enumerate(segments):
            pbar.update(i)
            if seg_type == "pause":
                if content > 0:
                    silence_samples = int(content * sr)
                    audio_results.append(np.zeros(silence_samples, dtype=np.float32))
            else:
                try:
                    wavs, current_sr = model_obj.generate_voice_design(
                        text=[content],
                        language=[language],
                        instruct=[instruct_str],
                        **gen_kwargs,
                    )
                    sr = current_sr
                    if wavs[0].ndim > 1:
                        audio_results.append(wavs[0].squeeze())
                    else:
                        audio_results.append(wavs[0])
                except Exception as e:
                    logger.error(f"Error segment '{content}': {e}")

        pbar.update(len(segments))
        if not audio_results:
            raise RuntimeError("No audio generated.")

        do_concat = output_mode == "Concatenate (Merge)"
        if do_concat:
            full_audio = np.concatenate(audio_results)
            return self._convert_audio_to_comfy([full_audio], sr, concat=True)
        else:
            return self._convert_audio_to_comfy(audio_results, sr, concat=False)


class Qwen3TTSVoiceClonePrompt(Qwen3TTSBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_obj": ("QWEN3_MODEL",),
                "ref_audio": ("AUDIO",),
                "x_vector_only": (
                    "BOOLEAN",
                    {"default": False, "label": "X-Vector Only (Ignore Text)"},
                ),
            },
            "optional": {
                "ref_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "参考音频里的具体内容文本。",
                        "placeholder": "If X-Vector Only is enabled, this can be empty.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("QWEN3_PROMPT",)
    RETURN_NAMES = ("voice_clone_prompt",)
    FUNCTION = "create_prompt"
    CATEGORY = "Qwen3-TTS"

    def create_prompt(self, model_obj, ref_audio, x_vector_only, ref_text=""):
        self._check_model_compatibility(model_obj, ["Base"])
        clear_memory()

        if not x_vector_only and not ref_text.strip():
            raise ValueError(
                "When 'X-Vector Only' is disabled, Reference Text (ref_text) is required."
            )

        dev = safe_get_device(model_obj)
        logger.info(f"Prompt: Computing on {dev} | X-Vector Only: {x_vector_only}")

        pbar = ProgressBar(2)
        pbar.update(1)

        wav_np, sr = self._audio_tensor_to_numpy_tuple(ref_audio)

        try:
            prompt = model_obj.create_voice_clone_prompt(
                ref_audio=(wav_np, sr),
                ref_text=ref_text if not x_vector_only else None,
                x_vector_only_mode=x_vector_only,
            )
            pbar.update(2)
            return (prompt,)
        except Exception as e:
            logger.error(f"Failed to create prompt: {e}")
            raise RuntimeError(f"Prompt Error: {e}")


class Qwen3TTSVoiceClone(Qwen3TTSBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model_obj": ("QWEN3_MODEL",),
                "target_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "我想用这个声音说这句话。[pause:0.5] 真的很有趣！",
                    },
                ),
                "target_language": (
                    [
                        "Chinese",
                        "English",
                        "Japanese",
                        "Korean",
                        "German",
                        "French",
                        "Russian",
                        "Portuguese",
                        "Spanish",
                        "Italian",
                    ],
                    {"default": "Chinese"},
                ),
                "output_mode": (
                    ["Batch (Separate)", "Concatenate (Merge)"],
                    {"default": "Concatenate (Merge)"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "ref_audio": ("AUDIO",),
                "ref_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Ref text required unless X-Vector Only is True",
                    },
                ),
                "voice_clone_prompt": ("QWEN3_PROMPT",),
                "instruct": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "可选：输入 '开心', 'sad' 或完整描述。",
                    },
                ),
                "enable_x_vector_instant": (
                    "BOOLEAN",
                    {"default": False, "label": "Instant X-Vector Mode"},
                ),
            },
        }
        inputs["optional"].update(cls.get_generation_config())
        return inputs

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(
        self,
        model_obj,
        target_text,
        target_language,
        output_mode,
        seed,
        ref_audio=None,
        ref_text="",
        voice_clone_prompt=None,
        instruct="",
        enable_x_vector_instant=False,
        **kwargs,
    ):
        self._check_model_compatibility(model_obj, ["Base"])
        clear_memory()
        set_random_seed(seed)

        dev = safe_get_device(model_obj)
        gen_kwargs = self._pack_gen_kwargs(kwargs)

        final_instruct = instruct.strip()
        if final_instruct in EMOTION_MAP:
            final_instruct = EMOTION_MAP[final_instruct]
        if not final_instruct:
            final_instruct = None

        prompt_item = None
        if voice_clone_prompt is not None:
            logger.info("Clone: Using cached Prompt.")
            prompt_item = voice_clone_prompt
        elif ref_audio is not None:
            if not enable_x_vector_instant and not ref_text.strip():
                raise ValueError("Ref text required for non-instant mode.")
            ref_wav, ref_sr = self._audio_tensor_to_numpy_tuple(ref_audio)
            prompt_item = model_obj.create_voice_clone_prompt(
                ref_audio=(ref_wav, ref_sr),
                ref_text=ref_text if not enable_x_vector_instant else None,
                x_vector_only_mode=enable_x_vector_instant,
            )
        else:
            raise ValueError("Input missing: need ref_audio OR voice_clone_prompt.")

        segments = self._parse_text_with_pauses(target_text)
        if not segments:
            raise ValueError("Target text cannot be empty.")

        logger.info(f"VoiceClone: Generating {len(segments)} segments on {dev}")

        audio_results = []
        sr = 24000
        pbar = ProgressBar(len(segments))

        for i, (seg_type, content) in enumerate(segments):
            pbar.update(i)
            if seg_type == "pause":
                if content > 0:
                    silence_samples = int(content * sr)
                    audio_results.append(np.zeros(silence_samples, dtype=np.float32))
            else:
                try:
                    wavs, current_sr = model_obj.generate_voice_clone(
                        text=[content],
                        language=[target_language],
                        voice_clone_prompt=prompt_item,
                        instruct=[final_instruct] if final_instruct else None,
                        **gen_kwargs,
                    )
                    sr = current_sr
                    if wavs[0].ndim > 1:
                        audio_results.append(wavs[0].squeeze())
                    else:
                        audio_results.append(wavs[0])
                except Exception as e:
                    logger.error(f"Error segment '{content}': {e}")

        pbar.update(len(segments))
        if not audio_results:
            raise RuntimeError("No audio generated.")

        do_concat = output_mode == "Concatenate (Merge)"
        if do_concat:
            full_audio = np.concatenate(audio_results)
            return self._convert_audio_to_comfy([full_audio], sr, concat=True)
        else:
            return self._convert_audio_to_comfy(audio_results, sr, concat=False)


class Qwen3TTSAudioSpeed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "speed": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
                "method": (
                    [
                        "FFmpeg (atempo) - Best for Speech",
                        "Time Stretch (Librosa)",
                        "Resampling (Pitch Shift)",
                    ],
                    {"default": "FFmpeg (atempo) - Best for Speech"},
                ),
                "channel_mode": (
                    ["Keep Original", "Force Mono", "Force Stereo"],
                    {"default": "Keep Original"},
                ),
                "n_fft": (
                    [2048, 4096, 8192],
                    {"default": 4096, "label": "Quality (Librosa only)"},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "adjust_speed"
    CATEGORY = "Qwen3-TTS"

    def adjust_speed(self, audio, speed, method, n_fft, channel_mode):
        waveform = audio["waveform"].clone()
        original_sr = audio["sample_rate"]

        if waveform.dim() == 3 and waveform.shape[-1] < 10 and waveform.shape[-2] > 100:
            waveform = waveform.permute(0, 2, 1)

        if channel_mode == "Force Mono":
            if waveform.shape[1] > 1:
                waveform = torch.mean(waveform, dim=1, keepdim=True)
        elif channel_mode == "Force Stereo":
            if waveform.shape[1] == 1:
                waveform = waveform.repeat(1, 2, 1)

        if speed == 1.0:
            return ({"waveform": waveform, "sample_rate": original_sr},)

        processed_input_audio = {"waveform": waveform, "sample_rate": original_sr}
        wav_numpy, sr = Qwen3TTSBaseNode()._audio_tensor_to_numpy_tuple(
            processed_input_audio
        )

        if "FFmpeg" in method:
            if not HAS_FFMPEG_PYTHON:
                print("⚠️ 'ffmpeg-python' lib not found. Fallback to Librosa.")
                method = "Time Stretch (Librosa)"
            else:
                temp_in = ""
                temp_out = ""
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ) as f_in, tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ) as f_out:
                        temp_in = f_in.name
                        temp_out = f_out.name

                    sf.write(temp_in, wav_numpy, sr)

                    stream = ffmpeg.input(temp_in)
                    curr_speed = speed

                    while curr_speed > 2.0:
                        stream = stream.filter("atempo", 2.0)
                        curr_speed /= 2.0
                    while curr_speed < 0.5:
                        stream = stream.filter("atempo", 0.5)
                        curr_speed /= 0.5
                    if abs(curr_speed - 1.0) > 0.01:
                        stream = stream.filter("atempo", curr_speed)

                    stream.output(temp_out).run(overwrite_output=True, quiet=True)

                    new_wav, new_sr = sf.read(temp_out)

                    if new_wav.ndim == 1:
                        new_wav = new_wav[np.newaxis, :]
                    else:
                        new_wav = new_wav.T

                    out_tensor = torch.from_numpy(new_wav).float().unsqueeze(0)
                    return ({"waveform": out_tensor, "sample_rate": new_sr},)

                except Exception as e:
                    error_str = str(e)
                    if hasattr(e, "stderr") and e.stderr:
                        error_str += f" | Details: {e.stderr.decode()}"
                    print(f"❌ FFmpeg error: {error_str}. Falling back to Librosa.")
                finally:
                    for p in [temp_in, temp_out]:
                        if p and os.path.exists(p):
                            try:
                                os.remove(p)
                            except:
                                pass

        if "Resampling" in method:
            samples = wav_numpy.shape[-1]
            new_samples = int(samples / speed)
            wav_tensor = torch.from_numpy(wav_numpy).unsqueeze(0).unsqueeze(0)
            new_wav = F.interpolate(
                wav_tensor, size=new_samples, mode="linear", align_corners=False
            )
            return ({"waveform": new_wav.squeeze(0), "sample_rate": sr},)

        if not HAS_LIBROSA:
            print("⚠️ Librosa not found. Fallback to Resampling.")
            return self.adjust_speed(
                processed_input_audio,
                speed,
                "Resampling (Pitch Shift)",
                n_fft,
                "Keep Original",
            )

        y_stretched = librosa.effects.time_stretch(wav_numpy, rate=speed, n_fft=n_fft)

        out_tensor = torch.from_numpy(y_stretched).float()
        if out_tensor.ndim == 1:
            out_tensor = out_tensor.unsqueeze(0)
        out_tensor = out_tensor.unsqueeze(0)

        return ({"waveform": out_tensor, "sample_rate": sr},)


class Qwen3TTSPromptManager:
    @classmethod
    def INPUT_TYPES(cls):
        output_dir = folder_paths.get_output_directory()
        save_dir = os.path.join(output_dir, "qwen3tts")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        files = [f for f in os.listdir(save_dir) if f.endswith(".qwen3tts")]
        file_list = sorted(files) if files else ["no_prompts_found"]

        return {
            "required": {
                "mode": (["Save", "Load"],),
                "load_file": (file_list,),
                "save_filename": ("STRING", {"default": "my_voice_01"}),
            },
            "optional": {
                "voice_clone_prompt": ("QWEN3_PROMPT",),
            },
        }

    RETURN_TYPES = ("QWEN3_PROMPT",)
    RETURN_NAMES = ("voice_clone_prompt",)
    FUNCTION = "manage_prompt"
    CATEGORY = "Qwen3-TTS"

    def manage_prompt(self, mode, load_file, save_filename, voice_clone_prompt=None):
        output_dir = folder_paths.get_output_directory()
        save_dir = os.path.join(output_dir, "qwen3tts")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        if mode == "Save":
            if voice_clone_prompt is None:
                raise ValueError(
                    "Save mode requires a connected 'voice_clone_prompt' input."
                )

            name = save_filename.strip() or "unnamed"
            if not name.endswith(".qwen3tts"):
                name += ".qwen3tts"
            save_path = os.path.join(save_dir, name)

            torch.save(voice_clone_prompt, save_path)
            logger.info(f"Prompt saved to: {save_path}")

            return (voice_clone_prompt,)

        else:
            if load_file == "no_prompts_found":
                raise ValueError("No history files found. Please save one first.")

            load_path = os.path.join(save_dir, load_file)
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"File not found: {load_path}")

            try:
                data = torch.load(load_path, weights_only=False)
            except TypeError:
                data = torch.load(load_path)
            logger.info(f"Prompt loaded from: {load_path}")
            return (data,)


class Qwen3TTSAudioPostProcess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fade_in_ms": ("INT", {"default": 10}),
                "fade_out_ms": ("INT", {"default": 50}),
                "target_sample_rate": ([24000, 44100, 48000], {"default": 44100}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "Qwen3-TTS"

    def process(self, audio, fade_in_ms, fade_out_ms, target_sample_rate):
        waveform = audio["waveform"].clone()
        sr = audio["sample_rate"]
        fi = int((fade_in_ms / 1000.0) * sr)
        fo = int((fade_out_ms / 1000.0) * sr)
        if fi > 0:
            waveform[..., :fi] *= torch.linspace(0.0, 1.0, fi)
        if fo > 0:
            waveform[..., -fo:] *= torch.linspace(1.0, 0.0, fo)
        if sr != target_sample_rate:
            waveform = F.interpolate(
                waveform,
                size=int(waveform.shape[-1] * target_sample_rate / sr),
                mode="linear",
                align_corners=False,
            )
            sr = target_sample_rate
        return ({"waveform": waveform, "sample_rate": sr},)


class Qwen3TTSRoleBank:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "prev_role_bank": ("QWEN3_ROLE_BANK",),
                "role_name_1": ("STRING", {"default": "Role1"}),
                "prompt_1": ("QWEN3_PROMPT",),
                "role_name_2": ("STRING", {"default": "Role2"}),
                "prompt_2": ("QWEN3_PROMPT",),
                "role_name_3": ("STRING", {"default": "Role3"}),
                "prompt_3": ("QWEN3_PROMPT",),
            },
        }

    RETURN_TYPES = ("QWEN3_ROLE_BANK",)
    RETURN_NAMES = ("role_bank",)
    FUNCTION = "create"
    CATEGORY = "Qwen3-TTS"

    def create(self, prev_role_bank=None, **kwargs):
        bank = prev_role_bank.copy() if prev_role_bank else {}
        for i in range(1, 4):
            name = kwargs.get(f"role_name_{i}", f"Role{i}")
            prompt = kwargs.get(f"prompt_{i}")

            if prompt is not None:
                bank[name] = prompt

        return (bank,)


class Qwen3TTSScriptProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "角色A: [开心] 你好！\n[pause:1.0]\n角色B: [冷酷] 没空。",
                    },
                ),
                "default_instruct": ("STRING", {"default": "正常语气说话。"}),
            }
        }

    RETURN_TYPES = ("LIST", "LIST", "LIST", "LIST")
    RETURN_NAMES = ("texts", "instructs", "roles", "pauses")
    FUNCTION = "parse"
    CATEGORY = "Qwen3-TTS"

    def parse(self, script, default_instruct):
        lines = [l.strip() for l in script.split("\n") if l.strip()]
        res = {"texts": [], "instructs": [], "roles": [], "pauses": []}
        for line in lines:
            pm = re.match(r"\[pause:(\d+\.?\d*)\]", line)
            if pm:
                res["texts"].append("")
                res["instructs"].append("")
                res["roles"].append("PAUSE")
                res["pauses"].append(float(pm.group(1)))
                continue
            role, content = (
                line.split(":", 1)
                if ":" in line
                else line.split("：", 1) if "：" in line else ("Default", line)
            )
            tags = re.findall(r"\[([^\]]+)\]", content)
            clean_text = re.sub(r"\[([^\]]+)\]", "", content).strip()
            instr = (
                " ".join([EMOTION_MAP.get(t, f"以{t}的语气说话。") for t in tags])
                if tags
                else default_instruct
            )
            res["texts"].append(clean_text)
            res["instructs"].append(instr)
            res["roles"].append(role.strip())
            res["pauses"].append(0.0)
        return (res["texts"], res["instructs"], res["roles"], res["pauses"])


class Qwen3TTSAdvancedDialogue(Qwen3TTSBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model_obj": ("QWEN3_MODEL",),
                "texts": ("LIST",),
                "instructs": ("LIST",),
                "roles": ("LIST",),
                "pauses": ("LIST",),
                "role_bank": ("QWEN3_ROLE_BANK",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }
        inputs["required"].update(cls.get_generation_config())
        return inputs

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(
        self, model_obj, texts, instructs, roles, pauses, role_bank, seed, **kwargs
    ):
        self._check_model_compatibility(model_obj, ["Base"])
        clear_memory()
        set_random_seed(seed)

        all_segments = []
        sr = 24000

        gen_kwargs = self._pack_gen_kwargs(kwargs)

        for i in range(len(texts)):
            if roles[i] == "PAUSE":
                silence_duration = pauses[i]
                if silence_duration > 0:
                    silence = np.zeros(int(silence_duration * sr), dtype=np.float32)
                    all_segments.append(silence)
                continue

            if roles[i] not in role_bank:
                print(
                    f"Warning: Role '{roles[i]}' not found in RoleBank, using default."
                )
                if role_bank:
                    prompt = list(role_bank.values())[0]
                else:
                    raise ValueError(
                        f"RoleBank is empty, cannot generate voice for {roles[i]}"
                    )
            else:
                prompt = role_bank[roles[i]]

            wavs, _ = model_obj.generate_voice_clone(
                text=[texts[i]],
                language=["auto"],
                voice_clone_prompt=prompt,
                instruct=[instructs[i]],
                **gen_kwargs,
            )

            audio_segment = wavs[0]
            if audio_segment.ndim > 1:
                audio_segment = audio_segment.squeeze()

            all_segments.append(audio_segment)

        if not all_segments:
            raise ValueError("No audio generated. Check your script input.")

        full_audio = np.concatenate(all_segments)

        return self._convert_audio_to_comfy([full_audio], sr)


class Qwen3TTSSenseVoiceASR(Qwen3TTSBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model_id": (["iic/SenseVoiceSmall"],),
                "language": (
                    ["auto", "zn", "en", "ja", "ko", "yue"],
                    {"default": "auto"},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "suggested_instruct")
    FUNCTION = "recognize"
    CATEGORY = "Qwen3-TTS"

    def recognize(self, audio, model_id, language):
        if not HAS_FUNASR:
            raise ImportError("Please install funasr: pip install funasr torchaudio")
        if not HAS_MODELSCOPE:
            raise ImportError("Need modelscope installed for SenseVoice.")

        tts_path_root = folder_paths.get_folder_paths("TTS")[0]
        model_dir = os.path.join(tts_path_root, "SenseVoiceSmall")
        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            logger.info(f"ASR: Downloading SenseVoiceSmall to {model_dir}")
            ms_snapshot_download(model_id=model_id, local_dir=model_dir)

        try:
            from comfy.model_management import get_torch_device

            device_obj = get_torch_device()
        except ImportError:
            device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        device_str = str(device_obj)

        logger.info(f"ASR: Loading SenseVoice model on {device_str}...")

        model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            device=device_str,
            disable_update=True,
        )

        wav_np, sr = self._audio_tensor_to_numpy_tuple(audio)
        temp_path = ""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name

        try:
            sf.write(temp_path, wav_np, sr)

            logger.info(f"ASR: Transcribing audio ({language})...")
            res = model.generate(
                input=temp_path,
                cache={},
                language=language,
                use_itn=True,
                batch_size_s=60,
            )

            text_result = ""
            emotion_tag = ""
            instruct = ""

            if res and isinstance(res, list):
                raw_text = res[0].get("text", "")
                clean_text = re.sub(r"<\|.*?\|>", "", raw_text).strip()
                text_result = clean_text

                if "<|HAPPY|>" in raw_text:
                    emotion_tag = "happy"
                elif "<|ANGRY|>" in raw_text:
                    emotion_tag = "angry"
                elif "<|SAD|>" in raw_text:
                    emotion_tag = "sad"

                if emotion_tag in EMOTION_MAP:
                    instruct = EMOTION_MAP[emotion_tag]
                elif emotion_tag:
                    instruct = f"Speak in a {emotion_tag} tone."

            logger.info(f"ASR Result: {text_result} | Emotion: {emotion_tag}")
            return (text_result, instruct)

        except Exception as e:
            logger.error(f"ASR Error: {e}")
            return ("", "")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if "model" in locals():
                del model
            clear_memory()
