import os
import torch
import numpy as np
import logging
import folder_paths
import sys
import gc
from typing import Optional, List, Tuple

from comfy.utils import ProgressBar

# --- 依赖检查 ---
HAS_MODELSCOPE = False
HAS_HUGGINGFACE = False

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

logger = logging.getLogger("ComfyUI-Qwen3-TTS")

# 注册模型路径
if "TTS" not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        "TTS", os.path.join(folder_paths.models_dir, "TTS")
    )


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


class Qwen3TTSLoader:
    """
    负责加载 Qwen3-TTS 模型。
    """

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
                    ["flash_attention_2", "sdpa", "eager"],
                    {"default": "flash_attention_2"},
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
        torch_dtype = dtype_map.get(precision, torch.bfloat16)

        try:
            from comfy.model_management import get_torch_device

            device = get_torch_device()
        except ImportError:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        attn_impl = attn_mode
        if attn_impl == "flash_attention_2":
            if device.type == "cpu" or torch_dtype == torch.float32:
                logger.warning(
                    "Loader: Flash Attention 2 req GPU+Half. Fallback to 'sdpa'."
                )
                attn_impl = "sdpa"

        try:
            model = Qwen3TTSModel.from_pretrained(
                local_model_path,
                device_map=device,
                dtype=torch_dtype,
                attn_implementation=attn_impl,
            )
            model.model_type_str = model_repo

            dev = safe_get_device(model)
            logger.info(f"Loader: Success. Device: {dev} | Dtype: {torch_dtype}")

            return (model,)
        except Exception as e:
            logger.error(f"Loader: Failed to load model: {e}")
            raise RuntimeError(f"Error loading Qwen3-TTS model: {e}")


class Qwen3TTSBaseNode:
    """基类"""

    def _convert_audio_to_comfy(self, wavs: List[np.ndarray], sr: int):
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
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                    "display": "number",
                },
            ),
            "temperature": (
                "FLOAT",
                {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "number",
                },
            ),
            "top_p": (
                "FLOAT",
                {
                    "default": 0.8,
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
        }


class Qwen3TTSCustomVoice(Qwen3TTSBaseNode):
    """CustomVoice 节点"""

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
                        "default": "第一行文本。\n第二行文本(将作为第二段音频生成)。",
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
            },
            "optional": {
                "instruct": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "可选：例如 '用特别愤怒的语气说'",
                    },
                ),
            },
        }
        inputs["optional"].update(cls.get_generation_config())
        return inputs

    # 【修复】补回了漏掉的 RETURN_TYPES 等属性
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(
        self,
        model_obj,
        text,
        speaker,
        language,
        instruct="",
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.8,
        top_k=50,
        repetition_penalty=1.05,
    ):
        self._check_model_compatibility(model_obj, ["CustomVoice"])
        clear_memory()

        real_speaker_id = self.SPEAKER_MAPPING.get(speaker, "Vivian")
        dev = safe_get_device(model_obj)

        text_list = [t.strip() for t in text.split("\n") if t.strip()]
        if not text_list:
            raise ValueError("Input text cannot be empty.")

        batch_size = len(text_list)
        logger.info(f"Generate: Custom Voice Batch Size: {batch_size} | Device: {dev}")

        if language == "Auto":
            lang_input = "auto"
        else:
            lang_input = language

        language_list = [lang_input] * batch_size
        speaker_list = [real_speaker_id] * batch_size
        instruct_val = instruct if instruct.strip() else None
        instruct_list = [instruct_val] * batch_size

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }

        wavs, sr = model_obj.generate_custom_voice(
            text=text_list,
            language=language_list,
            speaker=speaker_list,
            instruct=instruct_list,
            **gen_kwargs,
        )

        logger.info("Batch Generation complete!")
        return self._convert_audio_to_comfy(wavs, sr)


class Qwen3TTSVoiceDesign(Qwen3TTSBaseNode):
    """VoiceDesign 节点"""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model_obj": ("QWEN3_MODEL",),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "哥哥，你回来啦，人家等了你好久好久了，要抱抱！\nIt's in the top drawer... wait, it's empty?",
                    },
                ),
                "voice_instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。\nSpeak in an incredulous tone, but with a hint of panic.",
                        "placeholder": "描述声音特征。行数需与文本一致或仅有一行。",
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
            },
            "optional": {},
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
        voice_instruction,
        language,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.8,
        top_k=50,
        repetition_penalty=1.05,
    ):
        self._check_model_compatibility(model_obj, ["VoiceDesign"])
        clear_memory()

        dev = safe_get_device(model_obj)

        text_list = [t.strip() for t in text.split("\n") if t.strip()]
        if not text_list:
            raise ValueError("Input text cannot be empty.")

        instruct_list = [i.strip() for i in voice_instruction.split("\n") if i.strip()]
        if not instruct_list:
            raise ValueError("Voice instruction cannot be empty.")

        batch_size = len(text_list)
        logger.info(f"Generate: Voice Design Batch Size: {batch_size} | Device: {dev}")

        final_instructs = []
        if len(instruct_list) == 1:
            final_instructs = instruct_list * batch_size
            logger.info("Broadcasting single instruction.")
        elif len(instruct_list) == batch_size:
            final_instructs = instruct_list
            logger.info("Mapping instructions 1:1.")
        else:
            raise ValueError(
                f"Instruct count {len(instruct_list)} != Text count {batch_size}"
            )

        language_list = [language] * batch_size

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }

        wavs, sr = model_obj.generate_voice_design(
            text=text_list,
            language=language_list,
            instruct=final_instructs,
            **gen_kwargs,
        )

        logger.info(f"Voice Design Batch complete!")
        return self._convert_audio_to_comfy(wavs, sr)


class Qwen3TTSVoiceClonePrompt(Qwen3TTSBaseNode):
    """Pre-Compute Prompt 节点"""

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

        waveform = ref_audio["waveform"]
        sample_rate = ref_audio["sample_rate"]
        audio_tensor = waveform[0]
        if audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0)
        else:
            audio_tensor = audio_tensor.squeeze(0)
        ref_audio_numpy = audio_tensor.cpu().numpy()

        try:
            prompt = model_obj.create_voice_clone_prompt(
                ref_audio=(ref_audio_numpy, sample_rate),
                ref_text=ref_text if not x_vector_only else None,
                x_vector_only_mode=x_vector_only,
            )
            pbar.update(2)
            return (prompt,)
        except Exception as e:
            logger.error(f"Failed to create prompt: {e}")
            raise RuntimeError(f"Prompt Error: {e}")


class Qwen3TTSVoiceClone(Qwen3TTSBaseNode):
    """Clone 节点"""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model_obj": ("QWEN3_MODEL",),
                "target_text": (
                    "STRING",
                    {"multiline": True, "default": "我想用这个声音说这句话。"},
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
            },
            "optional": {
                "ref_audio": ("AUDIO",),
                "ref_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Required if using ref_audio (unless X-Vector Only)",
                    },
                ),
                "voice_clone_prompt": ("QWEN3_PROMPT",),
                "enable_x_vector_instant": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label": "Instant X-Vector Mode (Ignore Ref Text)",
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
        target_text,
        target_language,
        ref_audio=None,
        ref_text="",
        voice_clone_prompt=None,
        enable_x_vector_instant=False,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.8,
        top_k=50,
        repetition_penalty=1.05,
    ):

        self._check_model_compatibility(model_obj, ["Base"])
        clear_memory()

        dev = safe_get_device(model_obj)

        text_list = [t.strip() for t in target_text.split("\n") if t.strip()]
        if not text_list:
            if target_text.strip():
                text_list = [target_text.strip()]
            else:
                raise ValueError("Target text cannot be empty.")

        batch_size = len(text_list)
        logger.info(
            f"Generate: Clone Batch Size: {batch_size} on {dev} | Tokens: {max_new_tokens}"
        )

        if "cpu" in str(dev).lower() and max_new_tokens > 512:
            logger.warning("CPU Mode Detected: Reducing max_new_tokens to 512.")
            max_new_tokens = 512

        prompt_item = None
        pbar = ProgressBar(100)
        pbar.update(5)

        if voice_clone_prompt is not None:
            logger.info("Clone: Using cached Prompt.")
            prompt_item = voice_clone_prompt
        elif ref_audio is not None:
            if not enable_x_vector_instant and not ref_text.strip():
                raise ValueError(
                    "Reference Text (ref_text) is required unless X-Vector Only is enabled."
                )

            logger.info(
                f"Clone: Processing ref_audio (Instant)... X-Vector: {enable_x_vector_instant}"
            )
            waveform = ref_audio["waveform"]
            sample_rate = ref_audio["sample_rate"]
            audio_tensor = waveform[0]
            if audio_tensor.shape[0] > 1:
                audio_tensor = torch.mean(audio_tensor, dim=0)
            else:
                audio_tensor = audio_tensor.squeeze(0)
            ref_audio_numpy = audio_tensor.cpu().numpy()

            prompt_item = model_obj.create_voice_clone_prompt(
                ref_audio=(ref_audio_numpy, sample_rate),
                ref_text=ref_text if not enable_x_vector_instant else None,
                x_vector_only_mode=enable_x_vector_instant,
            )
        else:
            raise ValueError("Input missing: need ref_audio OR voice_clone_prompt.")

        pbar.update(20)

        language_list = [target_language] * batch_size

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }

        try:
            wavs, sr = model_obj.generate_voice_clone(
                text=text_list,
                language=language_list,
                voice_clone_prompt=prompt_item,
                **gen_kwargs,
            )

            pbar.update(100)
            logger.info("Generation complete!")
            return self._convert_audio_to_comfy(wavs, sr)

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Error: {e}")


class Qwen3TTSAudioSpeed:
    """
    【新增节点】Audio Speed Control
    Simple resampling-based speed adjustment (Pitch Shift).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "speed": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 2.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "adjust_speed"
    CATEGORY = "Qwen3-TTS"

    def adjust_speed(self, audio, speed):
        if speed == 1.0:
            return (audio,)

        logger.info(f"Adjusting Audio Speed: x{speed}")

        waveform = audio["waveform"]  # [batch, channels, samples]
        original_sr = audio["sample_rate"]

        batch, channels, samples = waveform.shape
        # Calculate new length: faster speed (speed > 1) means fewer samples
        new_samples = int(samples / speed)

        # Use linear interpolation for resampling
        new_waveform = torch.nn.functional.interpolate(
            waveform, size=new_samples, mode="linear", align_corners=False
        )

        return ({"waveform": new_waveform, "sample_rate": original_sr},)
