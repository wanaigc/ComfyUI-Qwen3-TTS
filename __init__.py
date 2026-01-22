from .nodes import (
    Qwen3TTSLoader,
    Qwen3TTSCustomVoice,
    Qwen3TTSVoiceDesign,
    Qwen3TTSVoiceClone,
    Qwen3TTSVoiceClonePrompt,
)

NODE_CLASS_MAPPINGS = {
    "Qwen3TTSLoader": Qwen3TTSLoader,
    "Qwen3TTSCustomVoice": Qwen3TTSCustomVoice,
    "Qwen3TTSVoiceDesign": Qwen3TTSVoiceDesign,
    "Qwen3TTSVoiceClone": Qwen3TTSVoiceClone,
    "Qwen3TTSVoiceClonePrompt": Qwen3TTSVoiceClonePrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTSLoader": "Qwen3 TTS Loader",
    "Qwen3TTSCustomVoice": "Qwen3 Custom Voice (Preset)",
    "Qwen3TTSVoiceDesign": "Qwen3 Voice Design (Prompt)",
    "Qwen3TTSVoiceClone": "Qwen3 Voice Clone (Reference)",
    "Qwen3TTSVoiceClonePrompt": "Qwen3 Pre-Compute Prompt",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
