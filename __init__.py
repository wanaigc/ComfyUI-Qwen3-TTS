from .nodes import (
    Qwen3TTSLoader,
    Qwen3TTSCustomVoice,
    Qwen3TTSVoiceDesign,
    Qwen3TTSVoiceClone,
    Qwen3TTSVoiceClonePrompt,
    Qwen3TTSAudioSpeed,
)

NODE_CLASS_MAPPINGS = {
    "Qwen3TTSLoader": Qwen3TTSLoader,
    "Qwen3TTSCustomVoice": Qwen3TTSCustomVoice,
    "Qwen3TTSVoiceDesign": Qwen3TTSVoiceDesign,
    "Qwen3TTSVoiceClone": Qwen3TTSVoiceClone,
    "Qwen3TTSVoiceClonePrompt": Qwen3TTSVoiceClonePrompt,
    "Qwen3TTSAudioSpeed": Qwen3TTSAudioSpeed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTSLoader": "Qwen3 TTS Loader",
    "Qwen3TTSCustomVoice": "Qwen3 Custom Voice (Preset)",
    "Qwen3TTSVoiceDesign": "Qwen3 Voice Design (Prompt)",
    "Qwen3TTSVoiceClone": "Qwen3 Voice Clone (Reference)",
    "Qwen3TTSVoiceClonePrompt": "Qwen3 Pre-Compute Prompt",
    "Qwen3TTSAudioSpeed": "Qwen3 Audio Speed Control",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
