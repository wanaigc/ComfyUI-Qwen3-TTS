__version__ = "1.0.3"

from .nodes import (
    Qwen3TTSLoader,
    Qwen3TTSCustomVoice,
    Qwen3TTSVoiceDesign,
    Qwen3TTSVoiceClone,
    Qwen3TTSVoiceClonePrompt,
    Qwen3TTSAudioSpeed,
    Qwen3TTSPromptManager,
    Qwen3TTSAudioPostProcess,
    Qwen3TTSRoleBank,
    Qwen3TTSScriptProcessor,
    Qwen3TTSAdvancedDialogue,
    Qwen3TTSSenseVoiceASR,
)

NODE_CLASS_MAPPINGS = {
    "Qwen3TTSLoader": Qwen3TTSLoader,
    "Qwen3TTSCustomVoice": Qwen3TTSCustomVoice,
    "Qwen3TTSVoiceDesign": Qwen3TTSVoiceDesign,
    "Qwen3TTSVoiceClone": Qwen3TTSVoiceClone,
    "Qwen3TTSVoiceClonePrompt": Qwen3TTSVoiceClonePrompt,
    "Qwen3TTSAudioSpeed": Qwen3TTSAudioSpeed,
    "Qwen3TTSPromptManager": Qwen3TTSPromptManager,
    "Qwen3TTSAudioPostProcess": Qwen3TTSAudioPostProcess,
    "Qwen3TTSScriptProcessor": Qwen3TTSScriptProcessor,
    "Qwen3TTSAdvancedDialogue": Qwen3TTSAdvancedDialogue,
    "Qwen3TTSRoleBank": Qwen3TTSRoleBank,
    "Qwen3TTSSenseVoiceASR": Qwen3TTSSenseVoiceASR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTSLoader": "Qwen3-TTS Loader 🚀",
    "Qwen3TTSCustomVoice": "Qwen3-TTS Custom Voice 🎭",
    "Qwen3TTSVoiceDesign": "Qwen3-TTS Voice Design 🎨",
    "Qwen3TTSVoiceClone": "Qwen3-TTS Voice Clone 👥",
    "Qwen3TTSVoiceClonePrompt": "Qwen3-TTS Clone Prompt 🧠",
    "Qwen3TTSAudioSpeed": "Qwen3-TTS Audio Speed ⚡",
    "Qwen3TTSPromptManager": "Qwen3-TTS Prompt Manager 📁",
    "Qwen3TTSAudioPostProcess": "Qwen3-TTS Audio Post-Process ✨",
    "Qwen3TTSScriptProcessor": "Qwen3-TTS Smart Script Processor 🧠",
    "Qwen3TTSAdvancedDialogue": "Qwen3-TTS Advanced Dialogue Engine 🎭",
    "Qwen3TTSRoleBank": "Qwen3-TTS Role Bank 👥",
    "Qwen3TTSSenseVoiceASR": "Qwen3-TTS SenseVoice ASR 🗣️",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
