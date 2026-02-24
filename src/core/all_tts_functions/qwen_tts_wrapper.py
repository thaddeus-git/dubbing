"""
Qwen3-TTS wrapper for VideoLingo
Supports both local model and DashScope API
"""

import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.config_utils import load_key

# Lazy import of backends
_QWEN_BACKEND = None
_QWEN_API_BACKEND = None

def _get_config_value(key, default=''):
    """Get config value with default fallback"""
    try:
        return load_key(key)
    except KeyError:
        return default

def _get_backend():
    """Get or create Qwen3-TTS local backend instance"""
    global _QWEN_BACKEND
    if _QWEN_BACKEND is None:
        from core.all_tts_functions.qwen_tts import QwenTTSBackend

        config = {
            'model': _get_config_value('qwen_tts.model', 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'),
            'speaker': _get_config_value('qwen_tts.speaker', 'Serena'),
            'instruct': _get_config_value('qwen_tts.instruct', '科技博主风格，专业热情，语速适中，标准普通话'),
        }
        _QWEN_BACKEND = QwenTTSBackend(config)
    return _QWEN_BACKEND

def _get_api_backend():
    """Get or create Qwen3-TTS API backend instance"""
    global _QWEN_API_BACKEND
    if _QWEN_API_BACKEND is None:
        from core.all_tts_functions.qwen_tts_api import QwenTTSAPBackend

        config = {
            'model': _get_config_value('qwen_tts_api.model', 'qwen3-tts-instruct-flash'),
            'voice': _get_config_value('qwen_tts_api.voice', 'Cherry'),
            'instruct': _get_config_value('qwen_tts_api.instruct', 'Tech video narration style, moderate pace'),
            'api_key': _get_config_value('qwen_tts_api.api_key', None),
        }
        _QWEN_API_BACKEND = QwenTTSAPBackend(config)
    return _QWEN_API_BACKEND

def qwen_tts(text, save_path, target_duration=None):
    """
    Qwen3-TTS function for VideoLingo (local model)

    Args:
        text: Text to synthesize (Chinese)
        save_path: Path to save audio file
        target_duration: Optional target duration for alignment
    """
    backend = _get_backend()

    try:
        result_path, orig_dur, aligned_dur = backend.synthesize(
            text=text,
            output_path=Path(save_path),
            target_duration=target_duration,
        )

        ratio = orig_dur / target_duration if target_duration else 1.0
        if target_duration:
            print(f"[Qwen3TTS] {save_path}: {orig_dur:.2f}s -> {aligned_dur:.2f}s (ratio: {ratio:.2f}x)")
        else:
            print(f"[Qwen3TTS] {save_path}: {orig_dur:.2f}s")

    except Exception as e:
        print(f"[Qwen3TTS] Error: {e}")
        # Create silent audio on error
        from pydub import AudioSegment
        silence = AudioSegment.silent(duration=100)
        silence.export(save_path, format="wav")
        raise

def qwen_tts_api(text, save_path, target_duration=None):
    """
    Qwen3-TTS API function using DashScope (cloud)

    Args:
        text: Text to synthesize (Chinese)
        save_path: Path to save audio file
        target_duration: Optional target duration for alignment
    """
    backend = _get_api_backend()

    try:
        result_path, orig_dur, aligned_dur = backend.synthesize(
            text=text,
            output_path=Path(save_path),
            target_duration=target_duration,
        )

        ratio = orig_dur / target_duration if target_duration else 1.0
        if target_duration:
            print(f"[Qwen3TTS-API] {save_path}: {orig_dur:.2f}s -> {aligned_dur:.2f}s (ratio: {ratio:.2f}x)")
        else:
            print(f"[Qwen3TTS-API] {save_path}: {orig_dur:.2f}s")

    except Exception as e:
        print(f"[Qwen3TTS-API] Error: {e}")
        # Create silent audio on error
        from pydub import AudioSegment
        silence = AudioSegment.silent(duration=100)
        silence.export(save_path, format="wav")
        raise

if __name__ == "__main__":
    # Test local
    print("Testing local TTS...")
    qwen_tts("这是一个测试句子", "test_qwen3.wav", target_duration=2.0)

    # Test API (requires DASHSCOPE_API_KEY)
    print("\nTesting API TTS...")
    try:
        qwen_tts_api("这是一个测试句子", "test_qwen3_api.wav", target_duration=2.0)
    except Exception as e:
        print(f"API test failed (expected if no API key): {e}")
