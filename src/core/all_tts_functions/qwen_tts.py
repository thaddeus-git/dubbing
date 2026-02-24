"""
Qwen3-TTS Backend for VideoLingo

Supports Qwen3-TTS-12Hz-1.7B-CustomVoice with:
- Custom Voice mode: Preset speakers (Vivian, Serena, Eric, Dylan, etc.)
- Voice control via natural language instructions
- Audio duration alignment using ffmpeg atempo
"""

import os
import re
import subprocess
import warnings
from pathlib import Path
from typing import Optional, Union, Tuple, List

import numpy as np  # Keep for compatibility, though not used directly
import soundfile as sf
import torch

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="flash-attn")

# Try to import Qwen TTS model
try:
    from qwen_tts import Qwen3TTSModel
    QWEN_TTS_AVAILABLE = True
except ImportError:
    QWEN_TTS_AVAILABLE = False
    warnings.warn("qwen_tts module not found. Install with: pip install qwen-tts")


# Actual Qwen3-TTS-12Hz speakers (from official docs)
QWEN3_SPEAKERS = {
    # Chinese speakers
    'Vivian': {'desc': 'Bright, slightly edgy young female voice', 'native': 'Chinese'},
    'Serena': {'desc': 'Warm, gentle young female voice', 'native': 'Chinese'},
    'Uncle_Fu': {'desc': 'Seasoned male voice with low, mellow timbre', 'native': 'Chinese'},
    'Dylan': {'desc': 'Youthful Beijing male voice, clear natural timbre', 'native': 'Chinese (Beijing)'},
    'Eric': {'desc': 'Lively Chengdu male voice, slightly husky brightness', 'native': 'Chinese (Sichuan)'},
    # English speakers
    'Ryan': {'desc': 'Dynamic male voice with strong rhythmic drive', 'native': 'English'},
    'Aiden': {'desc': 'Sunny American male voice with clear midrange', 'native': 'English'},
    # Other languages
    'Ono_Anna': {'desc': 'Playful Japanese female voice', 'native': 'Japanese'},
    'Sohee': {'desc': 'Warm Korean female voice with rich emotion', 'native': 'Korean'},
}

# Male tech voice instruction (optimized for male speaker)
TECH_VIDEO_INSTRUCT = '男声科技博主，专业沉稳，语速适中，标准普通话'


class QwenTTSBackend:
    """
    Qwen3-TTS Backend with audio duration alignment.
    """

    DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    DEFAULT_SPEAKER = "Eric"  # Lively Chengdu male voice (Serena = female)
    DEFAULT_INSTRUCT = TECH_VIDEO_INSTRUCT

    def __init__(self, config: Optional[dict] = None):
        if not QWEN_TTS_AVAILABLE:
            raise RuntimeError("qwen_tts not installed. Run: pip install qwen-tts")

        self.config = config or {}
        self.model_name = self.config.get('model', self.DEFAULT_MODEL)
        self.speaker = self.config.get('speaker', self.DEFAULT_SPEAKER)
        self.instruct = self.config.get('instruct', self.DEFAULT_INSTRUCT)
        self.device = self.config.get('device', 'auto')

        self._model = None

    def _load_model(self):
        """Lazy load the model"""
        if self._model is None:
            print(f"[Qwen3TTS] Loading model: {self.model_name}")

            # Detect best device - force CPU for stability on macOS
            if torch.cuda.is_available():
                device_map = "cuda:0"
                dtype = torch.bfloat16
            else:
                # Force CPU for stability (MPS has issues with Qwen3-TTS)
                device_map = "cpu"
                dtype = torch.float32

            self._model = Qwen3TTSModel.from_pretrained(
                self.model_name,
                device_map=device_map,
                torch_dtype=dtype,
            )
            print(f"[Qwen3TTS] Model loaded (device: {device_map})")

        return self._model

    def synthesize(
        self,
        text: str,
        output_path: Union[str, Path],
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
        target_duration: Optional[float] = None,
        max_speed_ratio: float = 1.5,
    ) -> Tuple[str, float, float]:
        """
        Synthesize speech with optional duration alignment.

        Args:
            text: Text to synthesize
            output_path: Path to save audio
            speaker: Speaker name (default: Serena)
            instruct: Voice instruction (default: tech video style)
            target_duration: If set, align audio to this duration
            max_speed_ratio: Maximum speed-up ratio (warn if exceeded)

        Returns:
            Tuple of (output_path, original_duration, aligned_duration)
        """
        model = self._load_model()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        speaker = speaker or self.speaker
        instruct = instruct if instruct is not None else self.instruct

        # Generate audio
        wavs, sr = model.generate_custom_voice(
            text=text,
            language='Chinese',
            speaker=speaker,
            instruct=instruct,
        )

        # Save raw audio first
        raw_path = output_path.with_suffix('.raw.wav')
        sf.write(raw_path, wavs[0], sr)
        original_duration = len(wavs[0]) / sr

        # Align duration if target specified
        if target_duration and target_duration > 0:
            aligned_duration, warning = align_audio_duration(
                raw_path, output_path, target_duration, max_speed_ratio
            )
            if warning:
                print(f"[Qwen3TTS] ⚠️ {warning}")
            raw_path.unlink(missing_ok=True)
        else:
            output_path.write_bytes(raw_path.read_bytes())
            raw_path.unlink(missing_ok=True)
            aligned_duration = original_duration

        return str(output_path), original_duration, aligned_duration


def align_audio_duration(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_duration: float,
    max_speed_ratio: float = 1.5,
) -> Tuple[float, Optional[str]]:
    """
    Align audio duration using ffmpeg atempo filter.

    Args:
        input_path: Input audio file
        output_path: Output audio file
        target_duration: Target duration in seconds
        max_speed_ratio: Warn if speed ratio exceeds this

    Returns:
        Tuple of (actual_duration, warning_message)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Get current duration
    current_duration = get_audio_duration(input_path)
    ratio = current_duration / target_duration

    warning = None

    if ratio > max_speed_ratio:
        warning = f"Speed ratio {ratio:.2f}x exceeds max {max_speed_ratio}x"

    if abs(ratio - 1.0) < 0.02:
        # Already aligned (within 2%)
        if input_path != output_path:
            output_path.write_bytes(input_path.read_bytes())
        return current_duration, warning

    # Calculate atempo filter chain
    speed_factor = 1.0 / ratio  # Inverse for speed-up
    atempo_filter = build_atempo_filter(speed_factor)

    # Run ffmpeg
    cmd = [
        'ffmpeg', '-y', '-i', str(input_path),
        '-filter:a', atempo_filter,
        '-vn', str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg alignment failed: {result.stderr[:200]}")

    new_duration = get_audio_duration(output_path)
    return new_duration, warning


def build_atempo_filter(speed: float) -> str:
    """
    Build ffmpeg atempo filter chain for any speed ratio.
    ffmpeg atempo only supports 0.5 to 2.0, so we chain filters.
    """
    ATEMPO_MIN = 0.5
    ATEMPO_MAX = 2.0

    if ATEMPO_MIN <= speed <= ATEMPO_MAX:
        return f"atempo={speed:.4f}"

    filters = []
    remaining = speed

    while remaining > ATEMPO_MAX:
        filters.append(f"atempo={ATEMPO_MAX}")
        remaining /= ATEMPO_MAX

    while remaining < ATEMPO_MIN:
        filters.append(f"atempo={ATEMPO_MIN}")
        remaining /= ATEMPO_MIN

    if remaining != 1.0:
        filters.append(f"atempo={remaining:.4f}")

    return ",".join(filters)


def get_audio_duration(audio_path: Union[str, Path]) -> float:
    """Get audio duration using ffprobe"""
    cmd = [
        'ffprobe', '-i', str(audio_path),
        '-show_entries', 'format=duration',
        '-v', 'quiet', '-of', 'csv=p=0'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


if __name__ == "__main__":
    # Quick test
    print("Qwen3-TTS Backend Test")
    print(f"Available speakers: {list(QWEN3_SPEAKERS.keys())}")

    if QWEN_TTS_AVAILABLE:
        backend = QwenTTSBackend({'speaker': 'Serena'})
        path, orig, aligned = backend.synthesize(
            "这是一个测试句子",
            "/tmp/qwen3_test.wav",
            target_duration=2.0
        )
        print(f"Generated: {path} (orig: {orig:.2f}s, aligned: {aligned:.2f}s)")
    else:
        print("qwen-tts not installed")
