"""
Qwen3-TTS API Backend using DashScope
Cloud-based TTS as alternative to local model
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Union, Tuple
import subprocess
import base64

import soundfile as sf

warnings.filterwarnings("ignore")

# Try to import dashscope
try:
    import dashscope
    from dashscope import MultiModalConversation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    warnings.warn("dashscope not installed. Run: pip install dashscope")


# DashScope API voices (from official docs)
DASHSCOPE_VOICES = {
    # Chinese voices
    'Cherry': {'desc': 'Sweet female voice, suitable for fashion/lifestyle', 'gender': 'female'},
    'Serena': {'desc': 'Warm gentle female voice', 'gender': 'female'},
    'Ethan': {'desc': 'Mature male voice, professional', 'gender': 'male'},
}

# Default instructions for tech video dubbing
TECH_VIDEO_INSTRUCT_API = '专业科技视频主播风格，语速适中，普通话标准'


class QwenTTSAPBackend:
    """
    Qwen3-TTS API Backend using DashScope cloud service.
    Faster than local model, no GPU required.
    """

    DEFAULT_MODEL = "qwen3-tts-instruct-flash"
    DEFAULT_VOICE = "Ethan"  # Male voice for tech content
    DEFAULT_INSTRUCT = TECH_VIDEO_INSTRUCT_API

    def __init__(self, config: Optional[dict] = None):
        if not DASHSCOPE_AVAILABLE:
            raise RuntimeError("dashscope not installed. Run: pip install dashscope")

        self.config = config or {}
        self.api_key = self.config.get('api_key') or os.getenv('DASHSCOPE_API_KEY')

        if not self.api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY not set. "
                "Set environment variable or pass api_key in config."
            )

        self.model = self.config.get('model', self.DEFAULT_MODEL)
        self.voice = self.config.get('voice', self.DEFAULT_VOICE)
        self.instruct = self.config.get('instruct', self.DEFAULT_INSTRUCT)

        # Configure DashScope
        dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

    def synthesize(
        self,
        text: str,
        output_path: Union[str, Path],
        voice: Optional[str] = None,
        instruct: Optional[str] = None,
        target_duration: Optional[float] = None,
        max_speed_ratio: float = 1.5,
    ) -> Tuple[str, float, float]:
        """
        Synthesize speech using DashScope API.

        Args:
            text: Text to synthesize
            output_path: Path to save audio
            voice: Voice name (default: Ethan for male, Cherry for female)
            instruct: Style instructions
            target_duration: If set, align audio to this duration
            max_speed_ratio: Maximum speed-up ratio

        Returns:
            Tuple of (output_path, original_duration, aligned_duration)
        """
        import requests

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        voice = voice or self.voice
        instruct = instruct if instruct is not None else self.instruct

        # Call DashScope API
        response = MultiModalConversation.call(
            model=self.model,
            api_key=self.api_key,
            text=text,
            voice=voice,
            instructions=instruct,
            optimize_instructions=True,
            stream=False
        )

        # Check response
        if response.status_code != 200:
            raise RuntimeError(f"DashScope API error: {response.code} - {response.message}")

        # Get audio from URL (DashScope returns URL, not inline data)
        audio_info = response.output.audio
        audio_url = audio_info.get('url')

        if not audio_url:
            raise RuntimeError("No audio URL in DashScope response")

        # Download audio
        audio_response = requests.get(audio_url)
        if audio_response.status_code != 200:
            raise RuntimeError(f"Failed to download audio: {audio_response.status_code}")

        # Save raw audio
        raw_path = output_path.with_suffix('.raw.wav')
        raw_path.write_bytes(audio_response.content)

        # Get duration
        original_duration = self._get_audio_duration(raw_path)

        # Align duration if target specified
        if target_duration and target_duration > 0:
            from .qwen_tts import align_audio_duration
            aligned_duration, warning = align_audio_duration(
                raw_path, output_path, target_duration, max_speed_ratio
            )
            if warning:
                print(f"[Qwen3TTS-API] ⚠️ {warning}")
            raw_path.unlink(missing_ok=True)
        else:
            output_path.write_bytes(raw_path.read_bytes())
            raw_path.unlink(missing_ok=True)
            aligned_duration = original_duration

        return str(output_path), original_duration, aligned_duration

    def _get_audio_duration(self, audio_path: Union[str, Path]) -> float:
        """Get audio duration using ffprobe"""
        cmd = [
            'ffprobe', '-i', str(audio_path),
            '-show_entries', 'format=duration',
            '-v', 'quiet', '-of', 'csv=p=0'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
