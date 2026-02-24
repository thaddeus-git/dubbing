# Qwen3-TTS Integration Implementation Plan

> **Status: COMPLETED** (2025-02-21)

**Goal:** Replace Edge TTS with Qwen3-TTS and implement audio duration alignment to fix the time-compression problem.

**Result:** Speed factor improved from 3.329x → 1.05x (audio now fits naturally within target duration)

**Tech Stack:** Python, Qwen3-TTS, ffmpeg (atempo), pydub, transformers

---

## Implementation Summary

### Files Modified

1. **`src/core/all_tts_functions/qwen_tts.py`** - Complete rewrite with:
   - Correct Qwen3-TTS speakers (Serena, Eric, Dylan, etc.)
   - Voice control via `instruct` parameter
   - Audio duration alignment using ffmpeg atempo
   - CPU mode for stability on macOS (MPS had issues)

2. **`src/core/all_tts_functions/qwen_tts_wrapper.py`** - Updated wrapper for new backend

3. **`workspace/config.yaml`** - Configuration changes:
   ```yaml
   tts_method: 'qwen_tts'
   qwen_tts:
     model: 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'
     speaker: 'Eric'  # Male voice (Serena = female alternative)
     instruct: '科技博主风格，专业热情，语速适中，标准普通话，像在给朋友讲解技术'
   max_workers: 1  # Single thread for stability
   ```

4. **`src/core/step11_merge_full_audio.py`** - Added `import numpy as np` for eval()

---

## Qwen3-TTS Usage Guide

### Available Speakers

| Speaker | Description | Native Language | Status |
|---------|-------------|-----------------|--------|
| **Eric** | Lively Chengdu male voice, slightly husky | Chinese (Sichuan) | ✅ **DEFAULT** |
| **Serena** | Warm, gentle young female voice | Chinese | ✅ Recommended |
| Vivian | Bright, slightly edgy young female voice | Chinese | ✅ Stable |
| Uncle_Fu | Seasoned male voice, low mellow timbre | Chinese | ⚠️ Sometimes unstable |
| Dylan | Youthful Beijing male voice | Chinese (Beijing) | ❌ Broken (70x bug) |
| Ryan | Dynamic male voice | English | ✅ Stable |
| Aiden | Sunny American male voice | English | ✅ Stable |
| Ono_Anna | Playful Japanese female voice | Japanese | ✅ Stable |
| Sohee | Warm Korean female voice | Korean | ✅ Stable |

### Voice Instructions (instruct parameter)

The `instruct` parameter controls voice style via natural language:

```python
# Tech video style (recommended for dubbing)
TECH_VIDEO_INSTRUCT = '科技博主风格，专业热情，语速适中，标准普通话，像在给朋友讲解技术'

# Other examples:
# '语速稍快，简洁干练' - Faster, more concise (CAUTION: can cause instability)
# '用特别愤怒的语气说' - Angry tone
# '温和友善的语气' - Gentle, friendly tone
```

### Device Configuration

```python
# For NVIDIA GPU (CUDA)
device_map = "cuda:0"
dtype = torch.bfloat16

# For macOS - FORCE CPU (MPS has stability issues)
device_map = "cpu"
dtype = torch.float32
```

**Important:** On macOS, MPS causes "Tensor.item() cannot be called on meta tensors" error. Force CPU mode for stability.

### Code Example

```python
from qwen_tts import Qwen3TTSModel
import soundfile as sf

# Load model
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cpu",  # Force CPU on macOS
    torch_dtype=torch.float32,
)

# Generate audio
wavs, sr = model.generate_custom_voice(
    text="这是一个测试句子",
    language="Chinese",
    speaker="Eric",  # Male voice (Serena = female)
    instruct="科技博主风格，专业热情，语速适中，标准普通话",
)

# Save
sf.write("output.wav", wavs[0], sr)
```

---

## Known Issues

### 1. Dylan (Beijing) Speaker Instability
- Generates 70x longer audio than expected
- Root cause: Unknown, possibly model bug
- Workaround: Use Serena or Eric instead

### 2. MPS (Apple Silicon) Instability
- Error: "Tensor.item() cannot be called on meta tensors"
- Root cause: PyTorch MPS backend incompatibility
- Workaround: Force `device_map="cpu"`

### 3. Parallel Execution Issues
- Running with max_workers > 1 causes inconsistent audio lengths
- Workaround: Set `max_workers: 1` in config.yaml

### 4. Voice Instruction Sensitivity
- Complex instructions can cause 5-15x audio length variations
- Keep instructions simple and consistent
- Best results: '科技博主风格，专业热情，语速适中，标准普通话'

---

## Performance Results

| Metric | Edge TTS | Qwen3-TTS (CPU) |
|--------|----------|-----------------|
| Speed factor | 3.329x | 1.05x |
| Audio quality | Good | Better |
| Stability | High | High (CPU mode) |
| Generation time | ~5s/line | ~10s/line |

---

## Original Plan Tasks (Completed)

---

## Task 1: Install Qwen3-TTS Dependencies

**Files:**
- Modify: `pyproject.toml` or `requirements.txt`

**Step 1: Add qwen-tts dependency**

```toml
# In pyproject.toml or requirements.txt
qwen-tts>=0.1.0
```

**Step 2: Install flash-attn for faster inference (optional)**

```bash
pip install flash-attn --no-build-isolation
```

**Step 3: Verify installation**

Run: `python -c "from qwen_tts import Qwen3TTSModel; print('OK')"`
Expected: "OK"

---

## Task 2: Create Qwen3-TTS Wrapper Module

**Files:**
- Create: `src/core/tts_qwen3.py`

**Step 1: Create the TTS wrapper class**

```python
"""
Qwen3-TTS wrapper for video dubbing pipeline
Supports custom voice generation with duration control
"""
import os
import time
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

class Qwen3TTSWrapper:
    """Wrapper for Qwen3-TTS model with dubbing-optimized settings"""

    DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    DEFAULT_SPEAKER = "Eric"  # Chengdu male voice - most stable
    DEFAULT_LANGUAGE = "Chinese"

    # Speaker options for Chinese:
    # - Vivian: Bright young female
    # - Serena: Warm gentle female
    # - Uncle_Fu: Low mellow male (sometimes unstable)
    # - Dylan: Beijing male (currently unstable - 70x duration bug)
    # - Eric: Chengdu male (most stable, recommended)

    def __init__(
        self,
        model_name: str = None,
        speaker: str = None,
        device: str = "auto",
        dtype: torch.dtype = torch.float32,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.speaker = speaker or self.DEFAULT_SPEAKER
        self.device = device
        self.dtype = dtype
        self._model = None

    def _load_model(self):
        """Lazy load the model"""
        if self._model is None:
            logger.info(f"Loading Qwen3-TTS model: {self.model_name}")
            from qwen_tts import Qwen3TTSModel

            self._model = Qwen3TTSModel.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=self.dtype,
            )
            logger.info("Qwen3-TTS model loaded successfully")
        return self._model

    def generate(
        self,
        text: str,
        output_path: str,
        language: str = None,
        speaker: str = None,
        instruct: str = "",
    ) -> Tuple[float, float]:
        """
        Generate audio for a single text segment

        Args:
            text: Chinese text to synthesize
            output_path: Path to save the .wav file
            language: Language code (default: Chinese)
            speaker: Speaker name (default: self.speaker)
            instruct: Optional instruction for style control

        Returns:
            Tuple of (duration_seconds, generation_time_seconds)
        """
        model = self._load_model()

        start_time = time.time()

        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language or self.DEFAULT_LANGUAGE,
            speaker=speaker or self.speaker,
            instruct=instruct,
        )

        gen_time = time.time() - start_time

        # Save audio
        sf.write(output_path, wavs[0], sr)

        duration = len(wavs[0]) / sr

        logger.debug(f"Generated {duration:.2f}s audio in {gen_time:.1f}s -> {output_path}")

        return duration, gen_time

    def generate_batch(
        self,
        texts: List[str],
        output_paths: List[str],
        language: str = None,
        speaker: str = None,
    ) -> List[Tuple[float, float]]:
        """
        Generate audio for multiple text segments (batch mode)

        Returns:
            List of (duration, gen_time) tuples
        """
        model = self._load_model()

        start_time = time.time()

        wavs, sr = model.generate_custom_voice(
            text=texts,
            language=[language or self.DEFAULT_LANGUAGE] * len(texts),
            speaker=[speaker or self.speaker] * len(texts),
        )

        gen_time = time.time() - start_time

        results = []
        for i, (wav, path) in enumerate(zip(wavs, output_paths)):
            sf.write(path, wav, sr)
            duration = len(wav) / sr
            results.append((duration, gen_time / len(texts)))

        return results
```

**Step 2: Verify the wrapper works**

Run: `python -c "from src.core.tts_qwen3 import Qwen3TTSWrapper; w = Qwen3TTSWrapper(); print('OK')"`
Expected: "OK"

---

## Task 3: Implement Audio Duration Alignment

**Files:**
- Create: `src/core/audio_alignment.py`

**Step 1: Create alignment functions using ffmpeg atempo**

```python
"""
Audio duration alignment using ffmpeg
Stretches/compresses audio to match target duration while preserving pitch
"""
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple
import soundfile as sf

logger = logging.getLogger(__name__)

# ffmpeg atempo supports 0.5 to 2.0 ratio
# For ratios outside this range, chain multiple atempo filters
ATEMPO_MIN = 0.5
ATEMPO_MAX = 2.0

def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe"""
    cmd = [
        'ffprobe', '-i', audio_path,
        '-show_entries', 'format=duration',
        '-v', 'quiet', '-of', 'csv=p=0'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def calculate_atempo_filters(ratio: float) -> str:
    """
    Calculate atempo filter chain for any ratio
    ffmpeg atempo only supports 0.5-2.0, so we chain filters for extreme ratios
    """
    if ATEMPO_MIN <= ratio <= ATEMPO_MAX:
        return f"atempo={ratio}"

    # Chain multiple atempo filters
    filters = []
    remaining_ratio = ratio

    while remaining_ratio > ATEMPO_MAX:
        filters.append(f"atempo={ATEMPO_MAX}")
        remaining_ratio /= ATEMPO_MAX

    while remaining_ratio < ATEMPO_MIN:
        filters.append(f"atempo={ATEMPO_MIN}")
        remaining_ratio /= ATEMPO_MIN

    if remaining_ratio != 1.0:
        filters.append(f"atempo={remaining_ratio}")

    return ",".join(filters)

def align_audio_duration(
    input_path: str,
    output_path: str,
    target_duration: float,
    max_speed_ratio: float = 1.3,
) -> Tuple[bool, float, str]:
    """
    Align audio duration to target using time stretching

    Args:
        input_path: Path to input audio file
        output_path: Path to save aligned audio
        target_duration: Target duration in seconds
        max_speed_ratio: Maximum allowed speed ratio (default 1.3)
            If actual ratio > max, audio will be too fast (return warning)

    Returns:
        Tuple of (success, actual_ratio, warning_message)
    """
    current_duration = get_audio_duration(input_path)
    ratio = current_duration / target_duration

    if abs(ratio - 1.0) < 0.02:
        # Already aligned (within 2% tolerance)
        Path(input_path).rename(output_path) if input_path != output_path else None
        return True, ratio, ""

    warning = ""

    if ratio > max_speed_ratio:
        warning = f"⚠️ Audio speed ratio {ratio:.2f}x exceeds max {max_speed_ratio}x - audio will be fast"

    # Build ffmpeg command
    atempo_filter = calculate_atempo_filters(1.0 / ratio)  # Inverse: we want to speed up

    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-filter:a', atempo_filter,
        '-vn', output_path
    ]

    logger.debug(f"Aligning audio: {current_duration:.2f}s -> {target_duration:.2f}s (ratio: {ratio:.2f}x)")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"ffmpeg alignment failed: {result.stderr}")
        return False, ratio, f"ffmpeg error: {result.stderr[:200]}"

    new_duration = get_audio_duration(output_path)
    logger.info(f"Aligned audio: {current_duration:.2f}s -> {new_duration:.2f}s (target: {target_duration:.2f}s)")

    return True, ratio, warning

def pad_audio_to_duration(
    input_path: str,
    output_path: str,
    target_duration: float,
) -> bool:
    """
    Pad audio with silence if shorter than target duration

    Args:
        input_path: Path to input audio file
        output_path: Path to save padded audio
        target_duration: Target duration in seconds

    Returns:
        True if successful
    """
    current_duration = get_audio_duration(input_path)

    if current_duration >= target_duration:
        # No padding needed
        if input_path != output_path:
            Path(input_path).rename(output_path)
        return True

    pad_duration = target_duration - current_duration

    # Use ffmpeg to add silence at the end
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-filter_complex', f'[0:a]apad=pad_dur={pad_duration}[a]',
        '-map', '[a]',
        '-vn', output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"ffmpeg padding failed: {result.stderr}")
        return False

    return True
```

**Step 2: Test alignment functions**

```bash
# Create a test audio file and verify alignment works
python -c "
from src.core.audio_alignment import align_audio_duration, get_audio_duration
# Test with the generated audio
result = align_audio_duration(
    'workspace/qwen3_demo/qwen3_eric_line_0.wav',
    '/tmp/test_aligned.wav',
    5.945,  # target duration
)
print(f'Result: {result}')
print(f'New duration: {get_audio_duration(\"/tmp/test_aligned.wav\")}s')
"
```

---

## Task 4: Add Translation Length Constraint

**Files:**
- Modify: `src/core/step4_translation.py` (or equivalent)

**Step 1: Add length constraint to translation prompt**

Find the translation prompt in the codebase and add this constraint:

```python
# In the translation prompt, add:
LENGTH_CONSTRAINT = """
翻译长度约束：
- 翻译后的中文字符数不超过原英文单词数的1.5倍
- 优先使用简洁口语化表达
- 避免冗余修饰词
- 示例：原文"the number one question I have gotten" (6 words) -> "大家最常问的问题" (7 chars, 1.17x)
"""
```

**Step 2: Add post-translation validation**

```python
def validate_translation_length(original: str, translated: str, max_ratio: float = 1.5) -> tuple[bool, str]:
    """
    Validate that Chinese translation is not too long

    Returns:
        Tuple of (is_valid, warning_message)
    """
    word_count = len(original.split())
    char_count = len(translated.replace(" ", ""))

    ratio = char_count / word_count if word_count > 0 else 0

    if ratio > max_ratio:
        return False, f"Translation too long: {char_count} chars for {word_count} words (ratio: {ratio:.2f}x > {max_ratio}x)"

    return True, ""
```

---

## Task 5: Integrate Qwen3-TTS into Pipeline

**Files:**
- Modify: `src/core/step10_tts.py` (or equivalent TTS step)

**Step 1: Add Qwen3-TTS as TTS option**

```python
# In step10_tts.py or TTS module

def generate_tts_audio_qwen3(
    text: str,
    output_path: str,
    target_duration: float,
    speaker: str = "Eric",
    max_speed_ratio: float = 1.3,
) -> dict:
    """
    Generate TTS audio using Qwen3-TTS with duration alignment

    Returns:
        Dict with: success, original_duration, aligned_duration, speed_ratio, warning
    """
    from src.core.tts_qwen3 import Qwen3TTSWrapper
    from src.core.audio_alignment import align_audio_duration, pad_audio_to_duration

    tts = Qwen3TTSWrapper(speaker=speaker)

    # Generate raw audio
    temp_path = output_path.replace('.wav', '_raw.wav')
    original_duration, gen_time = tts.generate(text, temp_path)

    # Calculate ratio
    ratio = original_duration / target_duration

    if ratio > 1.0:
        # Audio too long - need to speed up
        success, actual_ratio, warning = align_audio_duration(
            temp_path, output_path, target_duration, max_speed_ratio
        )
    else:
        # Audio shorter than target - pad with silence
        success = pad_audio_to_duration(temp_path, output_path, target_duration)
        warning = ""
        actual_ratio = ratio

    # Clean up temp file
    Path(temp_path).unlink(missing_ok=True)

    return {
        "success": success,
        "original_duration": original_duration,
        "aligned_duration": target_duration,
        "speed_ratio": actual_ratio,
        "warning": warning,
        "gen_time": gen_time,
    }
```

**Step 2: Update config to support Qwen3-TTS**

```yaml
# In config.yaml
tts_method: 'qwen3_tts'  # Changed from 'edge_tts'

# Qwen3-TTS settings
qwen3_tts:
  model: 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'
  speaker: 'Eric'  # Options: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
  language: 'Chinese'
  max_speed_ratio: 1.3  # Maximum speed-up ratio before warning
```

---

## Task 6: Handle Edge Cases

**Files:**
- Modify: `src/core/step10_tts.py`

**Step 1: Add smart splitting when ratio > 1.3x**

```python
def should_split_sentence(original: str, translated: str, ratio: float) -> bool:
    """
    Determine if sentence should be split due to length mismatch
    """
    if ratio <= 1.3:
        return False

    # If translation is >1.3x, consider splitting
    word_count = len(original.split())

    # Only split if original is long enough to split meaningfully
    return word_count >= 6

def split_long_sentence(text: str, llm_client) -> list[str]:
    """
    Use LLM to split a long sentence into shorter segments
    """
    prompt = f"""将以下中文句子分成2-3个较短的片段，保持语义完整。
每个片段应该能够独立成句。

原文：{text}

请直接输出分割后的句子，每行一个："""

    response = llm_client.generate(prompt)
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return lines
```

**Step 2: Add overlap handling**

```python
def handle_timeline_overlap(segments: list[dict]) -> list[dict]:
    """
    Adjust timelines to prevent overlaps
    If segment end > next segment start, adjust boundaries
    """
    adjusted = []

    for i, seg in enumerate(segments):
        if i == 0:
            adjusted.append(seg)
            continue

        prev_end = adjusted[-1]['end_time']
        curr_start = seg['start_time']

        if curr_start < prev_end:
            # Overlap detected - adjust current start time
            logger.warning(f"Timeline overlap detected: segment {i} starts at {curr_start}s but previous ends at {prev_end}s")
            seg['start_time'] = prev_end + 0.05  # 50ms gap

        adjusted.append(seg)

    return adjusted
```

---

## Task 7: Update Pipeline Configuration

**Files:**
- Modify: `workspace/config.yaml`

**Step 1: Add Qwen3-TTS configuration**

```yaml
tts_method: 'qwen3_tts'

qwen3_tts:
  model: 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'
  speaker: 'Eric'
  language: 'Chinese'
  max_speed_ratio: 1.3
  instruct: ''  # Optional style instructions
```

**Step 2: Run pipeline to verify**

Run: `cd workspace && python ../run_pipeline.py`
Expected: Pipeline completes with Qwen3-TTS audio generation

---

## Task 8: Test and Verify

**Files:**
- Create: `tests/test_qwen3_tts.py`

**Step 1: Write integration test**

```python
def test_qwen3_tts_with_alignment():
    """Test Qwen3-TTS generation with duration alignment"""
    from src.core.tts_qwen3 import Qwen3TTSWrapper
    from src.core.audio_alignment import align_audio_duration, get_audio_duration

    tts = Qwen3TTSWrapper(speaker="Eric")

    # Generate test audio
    text = "这是一个测试句子"
    output = "/tmp/test_qwen3.wav"

    duration, _ = tts.generate(text, output)
    assert duration > 0
    assert Path(output).exists()

    # Test alignment
    target = 1.5  # Force speed up
    success, ratio, warning = align_audio_duration(output, "/tmp/test_aligned.wav", target)

    assert success
    new_duration = get_audio_duration("/tmp/test_aligned.wav")
    assert abs(new_duration - target) < 0.1  # Within 100ms
```

**Step 2: Run tests**

Run: `pytest tests/test_qwen3_tts.py -v`
Expected: All tests pass

---

## Execution Checklist

- [ ] Task 1: Install dependencies
- [ ] Task 2: Create Qwen3-TTS wrapper
- [ ] Task 3: Implement audio alignment
- [ ] Task 4: Add translation constraint
- [ ] Task 5: Integrate into pipeline
- [ ] Task 6: Handle edge cases
- [ ] Task 7: Update config
- [ ] Task 8: Test and verify

---

## Known Issues / TODO

1. **Dylan (Beijing) speaker instability**: Currently generates 70x longer audio than expected. Needs debugging with Qwen team or using different generation parameters.

2. **Flash-attn on macOS**: Not supported, using eager attention. Could be slower on GPU systems.

3. **Batch generation**: Qwen3-TTS supports batch mode but current implementation processes one segment at a time for easier error handling.
