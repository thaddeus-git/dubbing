# Male Voice Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Find the best male Chinese voice and voice instruction to replace Eric with better timing/quality.

**Architecture:** Test all stable male speakers (Eric, Uncle_Fu) with various voice instructions optimized for male tech video style. Compare duration ratios and audio quality.

**Tech Stack:** Python, Qwen3-TTS, ffmpeg, soundfile

---

## Background

From previous testing:
- **Serena (female)** with tech instruct: 1.05x speed factor ✅
- **Eric (male)** with tech instruct: 1.21x speed factor ⚠️
- **Dylan (male, Beijing)**: BROKEN (70x audio bug) ❌

Goal: Find male voice with speed factor closer to 1.05x

---

## Task 1: Test Uncle_Fu (Mellow Male Voice)

**Files:**
- Create: `workspace/qwen3_demo/test_male_unclefu.py`

**Step 1: Create test script for Uncle_Fu**

```python
#!/usr/bin/env python3
"""Test Uncle_Fu male voice with tech instructions"""
import warnings
warnings.filterwarnings('ignore', message='flash-attn')

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

TEST_LINES = [
    {'id': 0, 'text': '这就是你们期待已久的视频 过去几周大家问得最多的一个问题就是', 'target': 5.945},
    {'id': 1, 'text': '自 Open Claw 发布以来 我知道这项技术很强大', 'target': 3.896},
    {'id': 2, 'text': '但在今天的视频里 我要说的是 它究竟如何让我们的生活变得更好', 'target': 3.213},
    {'id': 3, 'text': '今天我就来回答这个问题', 'target': 1.968},
]

# Male tech voice instructions to test
INSTRUCTS = {
    'tech_male': '科技博主风格，专业沉稳，语速适中，标准普通话，男声解说',
    'tech_simple': '科技解说风格，语速适中，标准普通话',
    'tech_deep': '低沉稳重的男声，科技解说风格，专业可靠',
    'empty': '',  # Baseline
}

print('Loading model...')
model = Qwen3TTSModel.from_pretrained(
    'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
    device_map='cpu',
    torch_dtype=torch.float32
)

for instruct_name, instruct_text in INSTRUCTS.items():
    print(f'\n=== Uncle_Fu + {instruct_name} ===')
    print(f'Instruct: {instruct_text[:40]}...' if instruct_text else 'Instruct: (empty)')

    total_ratio = 0
    for line in TEST_LINES:
        wavs, sr = model.generate_custom_voice(
            text=line['text'],
            language='Chinese',
            speaker='Uncle_Fu',
            instruct=instruct_text,
        )
        output = f"unclefu_{instruct_name}_{line['id']}.wav"
        sf.write(output, wavs[0], sr)
        duration = len(wavs[0]) / sr
        ratio = duration / line['target']
        total_ratio += ratio
        status = '✓' if ratio <= 1.3 else ('⚠️' if ratio <= 2.0 else '✗')
        print(f"  Line {line['id']}: {duration:.2f}s / {line['target']:.2f}s = {ratio:.2f}x {status}")

    avg_ratio = total_ratio / len(TEST_LINES)
    print(f"  Average ratio: {avg_ratio:.2f}x")

print('\nDone!')
```

**Step 2: Run the test**

Run: `cd workspace/qwen3_demo && python test_male_unclefu.py`
Expected: Duration ratios for each instruction variant

---

## Task 2: Test Eric with Optimized Male Instructions

**Files:**
- Create: `workspace/qwen3_demo/test_male_eric.py`

**Step 1: Create test script for Eric with male-optimized instructions**

```python
#!/usr/bin/env python3
"""Test Eric male voice with optimized male tech instructions"""
import warnings
warnings.filterwarnings('ignore', message='flash-attn')

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

TEST_LINES = [
    {'id': 0, 'text': '这就是你们期待已久的视频 过去几周大家问得最多的一个问题就是', 'target': 5.945},
    {'id': 1, 'text': '自 Open Claw 发布以来 我知道这项技术很强大', 'target': 3.896},
    {'id': 2, 'text': '但在今天的视频里 我要说的是 它究竟如何让我们的生活变得更好', 'target': 3.213},
    {'id': 3, 'text': '今天我就来回答这个问题', 'target': 1.968},
]

# Male-specific tech voice instructions
INSTRUCTS = {
    'current': '科技博主风格，专业热情，语速适中，标准普通话，像在给朋友讲解技术',
    'male_tech': '男声科技博主，专业沉稳，语速适中，标准普通话',
    'calm_male': '沉稳男声，科技解说风格，语速自然，标准普通话',
    'simple_male': '科技解说，男声，语速适中',
}

print('Loading model...')
model = Qwen3TTSModel.from_pretrained(
    'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
    device_map='cpu',
    torch_dtype=torch.float32
)

for instruct_name, instruct_text in INSTRUCTS.items():
    print(f'\n=== Eric + {instruct_name} ===')
    print(f'Instruct: {instruct_text[:40]}...')

    total_ratio = 0
    for line in TEST_LINES:
        wavs, sr = model.generate_custom_voice(
            text=line['text'],
            language='Chinese',
            speaker='Eric',
            instruct=instruct_text,
        )
        output = f"eric_{instruct_name}_{line['id']}.wav"
        sf.write(output, wavs[0], sr)
        duration = len(wavs[0]) / sr
        ratio = duration / line['target']
        total_ratio += ratio
        status = '✓' if ratio <= 1.3 else ('⚠️' if ratio <= 2.0 else '✗')
        print(f"  Line {line['id']}: {duration:.2f}s / {line['target']:.2f}s = {ratio:.2f}x {status}")

    avg_ratio = total_ratio / len(TEST_LINES)
    print(f"  Average ratio: {avg_ratio:.2f}x")

print('\nDone!')
```

**Step 2: Run the test**

Run: `cd workspace/qwen3_demo && python test_male_eric.py`
Expected: Duration ratios for each instruction variant

---

## Task 3: Compare Results and Select Best

**Step 1: Create comparison summary**

After running Task 1 and Task 2, compare results:

| Speaker | Instruct | Line 0 | Line 1 | Line 2 | Line 3 | Average |
|---------|----------|--------|--------|--------|--------|---------|
| Eric | current | ? | ? | ? | ? | 1.21x |
| Eric | male_tech | ? | ? | ? | ? | ? |
| Eric | calm_male | ? | ? | ? | ? | ? |
| Uncle_Fu | tech_male | ? | ? | ? | ? | ? |
| Uncle_Fu | tech_simple | ? | ? | ? | ? | ? |
| Serena | tech (baseline) | 1.49x | 1.19x | 1.87x | 1.71x | 1.57x |

**Step 2: Select winner**

Criteria:
1. Average ratio closest to 1.0x
2. No individual line > 2.0x
3. Voice quality (subjective, listen to samples)

---

## Task 4: Update Configuration

**Files:**
- Modify: `workspace/config.yaml`
- Modify: `src/core/all_tts_functions/qwen_tts.py`

**Step 1: Update config with winner**

```yaml
qwen_tts:
  model: 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'
  device: 'auto'
  speaker: '<WINNER_SPEAKER>'  # Based on test results
  instruct: '<WINNER_INSTRUCT>'  # Based on test results
```

**Step 2: Update default in code**

```python
DEFAULT_SPEAKER = "<WINNER_SPEAKER>"
DEFAULT_INSTRUCT = "<WINNER_INSTRUCT>"
```

**Step 3: Update documentation**

Update `docs/plans/2025-02-21-qwen3-tts-integration.md` with new recommended male voice.

---

## Task 5: Verify with Full Pipeline

**Step 1: Clean cache and run pipeline**

Run: `cd workspace && rm -rf output/audio/tmp output/audio/segs output/dub.mp3 output/dub.srt && python ../run_pipeline.py`

**Step 2: Verify speed factor**

Expected: Speed factor ≤ 1.2x (ideally ≤ 1.1x)

---

## Execution Checklist

- [ ] Task 1: Test Uncle_Fu with various instructions
- [ ] Task 2: Test Eric with male-optimized instructions
- [ ] Task 3: Compare results and select winner
- [ ] Task 4: Update configuration
- [ ] Task 5: Verify with full pipeline

---

## Notes

- Dylan (Beijing male) is excluded due to known 70x audio bug
- Run tests with `device_map='cpu'` for stability on macOS
- Listen to samples before making final decision
