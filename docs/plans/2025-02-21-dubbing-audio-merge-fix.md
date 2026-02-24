# VideoLingo Audio Merge Fix - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the audio merge step in VideoLingo so the Chinese TTS audio properly replaces the original English audio in the final dubbed video.

**Architecture:** Create a clean workspace with only necessary VideoLingo source files. The issue is in step 12 (merge_dub_to_vid) where dubbed audio isn't properly mixed. We'll examine the merge logic, fix the audio replacement, and verify the complete pipeline works end-to-end.

**Tech Stack:** Python, FFmpeg, OpenAI SDK for MiniMax API, WhisperX, Qwen-TTS

---

## Current State Analysis

**Problem:**
- TTS works: `output/dub.mp3` contains perfect Chinese audio
- Transcription works: `output/output_sub.mp4` has correct Chinese subtitles
- BUT: `output/output_dub.mp4` still has original English audio (not replaced)

**Root Cause:**
Step 12 (`step12_merge_dub_to_vid.py`) isn't properly replacing the original audio track with the dubbed Chinese audio from `dub.mp3`.

---

## Task 1: Create Clean Workspace Structure

**Files:**
- Create: `/Users/thaddeus/projects/dubbing/workspace/` - working directory
- Create: `/Users/thaddeus/projects/dubbing/src/` - VideoLingo source files
- Create: `/Users/thaddeus/projects/dubbing/config.yaml` - configuration

**Step 1: Create workspace directories**

Run:
```bash
mkdir -p /Users/thaddeus/projects/dubbing/{workspace,src,docs}
```

**Step 2: Copy essential VideoLingo source files**

Run:
```bash
cd /Users/thaddeus/git/VideoLingo
cp -r core /Users/thaddeus/projects/dubbing/src/
cp config.yaml /Users/thaddeus/projects/dubbing/config.yaml.example
cp run_pipeline.py /Users/thaddeus/projects/dubbing/
```

**Step 3: Verify structure**

Run:
```bash
ls -la /Users/thaddeus/projects/dubbing/
ls /Users/thaddeus/projects/dubbing/src/core/
```

Expected:
```
src/  workspace/  config.yaml.example  run_pipeline.py
ask_gpt.py  step2_whisperX.py  step3_*.py ... step12_merge_dub_to_vid.py
```

---

## Task 2: Fix Configuration for Chinese TTS

**Files:**
- Create: `/Users/thaddeus/projects/dubbing/config.yaml` - working config
- Modify: Review TTS settings in config

**Step 1: Create working config.yaml**

Create file `/Users/thaddeus/projects/dubbing/config.yaml`:

```yaml
version: "2.1.2"

api:
  key: 'sk-api-K0XRy0Szb3YElJHwFHqMMb9sE2Q8TPT8bBI-GMKvYvMymkjFjRzCFaxittwrCOSMJZZCZxaEd8Uv2Cd238yyh73HUAjbZXeIzBy6jKbbJKw2MSPHR_CMSBs'
  base_url: 'https://api.minimaxi.com/v1'
  model: 'MiniMax-M2.5-highspeed'

target_language: 'chinese'
demucs: false

whisper:
  model: 'medium'
  language: 'en'
  detected_language: 'en'

resolution: '640x360'

subtitle:
  max_length: 75
  target_multiplier: 1.2

summary_length: 8000
max_workers: 4
max_split_length: 20

reflect_translate: true
pause_before_translate: false

tts_method: 'qwen_tts'

qwen_tts:
  model: 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'
  device: 'auto'
  speaker: 'Ryan'
  language: 'English'

speed_factor:
  min: 1
  accept: 1.2
  max: 1.4

min_subtitle_duration: 2.5
min_trim_duration: 3.5
tolerance: 1.5

dub_volume: 1.5

model_dir: './_model_cache'

allowed_video_formats:
- 'mp4'
- 'mov'
- 'avi'
- 'mkv'
- 'flv'
- 'wmv'
- 'webm'

llm_support_json:
- 'MiniMax-M2.5'
- 'MiniMax-M2.5-highspeed'
```

---

## Task 3: Fix Audio Merge Logic in Step 12

**Files:**
- Modify: `/Users/thaddeus/projects/dubbing/src/core/step12_merge_dub_to_vid.py`

**Step 1: Read current merge logic**

Read `/Users/thaddeus/projects/dubbing/src/core/step12_merge_dub_to_vid.py` to understand how audio is currently merged.

**Step 2: Identify the bug**

The issue: `dub.mp3` (Chinese audio) exists and is correct, but the final video still has original English audio.

This means the FFmpeg command is either:
- Not including the dubbed audio at all
- Including it but at 0% volume
- Layering it UNDER the original audio instead of REPLACING it

**Step 3: Fix the FFmpeg merge command**

The correct FFmpeg command should:
1. Take the original video: `-i output/demo.mkv`
2. Take the dubbed audio: `-i output/dub.mp3`
3. Map video from original: `-map 0:v`
4. Map audio from dubbed: `-map 1:a`
5. Copy video codec: `-c:v copy`
6. Encode audio: `-c:a aac -b:a 192k`

Replace the problematic merge logic with:

```python
import subprocess
from pathlib import Path

def merge_dub_to_vid(video_file, dub_audio, output_file):
    """Merge dubbed audio with video, replacing original audio."""
    cmd = [
        'ffmpeg', '-y',
        '-i', video_file,      # Original video (input 0)
        '-i', dub_audio,       # Dubbed audio (input 1)
        '-map', '0:v',         # Use video from input 0
        '-map', '1:a',         # Use audio from input 1
        '-c:v', 'copy',        # Copy video codec
        '-c:a', 'aac',         # Encode audio as AAC
        '-b:a', '192k',        # Audio bitrate
        output_file
    ]
    subprocess.run(cmd, check=True)
    return output_file
```

**Step 4: Update the main step12 function**

Replace the existing `merge_dub_to_vid()` function with the corrected version above.

---

## Task 4: Update Pipeline Runner

**Files:**
- Modify: `/Users/thaddeus/projects/dubbing/run_pipeline.py`

**Step 1: Update paths to use workspace**

Modify `run_pipeline.py` to:
- Use `/Users/thaddeus/projects/dubbing/workspace/` as the working directory
- Copy input video to `workspace/output/` before processing
- Set `os.chdir()` to workspace before running pipeline

**Step 2: Add clean workspace check**

Before running, clean the workspace:
```python
import shutil
from pathlib import Path

workspace = Path('/Users/thaddeus/projects/dubbing/workspace')
output_dir = workspace / 'output'

# Clean previous run
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
```

---

## Task 5: Test End-to-End Pipeline

**Files:**
- Test: `/Users/thaddeus/projects/dubbing/workspace/output/output_dub.mp4`

**Step 1: Run pipeline with test video**

```bash
cd /Users/thaddeus/projects/dubbing
cp /Users/thaddeus/Downloads/test_15s.mp4 workspace/output/demo.mkv
python run_pipeline.py
```

**Step 2: Verify outputs**

Check all outputs exist:
- `workspace/output/dub.mp3` - Should have Chinese audio
- `workspace/output/output_sub.mp4` - Should have Chinese subtitles
- `workspace/output/output_dub.mp4` - Should have Chinese audio replacing English

**Step 3: Test the dubbed video**

Play `output_dub.mp4` and verify:
- [ ] Video plays correctly
- [ ] Audio is in Chinese (not English)
- [ ] Audio syncs with video timing
- [ ] No original English audio bleeding through

---

## Summary

This plan will:
1. Create a clean workspace at `/Users/thaddeus/projects/dubbing/`
2. Fix the audio merge logic in step 12 to properly replace original audio with dubbed Chinese audio
3. Update configuration for Chinese TTS (Qwen3)
4. Test end-to-end to verify the fix works

**Expected Result:** The final `output_dub.mp4` will have Chinese audio instead of the original English audio.