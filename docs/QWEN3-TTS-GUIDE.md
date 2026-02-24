# Qwen3-TTS Integration Guide

Quick reference for using Qwen3-TTS in the VideoLingo dubbing pipeline.

## Configuration

In `workspace/config.yaml`:

```yaml
tts_method: 'qwen_tts'

qwen_tts:
  model: 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'
  speaker: 'Eric'  # Male voice
  instruct: '男声科技博主，专业沉稳，语速适中，标准普通话'  # Optimized for male

max_workers: 1  # IMPORTANT: Single thread for stability
```

## Voice Instructions by Speaker

| Speaker | Best Instruct | Speed Factor |
|---------|---------------|--------------|
| **Eric** (male) | `男声科技博主，专业沉稳，语速适中，标准普通话` | **1.09x** ⭐ |
| Serena (female) | `科技博主风格，专业热情，语速适中，标准普通话` | ~1.05x |

## Available Chinese Speakers

| Speaker | Voice Style | Stability | Speed Factor |
|---------|-------------|-----------|--------------|
| **Eric** | Lively Chengdu male, slightly husky | ✅ **DEFAULT** | 1.09x |
| **Serena** | Warm, gentle female | ✅ Alternative | 1.05x |
| Vivian | Bright, edgy female | ✅ Stable | ~1.5x |
| Uncle_Fu | Low, mellow male | ⚠️ Sometimes unstable | 1.51x |
| Dylan | Beijing male | ❌ Broken (70x bug) | N/A |

## Device Settings

- **NVIDIA GPU:** `device_map="cuda:0"`, `dtype=torch.bfloat16`
- **macOS:** `device_map="cpu"`, `dtype=torch.float32` (MPS has bugs)

## Key Files

- `src/core/all_tts_functions/qwen_tts.py` - Main TTS backend
- `src/core/all_tts_functions/qwen_tts_wrapper.py` - VideoLingo wrapper
- `docs/plans/2025-02-21-qwen3-tts-integration.md` - Full implementation details

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Tensor.item() cannot be called on meta tensors" | Force CPU mode |
| Audio 70x too long | Don't use Dylan speaker |
| Inconsistent lengths | Set max_workers=1 |
| Slow generation | Use CUDA GPU instead of CPU |

## Results

- **Before (Edge TTS):** Speed factor 3.329x (audio too long)
- **After (Qwen3-TTS Eric):** Speed factor 1.09x (audio fits naturally)
