# Video Dubbing (English → Chinese)

Agent Skills for dubbing English videos to Chinese using Qwen3 TTS with WhisperX transcription and MiniMax translation.

These skills follow the [Agent Skills specification](https://agentskills.io/specification) so they can be used by any skills-compatible agent, including Claude Code and Codex CLI.

## Installation

### Marketplace

```
/plugin marketplace add thaddeus-git/dubbing
/plugin install video-dubbing@dubbing
```

### Manually

```bash
git clone https://github.com/thaddeus-git/dubbing.git
cp -r dubbing ~/.claude/skills/video-dubbing
```

## Quick Start

```bash
# 1. Configure
cp config.yaml.example config.yaml
# Edit config.yaml with your API keys

# 2. Add video
cp your-video.mp4 workspace/output/demo.mkv

# 3. Run
cd workspace && python ../run_pipeline.py
```

## TTS Options

| Mode | Speed | Quality | Requirements |
|------|-------|---------|--------------|
| **Local (CPU)** | ~10s/line | Best | 8GB+ RAM, qwen-tts |
| **API (DashScope)** | ~2s/line | Good | API key, dashscope |

## Configuration

```yaml
# config.yaml
tts_method: 'qwen_tts'  # or 'qwen_tts_api'

# Local mode
qwen_tts:
  speaker: 'Serena'  # Female (Eric = male)
  instruct: '科技博主风格，专业热情，语速适中，标准普通话'

# API mode
qwen_tts_api:
  voice: 'Ethan'  # Male (Cherry/Serena = female)
  instruct: '专业科技视频主播风格，语速适中，普通话标准'

max_workers: 1  # IMPORTANT: Single thread for stability
```

## Voice Options

### Local (Qwen3-TTS-12Hz)

| Speaker | Gender | Speed Factor | Best For |
|---------|--------|--------------|----------|
| **Serena** | Female | 1.05x | Tech videos (recommended) |
| Eric | Male | 1.09x | Male voice |

### API (DashScope)

| Voice | Gender | Speed | Best For |
|-------|--------|-------|----------|
| **Ethan** | Male | 0.93x | Tech videos (recommended) |
| Cherry | Female | ~1.0x | Fashion, lifestyle |

## Output Files

```
workspace/output/
├── output_sub.mp4   # Subtitled video
├── output_dub.mp4   # Dubbed video
├── dub.srt          # Chinese subtitles
└── dub.mp3          # Chinese audio
```

## Environment Variables

```bash
# For API TTS
export DASHSCOPE_API_KEY=your-key
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Tensor.item() meta tensors" | Force CPU mode (macOS) |
| Speed factor > 2x | Check translation length |
| API auth error | Set DASHSCOPE_API_KEY |

## License

MIT
