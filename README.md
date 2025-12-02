# Vox - AI Audio Intelligence

End-to-end audio transcription and analysis pipeline using Whisper ASR and Llama 3.3 70B.

## Features

- **Speech-to-Text**: Faster-Whisper ASR with word-level timestamps
- **Speaker Identification**: LLM-powered intelligent speaker separation
- **Structured Output**: Summary, action items, key points, decisions
- **Web Interface**: Modern drag-and-drop UI with real-time progress
- **API**: RESTful endpoints for programmatic access

## Tech Stack

| Component | Technology |
|-----------|------------|
| ASR | faster-whisper (small model) |
| LLM | Groq API (Llama 3.3 70B Versatile) |
| Backend | FastAPI + Python 3.10+ |
| Frontend | Vanilla JS + CSS |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Edit `config.yaml` and set your Groq API key:

```yaml
llm:
  api_key: "your-groq-api-key"
```

Get a free API key at [console.groq.com](https://console.groq.com)

### 3. Run Server

```bash
python run_server.py
```

Open http://localhost:8000/app

## Usage

### Web Interface

1. Open http://localhost:8000/app
2. Drag & drop audio file (WAV, MP3, OGG, FLAC)
3. Click "Analyze Audio"
4. View results with speaker separation

### CLI

```bash
python scripts/run_batch.py audio.wav -o output.json
```

### API

```bash
# Upload and process
curl -X POST -F "file=@audio.wav" http://localhost:8000/api/upload

# Check status
curl http://localhost:8000/api/status/{job_id}

# Get result
curl http://localhost:8000/api/result/{job_id}
```

## Output Format

```json
{
  "summary": "Brief summary of the conversation",
  "action_items": [{"item": "...", "confidence": 0.95}],
  "decisions": [{"decision": "...", "confidence": 0.90}],
  "key_points": [{"point": "...", "confidence": 0.85}],
  "transcript": {
    "segments": [
      {"text": "...", "speaker": "SPEAKER_01", "start": 0.0, "end": 5.0}
    ]
  },
  "speaker_profiles": {
    "SPEAKER_01": {"likely_role": "Sales Person"},
    "SPEAKER_02": {"likely_role": "Customer"}
  }
}
```

## Configuration

Key settings in `config.yaml`:

```yaml
asr:
  model: "small"
  device: "cpu"
  language: "en"

llm:
  backend: "groq"
  model: "llama-3.3-70b-versatile"
  api_key: "your-api-key"
```

## Requirements

- Python 3.10+
- ffmpeg (for audio conversion)
- Groq API key (free tier available)
- ~2GB disk space for Whisper model

## License

MIT
