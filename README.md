# Transcriber

[![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)](https://github.com/rnckp/transcriber)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Frontend](https://img.shields.io/badge/frontend-HTML5-E34F26?logo=html5&logoColor=white)](https://developer.mozilla.org/docs/Web/HTML)
[![Platform](https://img.shields.io/badge/macOS-Apple%20Silicon-000000?logo=apple&logoColor=white)](https://www.apple.com/mac/)
![GitHub License](https://img.shields.io/github/license/rnckp/transcriber)
[![GitHub Stars](https://img.shields.io/github/stars/rnckp/transcriber.svg)](https://github.com/rnckp/transcriber/stargazers)
<a href="https://github.com/astral-sh/ruff"><img alt="linting - Ruff" class="off-glb" loading="lazy" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>

Local browser-based transcription app for Apple Silicon Macs. It serves a FastAPI backend and vanilla HTML/CSS/JS frontend, then transcribes audio locally with `mlx-whisper` or the optional Microsoft VibeVoice ASR backend for diarized output.

![Transcriber UI](imgs/app_ui.png)

## Requirements

- macOS on Apple Silicon
- Python 3.12+
- `uv`
- For `vibevoice-7b`: enough memory for the 7B ASR model; browser `.webm` recordings also need the `ffmpeg` command available on `PATH` for audio decoding

## Setup

```bash
uv sync
```

The VibeVoice ASR source needed by this app is vendored in `vibevoice/` under the upstream MIT license, so no second VibeVoice checkout is required.

## Run

```bash
uv run python -m app.main
```

Open `http://127.0.0.1:8000`.

Choose a language and model, then either record from the browser microphone or choose an audio file. The UI shows upload and transcription progress, result metadata, and lets you copy or save the transcript as `.txt`.

## Models

Configured model options live in `config.yaml`:

- `tiny`, `base`, `small`, `medium`, `large`: `mlx-whisper` models cached under `transcription.cache_dir`
- `vibevoice-7b`: VibeVoice ASR with speaker segments, loaded through the local vendored VibeVoice package

Models download on first use and are reused from their cache afterward. VibeVoice also downloads its Hugging Face model on first real use and can require substantial memory; keep `transcription.vibevoice_max_new_tokens` conservative unless you know the target machine can handle more.

## API

`POST /api/transcriptions`

Multipart form fields:

- `audio`: audio file or browser recording
- `language`: configured language code, currently `de` or `en`
- `model_size`: configured model id, for example `small`, `large`, or `vibevoice-7b`

Successful response:

```json
{
  "transcript": "Text...",
  "language": "de",
  "model_size": "small",
  "duration_seconds": 12.3,
  "segments": []
}
```

Error response:

```json
{
  "error": {
    "code": "unsupported_language",
    "message": "Choose German or English."
  }
}
```

Invalid requests return `400`, oversized uploads return `413`, and transcription failures return `500`.

## Configuration

Edit `config.yaml` for runtime settings:

- `transcription.cache_dir`: local model cache directory
- `transcription.max_upload_size_mb`: upload size limit
- `transcription.upload_chunk_size_mb`: temp-file streaming chunk size
- `transcription.supported_languages`: language picker and backend validation
- `transcription.supported_model_sizes`: model picker, repo ids, and backend routing
- `transcription.vibevoice_*`: VibeVoice device, dtype, attention mode, and generation settings
- `logging.level`: structured JSON log level
- `ui.*`: browser labels, status messages, and progress timing heuristics

`server.host` and `server.port` control the bind address used by `uv run python -m app.main`.

## Development

```bash
uv run ruff format .
uv run ruff check .
uv run pytest -v
```

Project layout:

- `app/main.py`: FastAPI app, config loading, frontend template rendering
- `app/api/`: transcription API
- `app/services/`: backend routing, `mlx-whisper`, and VibeVoice integration
- `app/static/`: browser UI
- `tests/`: API, config, and service tests
- `vibevoice/`: vendored VibeVoice ASR model and processor code from Microsoft VibeVoice, MIT licensed

## License

MIT License. Vendored VibeVoice code under `vibevoice/` is also MIT licensed; see `vibevoice/LICENSE`.
