from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.config import SupportedLanguage, SupportedModelSize, TranscriptionConfig
from app.services.transcription import TranscriptionError
from app.services.vibevoice_asr import VibeVoiceASRService


def test_vibevoice_formats_diarized_segments(tmp_path: Path) -> None:
    repo_path = tmp_path / "VibeVoice"
    repo_path.mkdir()
    config = TranscriptionConfig(
        supported_languages=[SupportedLanguage(code="de", label="German")],
        supported_model_sizes=[
            SupportedModelSize(
                id="vibevoice-7b",
                label="VibeVoice ASR 7B",
                model_name="microsoft/VibeVoice-ASR",
                backend="vibevoice_asr",
            )
        ],
        vibevoice_repo_path=repo_path,
    )
    backend = MagicMock()
    backend.transcribe.return_value = {
        "raw_text": "raw json",
        "segments": [
            {
                "speaker_id": 1,
                "start_time": 0.0,
                "end_time": 1.25,
                "text": "Hallo Welt",
            },
            {
                "speaker_id": "2",
                "start_time": "1.25",
                "end_time": "2.5",
                "text": "Guten Tag",
            },
        ],
    }

    with patch.object(VibeVoiceASRService, "_load_backend", return_value=backend):
        result = VibeVoiceASRService(config).transcribe(
            tmp_path / "audio.webm",
            "de",
            "vibevoice-7b",
        )

    assert result.transcript == "Speaker 1: Hallo Welt\n\nSpeaker 2: Guten Tag"
    assert result.duration_seconds == 2.5
    assert [segment.speaker for segment in result.segments] == ["1", "2"]
    assert backend.transcribe.call_args.kwargs == {
        "audio_path": str(tmp_path / "audio.webm"),
        "max_new_tokens": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }


def test_vibevoice_requires_repo_path(tmp_path: Path) -> None:
    config = TranscriptionConfig(
        supported_languages=[SupportedLanguage(code="de", label="German")],
        supported_model_sizes=[
            SupportedModelSize(
                id="vibevoice-7b",
                label="VibeVoice ASR 7B",
                model_name="microsoft/VibeVoice-ASR",
                backend="vibevoice_asr",
            )
        ],
    )

    with pytest.raises(TranscriptionError, match="vibevoice_repo_path"):
        VibeVoiceASRService(config).transcribe(
            tmp_path / "audio.webm", "de", "vibevoice-7b"
        )
