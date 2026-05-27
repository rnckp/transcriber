from pathlib import Path

import pytest

from app.config import SupportedLanguage, SupportedModelSize, TranscriptionConfig
from app.services.transcription import (
    TranscriptionResult,
    UnsupportedModelSizeError,
)
from app.services.transcription_router import TranscriptionRouterService


class RecordingBackend:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.calls: list[tuple[Path, str, str]] = []

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        model_size: str,
    ) -> TranscriptionResult:
        self.calls.append((audio_path, language, model_size))
        return TranscriptionResult(
            transcript=self.transcript,
            language=language,
            model_size=model_size,
        )


def test_router_dispatches_by_model_backend(tmp_path: Path) -> None:
    whisper = RecordingBackend("whisper")
    vibevoice = RecordingBackend("vibevoice")
    router = TranscriptionRouterService(
        config=TranscriptionConfig(
            supported_languages=[SupportedLanguage(code="de", label="German")],
            supported_model_sizes=[
                SupportedModelSize(
                    id="small",
                    label="Small",
                    model_name="mlx-community/whisper-small-mlx",
                ),
                SupportedModelSize(
                    id="vibevoice-7b",
                    label="VibeVoice ASR 7B",
                    model_name="microsoft/VibeVoice-ASR",
                    backend="vibevoice_asr",
                ),
            ],
        ),
        backends={
            "mlx_whisper": whisper,
            "vibevoice_asr": vibevoice,
        },
    )
    audio_path = tmp_path / "audio.webm"

    whisper_result = router.transcribe(audio_path, "de", "small")
    vibevoice_result = router.transcribe(audio_path, "de", "vibevoice-7b")

    assert whisper_result.transcript == "whisper"
    assert vibevoice_result.transcript == "vibevoice"
    assert whisper.calls == [(audio_path, "de", "small")]
    assert vibevoice.calls == [(audio_path, "de", "vibevoice-7b")]


def test_router_rejects_unconfigured_model(tmp_path: Path) -> None:
    router = TranscriptionRouterService(
        config=TranscriptionConfig(),
        backends={},
    )

    with pytest.raises(
        UnsupportedModelSizeError, match="Unsupported model size: missing"
    ):
        router.transcribe(tmp_path / "audio.webm", "de", "missing")
