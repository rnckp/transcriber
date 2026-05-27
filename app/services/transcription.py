from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class TranscriptionSegment:
    speaker: str
    text: str
    start_time: float | None = None
    end_time: float | None = None


@dataclass(slots=True)
class TranscriptionResult:
    transcript: str
    language: str
    model_size: str
    duration_seconds: float | None = None
    segments: list[TranscriptionSegment] | None = None


class TranscriptionError(Exception):
    """Raised when transcription cannot be completed."""


class UnsupportedLanguageError(TranscriptionError):
    """Raised when the requested language is not configured."""


class UnsupportedModelSizeError(TranscriptionError):
    """Raised when the requested model size is not configured."""


class TranscriptionService(Protocol):
    def transcribe(
        self,
        audio_path: Path,
        language: str,
        model_size: str,
    ) -> TranscriptionResult: ...
