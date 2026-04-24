from pathlib import Path

from app.services.transcription import TranscriptionResult


class StubService:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str, str]] = []

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        model_size: str,
    ) -> TranscriptionResult:
        self.calls.append((audio_path, language, model_size))
        return TranscriptionResult(
            transcript="Hallo Welt",
            language=language,
            model_size=model_size,
            duration_seconds=1.2,
        )