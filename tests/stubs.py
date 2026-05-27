from pathlib import Path

from app.services.transcription import TranscriptionResult, TranscriptionSegment


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


class DiarizingStubService(StubService):
    def transcribe(
        self,
        audio_path: Path,
        language: str,
        model_size: str,
    ) -> TranscriptionResult:
        self.calls.append((audio_path, language, model_size))
        return TranscriptionResult(
            transcript="Speaker 1: Hallo Welt\nSpeaker 2: Guten Tag",
            language=language,
            model_size=model_size,
            duration_seconds=2.4,
            segments=[
                TranscriptionSegment(
                    speaker="1",
                    start_time=0.0,
                    end_time=1.1,
                    text="Hallo Welt",
                ),
                TranscriptionSegment(
                    speaker="2",
                    start_time=1.1,
                    end_time=2.4,
                    text="Guten Tag",
                ),
            ],
        )
