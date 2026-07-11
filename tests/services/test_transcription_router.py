from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from unittest.mock import MagicMock

import pytest

from app.config import SupportedLanguage, SupportedModelSize, TranscriptionConfig
from app.services.transcription import (
    TranscriptionError,
    TranscriptionResult,
    UnsupportedModelSizeError,
)
from app.services.transcription_router import TranscriptionRouterService


class RecordingBackend:
    def __init__(self, transcript: str, events: list[str] | None = None) -> None:
        self.transcript = transcript
        self.calls: list[tuple[Path, str, str]] = []
        self.release_count = 0
        self.events = events

    def release(self) -> None:
        self.release_count += 1
        if self.events is not None:
            self.events.append(f"release:{self.transcript}")

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        model_size: str,
    ) -> TranscriptionResult:
        self.calls.append((audio_path, language, model_size))
        if self.events is not None:
            self.events.append(f"transcribe:{self.transcript}")
        return TranscriptionResult(
            transcript=self.transcript,
            language=language,
            model_size=model_size,
        )


def _build_router(
    whisper: RecordingBackend,
    vibevoice: RecordingBackend,
) -> TranscriptionRouterService:
    return TranscriptionRouterService(
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
        backends={"mlx_whisper": whisper, "vibevoice_asr": vibevoice},
    )


def test_router_dispatches_by_model_backend(tmp_path: Path) -> None:
    whisper = RecordingBackend("whisper")
    vibevoice = RecordingBackend("vibevoice")
    router = _build_router(whisper, vibevoice)
    audio_path = tmp_path / "audio.webm"

    whisper_result = router.transcribe(audio_path, "de", "small")
    vibevoice_result = router.transcribe(audio_path, "de", "vibevoice-7b")

    assert whisper_result.transcript == "whisper"
    assert vibevoice_result.transcript == "vibevoice"
    assert whisper.calls == [(audio_path, "de", "small")]
    assert vibevoice.calls == [(audio_path, "de", "vibevoice-7b")]


def test_router_releases_previous_backend_before_switch(tmp_path: Path) -> None:
    events: list[str] = []
    whisper = RecordingBackend("whisper", events)
    vibevoice = RecordingBackend("vibevoice", events)
    router = _build_router(whisper, vibevoice)

    router.transcribe(tmp_path / "first.webm", "de", "small")
    router.transcribe(tmp_path / "second.webm", "de", "vibevoice-7b")
    router.transcribe(tmp_path / "third.webm", "de", "small")

    assert events == [
        "transcribe:whisper",
        "release:whisper",
        "transcribe:vibevoice",
        "release:vibevoice",
        "transcribe:whisper",
    ]


def test_router_reuses_same_backend_without_release(tmp_path: Path) -> None:
    whisper = RecordingBackend("whisper")
    router = _build_router(whisper, RecordingBackend("vibevoice"))

    router.transcribe(tmp_path / "first.webm", "de", "small")
    router.transcribe(tmp_path / "second.webm", "de", "small")

    assert whisper.release_count == 0


def test_router_does_not_dispatch_new_backend_when_release_fails(
    tmp_path: Path,
) -> None:
    whisper = RecordingBackend("whisper")
    vibevoice = RecordingBackend("vibevoice")
    router = _build_router(whisper, vibevoice)
    router.transcribe(tmp_path / "first.webm", "de", "small")
    whisper.release = MagicMock(side_effect=RuntimeError("cleanup failed"))

    with pytest.raises(
        TranscriptionError,
        match="Failed to release transcription backend mlx_whisper: cleanup failed",
    ):
        router.transcribe(tmp_path / "second.webm", "de", "vibevoice-7b")

    assert vibevoice.calls == []


def test_router_releases_failed_backend_on_later_switch(tmp_path: Path) -> None:
    whisper = RecordingBackend("whisper")
    vibevoice = RecordingBackend("vibevoice")
    whisper.transcribe = MagicMock(side_effect=TranscriptionError("decode failed"))
    router = _build_router(whisper, vibevoice)

    with pytest.raises(TranscriptionError, match="decode failed"):
        router.transcribe(tmp_path / "first.webm", "de", "small")
    router.transcribe(tmp_path / "second.webm", "de", "vibevoice-7b")

    assert whisper.release_count == 1


def test_router_serializes_transcription_across_backend_switches(
    tmp_path: Path,
) -> None:
    first_started = Event()
    release_first = Event()
    second_waiting = Event()
    whisper = RecordingBackend("whisper")
    vibevoice = RecordingBackend("vibevoice")
    original_transcribe = whisper.transcribe

    def blocking_transcribe(
        audio_path: Path, language: str, model_size: str
    ) -> TranscriptionResult:
        first_started.set()
        assert release_first.wait(timeout=2)
        return original_transcribe(audio_path, language, model_size)

    whisper.transcribe = blocking_transcribe  # type: ignore[method-assign]
    router = _build_router(whisper, vibevoice)
    original_wait = router._lifecycle_condition.wait

    def observed_wait(timeout: float | None = None) -> bool:
        second_waiting.set()
        return original_wait(timeout)

    router._lifecycle_condition.wait = observed_wait  # type: ignore[method-assign]

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            router.transcribe, tmp_path / "first.webm", "de", "small"
        )
        assert first_started.wait(timeout=2)
        second = executor.submit(
            router.transcribe, tmp_path / "second.webm", "de", "vibevoice-7b"
        )
        assert second_waiting.wait(timeout=2)
        assert vibevoice.calls == []

        release_first.set()
        assert first.result(timeout=2).transcript == "whisper"
        assert second.result(timeout=2).transcript == "vibevoice"


def test_router_allows_concurrent_calls_to_same_backend(tmp_path: Path) -> None:
    first_entered = Event()
    second_entered = Event()
    release_first = Event()
    backend = RecordingBackend("vibevoice")
    call_lock = Lock()
    call_count = 0

    def concurrent_transcribe(
        audio_path: Path, language: str, model_size: str
    ) -> TranscriptionResult:
        nonlocal call_count
        with call_lock:
            call_count += 1
            current_call = call_count
        if current_call == 1:
            first_entered.set()
            assert release_first.wait(timeout=2)
        else:
            second_entered.set()
        return TranscriptionResult("vibevoice", language, model_size)

    backend.transcribe = concurrent_transcribe  # type: ignore[method-assign]
    router = _build_router(RecordingBackend("whisper"), backend)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            router.transcribe, tmp_path / "first.webm", "de", "vibevoice-7b"
        )
        assert first_entered.wait(timeout=2)
        second = executor.submit(
            router.transcribe, tmp_path / "second.webm", "de", "vibevoice-7b"
        )
        assert second_entered.wait(timeout=2)
        release_first.set()
        assert first.result(timeout=2).transcript == "vibevoice"
        assert second.result(timeout=2).transcript == "vibevoice"


def test_router_rejects_unconfigured_model(tmp_path: Path) -> None:
    router = TranscriptionRouterService(
        config=TranscriptionConfig(),
        backends={},
    )

    with pytest.raises(
        UnsupportedModelSizeError, match="Unsupported model size: missing"
    ):
        router.transcribe(tmp_path / "audio.webm", "de", "missing")
