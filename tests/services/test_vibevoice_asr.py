import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock, patch

import pytest

from app.config import SupportedLanguage, SupportedModelSize, TranscriptionConfig
from app.services.vibevoice_asr import VibeVoiceASRService
from app.services.transcription import TranscriptionBusyError


def test_release_drops_vibevoice_backends_and_clears_available_mps_cache() -> None:
    service = VibeVoiceASRService(TranscriptionConfig())
    service._backends["vibevoice-7b"] = object()

    with (
        patch("gc.collect") as collect,
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.mps.empty_cache") as empty_cache,
    ):
        service.release()

    assert service._backends == {}
    collect.assert_called_once_with()
    empty_cache.assert_called_once_with()


def test_release_skips_mps_cache_when_mps_is_unavailable() -> None:
    service = VibeVoiceASRService(TranscriptionConfig())

    with (
        patch("gc.collect") as collect,
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.mps.empty_cache") as empty_cache,
    ):
        service.release()

    collect.assert_called_once_with()
    empty_cache.assert_not_called()


def test_vibevoice_formats_diarized_segments(tmp_path: Path) -> None:
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


def test_vibevoice_uses_configured_generation_settings(tmp_path: Path) -> None:
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
        vibevoice_temperature=0.2,
        vibevoice_top_p=0.8,
        vibevoice_do_sample=True,
        vibevoice_num_beams=3,
        vibevoice_repetition_penalty=1.1,
    )
    backend = MagicMock()
    backend.transcribe.return_value = {"raw_text": "transcript", "segments": []}

    with patch.object(VibeVoiceASRService, "_load_backend", return_value=backend):
        VibeVoiceASRService(config).transcribe(
            tmp_path / "audio.webm",
            "de",
            "vibevoice-7b",
        )

    assert backend.transcribe.call_args.kwargs == {
        "audio_path": str(tmp_path / "audio.webm"),
        "max_new_tokens": 4096,
        "temperature": 0.2,
        "top_p": 0.8,
        "do_sample": True,
        "num_beams": 3,
        "repetition_penalty": 1.1,
    }


def test_vibevoice_caches_backend_per_model_id(tmp_path: Path) -> None:
    config = TranscriptionConfig(
        supported_languages=[SupportedLanguage(code="de", label="German")],
        supported_model_sizes=[
            SupportedModelSize(
                id="vibevoice-7b",
                label="VibeVoice ASR 7B",
                model_name="microsoft/VibeVoice-ASR",
                backend="vibevoice_asr",
            ),
            SupportedModelSize(
                id="vibevoice-large",
                label="VibeVoice ASR Large",
                model_name="microsoft/VibeVoice-ASR-Large",
                backend="vibevoice_asr",
            ),
        ],
    )
    first_backend = MagicMock()
    first_backend.transcribe.return_value = {"raw_text": "first", "segments": []}
    second_backend = MagicMock()
    second_backend.transcribe.return_value = {"raw_text": "second", "segments": []}

    with patch.object(
        VibeVoiceASRService,
        "_load_backend",
        side_effect=[first_backend, second_backend],
    ) as mocked_load:
        service = VibeVoiceASRService(config)
        first_result = service.transcribe(tmp_path / "audio.webm", "de", "vibevoice-7b")
        second_result = service.transcribe(
            tmp_path / "audio.webm", "de", "vibevoice-large"
        )

    assert first_result.transcript == "first"
    assert second_result.transcript == "second"
    assert [call.args[0] for call in mocked_load.call_args_list] == [
        "microsoft/VibeVoice-ASR",
        "microsoft/VibeVoice-ASR-Large",
    ]


def test_vibevoice_rejects_concurrent_transcription_for_same_model(
    tmp_path: Path,
) -> None:
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
    first_started = Event()
    release_first = Event()
    backend = MagicMock()

    def transcribe(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        first_started.set()
        release_first.wait(timeout=2)
        return {"raw_text": "first", "segments": []}

    backend.transcribe.side_effect = transcribe
    service = VibeVoiceASRService(config)

    with patch.object(VibeVoiceASRService, "_load_backend", return_value=backend):
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                service.transcribe,
                tmp_path / "audio.webm",
                "de",
                "vibevoice-7b",
            )
            assert first_started.wait(timeout=2)

            with pytest.raises(TranscriptionBusyError):
                service.transcribe(tmp_path / "other.webm", "de", "vibevoice-7b")

            release_first.set()
            assert first.result(timeout=2).transcript == "first"

    assert backend.transcribe.call_count == 1


def test_vibevoice_loads_local_inference_adapter() -> None:
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
    backend = object()
    backend_class = MagicMock(return_value=backend)
    fake_module = types.SimpleNamespace(VibeVoiceASRInference=backend_class)

    with (
        patch.dict(sys.modules, {"app.services.vibevoice_inference": fake_module}),
        patch.object(VibeVoiceASRService, "_detect_device", return_value="cpu"),
        patch.object(VibeVoiceASRService, "_resolve_dtype", return_value="float32"),
        patch.object(VibeVoiceASRService, "_resolve_attention", return_value="sdpa"),
    ):
        result = VibeVoiceASRService(config)._load_backend("microsoft/VibeVoice-ASR")

    assert result is backend
    backend_class.assert_called_once_with(
        model_path="microsoft/VibeVoice-ASR",
        device="cpu",
        dtype="float32",
        attn_implementation="sdpa",
    )
