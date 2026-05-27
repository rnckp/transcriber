from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from time import sleep
from unittest.mock import patch

import pytest

from app.config import SupportedLanguage, SupportedModelSize, TranscriptionConfig
from app.services.mlx_whisper import MlxWhisperService
from app.services.transcription import UnsupportedLanguageError


def test_get_model_reuses_cached_instance(tmp_path: Path) -> None:
    config = TranscriptionConfig(
        cache_dir=tmp_path / "models",
        supported_languages=[],
        supported_model_sizes=[
            SupportedModelSize(
                id="small",
                label="Small",
                model_name="mlx-community/whisper-small-mlx",
            )
        ],
    )
    service = MlxWhisperService(config)

    expected_model_path = tmp_path / "models" / "small"
    with patch(
        "app.services.mlx_whisper.load_models.snapshot_download",
        return_value=str(expected_model_path),
    ) as mocked_download:
        first = service._get_or_load_model("small")
        second = service._get_or_load_model("small")

    assert first is second
    assert first == expected_model_path
    assert mocked_download.call_count == 1
    assert mocked_download.call_args.kwargs == {
        "repo_id": "mlx-community/whisper-small-mlx",
        "local_dir": expected_model_path,
        "token": False,
    }


def test_get_model_uses_existing_local_directory_without_hub_call(
    tmp_path: Path,
) -> None:
    config = TranscriptionConfig(
        cache_dir=tmp_path / "models",
        supported_languages=[],
        supported_model_sizes=[
            SupportedModelSize(
                id="small",
                label="Small",
                model_name="mlx-community/whisper-small-mlx",
            )
        ],
    )
    service = MlxWhisperService(config)
    expected_model_path = tmp_path / "models" / "small"
    expected_model_path.mkdir(parents=True)
    (expected_model_path / "config.json").write_text("{}", encoding="utf-8")
    (expected_model_path / "weights.safetensors").write_bytes(b"weights")

    with patch(
        "app.services.mlx_whisper.load_models.snapshot_download"
    ) as mocked_download:
        model_path = service._get_or_load_model("small")

    assert model_path == expected_model_path
    assert mocked_download.call_count == 0


def test_get_model_normalizes_legacy_repo_id(tmp_path: Path) -> None:
    config = TranscriptionConfig(
        cache_dir=tmp_path / "models",
        supported_languages=[],
        supported_model_sizes=[
            SupportedModelSize(
                id="small",
                label="Small",
                model_name="mlx-community/whisper-small",
            )
        ],
    )
    service = MlxWhisperService(config)
    expected_model_path = tmp_path / "models" / "small"

    with patch(
        "app.services.mlx_whisper.load_models.snapshot_download",
        return_value=str(expected_model_path),
    ) as mocked_download:
        model_path = service._get_or_load_model("small")

    assert model_path == expected_model_path
    assert mocked_download.call_args.kwargs == {
        "repo_id": "mlx-community/whisper-small-mlx",
        "local_dir": expected_model_path,
        "token": False,
    }


def test_transcribe_uses_selected_model_size_once_per_process(tmp_path: Path) -> None:
    config = TranscriptionConfig(
        cache_dir=tmp_path / "models",
        supported_languages=[
            SupportedLanguage(code="de", label="German"),
        ],
        supported_model_sizes=[
            SupportedModelSize(
                id="small",
                label="Small",
                model_name="mlx-community/whisper-small-mlx",
            )
        ],
    )
    service = MlxWhisperService(config)
    audio_path = tmp_path / "audio.webm"
    audio_path.write_bytes(b"audio")
    expected_model_path = tmp_path / "models" / "small"

    with (
        patch(
            "app.services.mlx_whisper.load_models.snapshot_download",
            return_value=str(expected_model_path),
        ) as mocked_download,
        patch(
            "app.services.mlx_whisper.mlx_whisper.transcribe",
            return_value={"text": " Hallo Welt "},
        ) as mocked,
    ):
        first = service.transcribe(audio_path, "de", "small")
        second = service.transcribe(audio_path, "de", "small")

    assert first.transcript == "Hallo Welt"
    assert second.transcript == "Hallo Welt"
    assert service._models["small"] == expected_model_path
    assert mocked_download.call_count == 1
    assert mocked.call_count == 2
    assert mocked.call_args.kwargs["path_or_hf_repo"] == str(expected_model_path)


def test_transcribe_returns_duration_when_available(tmp_path: Path) -> None:
    config = TranscriptionConfig(
        cache_dir=tmp_path / "models",
        supported_languages=[
            SupportedLanguage(code="de", label="German"),
        ],
        supported_model_sizes=[
            SupportedModelSize(
                id="small",
                label="Small",
                model_name="mlx-community/whisper-small-mlx",
            )
        ],
    )
    service = MlxWhisperService(config)
    audio_path = tmp_path / "audio.webm"
    audio_path.write_bytes(b"audio")
    expected_model_path = tmp_path / "models" / "small"

    with (
        patch(
            "app.services.mlx_whisper.load_models.snapshot_download",
            return_value=str(expected_model_path),
        ),
        patch(
            "app.services.mlx_whisper.mlx_whisper.transcribe",
            return_value={"text": " Hallo Welt ", "duration": 3.5},
        ),
    ):
        result = service.transcribe(audio_path, "de", "small")

    assert result.duration_seconds == 3.5


def test_transcribe_preserves_zero_duration(tmp_path: Path) -> None:
    config = TranscriptionConfig(
        cache_dir=tmp_path / "models",
        supported_languages=[
            SupportedLanguage(code="de", label="German"),
        ],
        supported_model_sizes=[
            SupportedModelSize(
                id="small",
                label="Small",
                model_name="mlx-community/whisper-small-mlx",
            )
        ],
    )
    service = MlxWhisperService(config)
    audio_path = tmp_path / "audio.webm"
    audio_path.write_bytes(b"audio")
    expected_model_path = tmp_path / "models" / "small"

    with (
        patch(
            "app.services.mlx_whisper.load_models.snapshot_download",
            return_value=str(expected_model_path),
        ),
        patch(
            "app.services.mlx_whisper.mlx_whisper.transcribe",
            return_value={"text": " Hallo Welt ", "duration_seconds": 0.0},
        ),
    ):
        result = service.transcribe(audio_path, "de", "small")

    assert result.duration_seconds == 0.0


def test_transcribe_rejects_unsupported_language(tmp_path: Path) -> None:
    config = TranscriptionConfig(
        cache_dir=tmp_path / "models",
        supported_languages=[
            SupportedLanguage(code="de", label="German"),
        ],
        supported_model_sizes=[
            SupportedModelSize(
                id="small",
                label="Small",
                model_name="mlx-community/whisper-small-mlx",
            )
        ],
    )
    service = MlxWhisperService(config)
    audio_path = tmp_path / "audio.webm"
    audio_path.write_bytes(b"audio")

    with pytest.raises(UnsupportedLanguageError, match="Unsupported language: en"):
        service.transcribe(audio_path, "en", "small")


def test_get_model_loads_once_under_concurrency(tmp_path: Path) -> None:
    config = TranscriptionConfig(
        cache_dir=tmp_path / "models",
        supported_languages=[],
        supported_model_sizes=[
            SupportedModelSize(
                id="small",
                label="Small",
                model_name="mlx-community/whisper-small-mlx",
            )
        ],
    )
    service = MlxWhisperService(config)
    expected_model_path = tmp_path / "models" / "small"
    call_count = 0
    counter_lock = Lock()

    def fake_download(*, repo_id: str, local_dir: Path, token: bool) -> str:
        del repo_id, token
        nonlocal call_count
        with counter_lock:
            call_count += 1
        sleep(0.05)
        return str(local_dir)

    with patch(
        "app.services.mlx_whisper.load_models.snapshot_download",
        side_effect=fake_download,
    ):
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(service._get_or_load_model, "small")
            second = executor.submit(service._get_or_load_model, "small")

        assert first.result() == expected_model_path
        assert second.result() == expected_model_path

    assert call_count == 1
