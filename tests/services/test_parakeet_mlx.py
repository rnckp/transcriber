from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.config import SupportedLanguage, SupportedModelSize, TranscriptionConfig
from app.services.parakeet_mlx import ParakeetMlxService
from app.services.transcription import UnsupportedLanguageError


def _config(tmp_path: Path) -> TranscriptionConfig:
    return TranscriptionConfig(
        cache_dir=tmp_path / "models",
        supported_languages=[
            SupportedLanguage(code="de", label="German"),
            SupportedLanguage(code="en", label="English"),
        ],
        supported_model_sizes=[
            SupportedModelSize(
                id="parakeet-v3",
                label="NVIDIA Parakeet TDT 0.6B V3",
                model_name="mlx-community/parakeet-tdt-0.6b-v3",
                backend="parakeet_mlx",
            ),
            SupportedModelSize(
                id="parakeet-v2-en",
                label="NVIDIA Parakeet TDT 0.6B V2 (English)",
                model_name="mlx-community/parakeet-tdt-0.6b-v2",
                backend="parakeet_mlx",
                language_codes=["en"],
            ),
        ],
        parakeet_chunk_duration_seconds=90.0,
        parakeet_overlap_duration_seconds=10.0,
    )


def test_parakeet_transcribes_with_timestamps_and_chunk_settings(
    tmp_path: Path,
) -> None:
    backend = MagicMock()
    backend.transcribe.return_value = SimpleNamespace(
        text=" Hallo Welt ",
        sentences=[
            SimpleNamespace(text=" Hallo ", start=0.1, end=0.8),
            SimpleNamespace(text="Welt", start=0.8, end=1.4),
        ],
    )

    with patch.object(ParakeetMlxService, "_load_model", return_value=backend):
        result = ParakeetMlxService(_config(tmp_path)).transcribe(
            tmp_path / "audio.wav",
            "de",
            "parakeet-v3",
        )

    assert result.transcript == "Hallo Welt"
    assert result.duration_seconds == 1.4
    assert [segment.text for segment in result.segments] == ["Hallo", "Welt"]
    backend.transcribe.assert_called_once_with(
        tmp_path / "audio.wav",
        chunk_duration=90.0,
        overlap_duration=10.0,
    )


def test_parakeet_caches_loaded_model(tmp_path: Path) -> None:
    backend = MagicMock()
    backend.transcribe.return_value = SimpleNamespace(text="text", sentences=[])

    with patch.object(
        ParakeetMlxService,
        "_load_model",
        return_value=backend,
    ) as load_model:
        service = ParakeetMlxService(_config(tmp_path))
        service.transcribe(tmp_path / "one.wav", "de", "parakeet-v3")
        service.transcribe(tmp_path / "two.wav", "de", "parakeet-v3")

    load_model.assert_called_once_with("mlx-community/parakeet-tdt-0.6b-v3")


def test_parakeet_rejects_language_unsupported_by_checkpoint(
    tmp_path: Path,
) -> None:
    service = ParakeetMlxService(_config(tmp_path))

    with pytest.raises(
        UnsupportedLanguageError,
        match="Model parakeet-v2-en does not support language: de",
    ):
        service.transcribe(tmp_path / "audio.wav", "de", "parakeet-v2-en")


def test_parakeet_release_drops_models_and_clears_mlx_cache(tmp_path: Path) -> None:
    service = ParakeetMlxService(_config(tmp_path))
    service._models["parakeet-v3"] = object()

    with (
        patch("app.services.parakeet_mlx.gc.collect") as collect,
        patch("app.services.parakeet_mlx.mx.clear_cache") as clear_cache,
    ):
        service.release()

    assert service._models == {}
    collect.assert_called_once_with()
    clear_cache.assert_called_once_with()


def test_parakeet_loads_model_into_configured_hugging_face_cache(
    tmp_path: Path,
) -> None:
    backend = object()

    with patch("parakeet_mlx.from_pretrained", return_value=backend) as load:
        result = ParakeetMlxService(_config(tmp_path))._load_model(
            "mlx-community/parakeet-tdt-0.6b-v3"
        )

    assert result is backend
    load.assert_called_once_with(
        "mlx-community/parakeet-tdt-0.6b-v3",
        cache_dir=tmp_path / "models" / "huggingface",
    )
