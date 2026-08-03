from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.config import SupportedLanguage, SupportedModelSize, TranscriptionConfig
from app.services.cohere_onnx import (
    END_OF_TEXT,
    INVERSE_TEXT_NORMALIZATION,
    NO_TIMESTAMPS,
    PUNCTUATION,
    START_OF_TRANSCRIPT,
    CohereOnnxBackend,
    CohereOnnxService,
)
from app.services.transcription import TranscriptionError


def _config(tmp_path: Path, provider: str = "auto") -> TranscriptionConfig:
    return TranscriptionConfig(
        cache_dir=tmp_path / "models",
        supported_languages=[
            SupportedLanguage(code="de", label="German"),
            SupportedLanguage(code="en", label="English"),
        ],
        supported_model_sizes=[
            SupportedModelSize(
                id="cohere-transcribe-int8",
                label="Cohere Transcribe INT8",
                model_name="smcleod/cohere-transcribe-03-2026-int8",
                backend="cohere_onnx",
                language_codes=["de", "en"],
            )
        ],
        cohere_execution_provider=provider,
    )


def _backend() -> CohereOnnxBackend:
    tokens = {
        END_OF_TEXT: 3,
        START_OF_TRANSCRIPT: 4,
        PUNCTUATION: 5,
        INVERSE_TEXT_NORMALIZATION: 8,
        NO_TIMESTAMPS: 11,
        "<|de|>": 42,
        "▁Hallo": 256,
        "▁Welt": 257,
    }
    return CohereOnnxBackend(
        encoder=MagicMock(),
        decoder=MagicMock(),
        token_to_id=tokens,
        id_to_token={token_id: token for token, token_id in tokens.items()},
    )


def test_cohere_decode_runs_encoder_and_cached_decoder(tmp_path: Path) -> None:
    backend = _backend()
    cross_k = np.zeros((8, 1, 2, 1024), dtype=np.float32)
    cross_v = np.zeros((8, 1, 2, 1024), dtype=np.float32)
    backend.encoder.run.return_value = [cross_k, cross_v]
    generated_ids = iter([256, 257, 3])

    def decoder_run(output_names, inputs):  # type: ignore[no-untyped-def]
        del output_names
        logits = np.zeros((1, inputs["tokens"].shape[1], 300), dtype=np.float32)
        logits[0, -1, next(generated_ids)] = 1.0
        return [
            logits,
            inputs["in_n_layer_self_k_cache"],
            inputs["in_n_layer_self_v_cache"],
        ]

    backend.decoder.run.side_effect = decoder_run
    service = CohereOnnxService(_config(tmp_path))

    transcript = service._decode(
        backend,
        np.asarray([0.1, -0.1], dtype=np.float32),
        "de",
    )

    assert transcript == "Hallo Welt"
    assert backend.decoder.run.call_count == 3
    offsets = [
        int(call.args[1]["offset"]) for call in backend.decoder.run.call_args_list
    ]
    assert offsets == [0, 5, 6]


def test_cohere_transcribe_loads_audio_at_16khz(tmp_path: Path) -> None:
    service = CohereOnnxService(_config(tmp_path))
    backend = _backend()
    audio = np.zeros(32_000, dtype=np.float32)

    with (
        patch.object(service, "_get_or_load_backend", return_value=backend),
        patch(
            "app.services.cohere_onnx.librosa.load", return_value=(audio, 16_000)
        ) as load,
        patch.object(service, "_decode", return_value="Hallo Welt") as decode,
    ):
        result = service.transcribe(
            tmp_path / "audio.m4a",
            "de",
            "cohere-transcribe-int8",
        )

    assert result.transcript == "Hallo Welt"
    assert result.duration_seconds == 2.0
    load.assert_called_once_with(
        tmp_path / "audio.m4a",
        sr=16_000,
        mono=True,
        dtype=np.float32,
    )
    decode.assert_called_once()
    decode_backend, decode_audio, decode_language = decode.call_args.args
    assert decode_backend is backend
    np.testing.assert_array_equal(decode_audio, audio)
    assert decode_language == "de"


def test_cohere_transcribe_splits_audio_to_fit_encoder_context(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.cohere_chunk_duration_seconds = 2.0
    service = CohereOnnxService(config)
    backend = _backend()
    audio = np.zeros(5 * 16_000, dtype=np.float32)

    with (
        patch.object(service, "_get_or_load_backend", return_value=backend),
        patch("app.services.cohere_onnx.librosa.load", return_value=(audio, 16_000)),
        patch.object(
            service,
            "_decode",
            side_effect=["first", "second", "third"],
        ) as decode,
    ):
        result = service.transcribe(
            tmp_path / "audio.m4a",
            "de",
            "cohere-transcribe-int8",
        )

    assert result.transcript == "first second third"
    assert [call.args[1].shape[0] for call in decode.call_args_list] == [
        32_000,
        32_000,
        16_000,
    ]


def test_cohere_auto_provider_uses_cpu_for_compatibility(tmp_path: Path) -> None:
    service = CohereOnnxService(_config(tmp_path))

    providers = service._execution_providers()

    assert providers == ["CPUExecutionProvider"]


def test_cohere_required_coreml_provider_fails_clearly(tmp_path: Path) -> None:
    service = CohereOnnxService(_config(tmp_path, provider="coreml"))

    with (
        patch(
            "app.services.cohere_onnx.ort.get_available_providers",
            return_value=["CPUExecutionProvider"],
        ),
        pytest.raises(TranscriptionError, match="Core ML execution was requested"),
    ):
        service._execution_providers()


def test_cohere_uses_existing_model_files_without_download(tmp_path: Path) -> None:
    service = CohereOnnxService(_config(tmp_path))
    model_directory = tmp_path / "models" / "cohere-transcribe-int8"
    model_directory.mkdir(parents=True)
    for filename in (
        "cohere-encoder.int8.onnx",
        "cohere-encoder.int8.onnx.data",
        "cohere-decoder.int8.onnx",
        "tokens.txt",
    ):
        (model_directory / filename).touch()

    with patch("app.services.cohere_onnx.snapshot_download") as download:
        result = service._model_directory(
            "cohere-transcribe-int8",
            "smcleod/cohere-transcribe-03-2026-int8",
        )

    assert result == model_directory
    download.assert_not_called()
