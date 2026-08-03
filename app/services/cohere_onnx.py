import gc
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import librosa
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download

from app.config import TranscriptionConfig
from app.services.transcription import (
    TranscriptionBusyError,
    TranscriptionError,
    TranscriptionResult,
    UnsupportedLanguageError,
    UnsupportedModelSizeError,
)

MODEL_FILES = (
    "cohere-encoder.int8.onnx",
    "cohere-encoder.int8.onnx.data",
    "cohere-decoder.int8.onnx",
    "tokens.txt",
)
START_OF_TRANSCRIPT = "<|startoftranscript|>"
PUNCTUATION = "<|pnc|>"
NO_TIMESTAMPS = "<|notimestamp|>"
INVERSE_TEXT_NORMALIZATION = "<|itn|>"
END_OF_TEXT = "<|endoftext|>"
CPU_EXECUTION_PROVIDER = "CPUExecutionProvider"
COREML_EXECUTION_PROVIDER = "CoreMLExecutionProvider"


@dataclass(slots=True)
class CohereOnnxBackend:
    encoder: Any
    decoder: Any
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]


class CohereOnnxService:
    """Run the INT8 Cohere Transcribe export with Core ML when available."""

    def __init__(self, config: TranscriptionConfig) -> None:
        self._config = config
        self._model_map = {
            model.id: model
            for model in config.supported_model_sizes
            if model.backend == "cohere_onnx"
        }
        self._supported_languages = {
            language.code for language in config.supported_languages
        }
        self._backends: dict[str, CohereOnnxBackend] = {}
        self._backend_lock = Lock()
        self._inference_locks: dict[str, Lock] = {}
        self._inference_locks_guard = Lock()

    def release(self) -> None:
        with self._backend_lock:
            self._backends.clear()
        gc.collect()

    def _get_inference_lock(self, model_id: str) -> Lock:
        with self._inference_locks_guard:
            return self._inference_locks.setdefault(model_id, Lock())

    def _model_directory(self, model_id: str, repo_id: str) -> Path:
        model_directory = self._config.cache_dir / model_id
        if all((model_directory / filename).is_file() for filename in MODEL_FILES):
            return model_directory

        model_directory.mkdir(parents=True, exist_ok=True)
        return Path(
            snapshot_download(
                repo_id=repo_id,
                local_dir=model_directory,
                allow_patterns=list(MODEL_FILES),
                token=False,
            )
        )

    def _execution_providers(self) -> list[str]:
        requested_provider = self._config.cohere_execution_provider
        if requested_provider in {"auto", "cpu"}:
            return [CPU_EXECUTION_PROVIDER]
        if COREML_EXECUTION_PROVIDER not in set(ort.get_available_providers()):
            raise TranscriptionError(
                "Core ML execution was requested but is unavailable in ONNX Runtime."
            )
        return [COREML_EXECUTION_PROVIDER, CPU_EXECUTION_PROVIDER]

    def _load_vocabulary(
        self,
        tokens_path: Path,
    ) -> tuple[dict[str, int], dict[int, str]]:
        token_to_id: dict[str, int] = {}
        with tokens_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.rstrip("\n")
                try:
                    token, raw_token_id = stripped.rsplit(maxsplit=1)
                    token_id = int(raw_token_id)
                except ValueError as exc:
                    raise TranscriptionError(
                        f"Invalid Cohere vocabulary entry on line {line_number}."
                    ) from exc
                token_to_id[token] = token_id
        id_to_token = {token_id: token for token, token_id in token_to_id.items()}
        return token_to_id, id_to_token

    def _load_backend(self, model_id: str, repo_id: str) -> CohereOnnxBackend:
        model_directory = self._model_directory(model_id, repo_id)
        providers = self._execution_providers()
        token_to_id, id_to_token = self._load_vocabulary(model_directory / "tokens.txt")
        required_tokens = {
            START_OF_TRANSCRIPT,
            PUNCTUATION,
            NO_TIMESTAMPS,
            INVERSE_TEXT_NORMALIZATION,
            END_OF_TEXT,
        }
        missing_tokens = required_tokens - token_to_id.keys()
        if missing_tokens:
            missing = ", ".join(sorted(missing_tokens))
            raise TranscriptionError(
                f"Cohere vocabulary is missing required tokens: {missing}"
            )
        return CohereOnnxBackend(
            encoder=ort.InferenceSession(
                model_directory / "cohere-encoder.int8.onnx",
                providers=providers,
            ),
            decoder=ort.InferenceSession(
                model_directory / "cohere-decoder.int8.onnx",
                providers=providers,
            ),
            token_to_id=token_to_id,
            id_to_token=id_to_token,
        )

    def _get_or_load_backend(self, model_id: str) -> CohereOnnxBackend:
        configured_model = self._model_map.get(model_id)
        if configured_model is None:
            raise UnsupportedModelSizeError(f"Unsupported model size: {model_id}")

        backend = self._backends.get(model_id)
        if backend is not None:
            return backend
        with self._backend_lock:
            backend = self._backends.get(model_id)
            if backend is None:
                backend = self._load_backend(model_id, configured_model.model_name)
                self._backends[model_id] = backend
        return backend

    def _decode(
        self,
        backend: CohereOnnxBackend,
        audio: np.ndarray,
        language: str,
    ) -> str:
        language_token = f"<|{language}|>"
        try:
            prompt = [
                backend.token_to_id[START_OF_TRANSCRIPT],
                backend.token_to_id[language_token],
                backend.token_to_id[PUNCTUATION],
                backend.token_to_id[NO_TIMESTAMPS],
                backend.token_to_id[INVERSE_TEXT_NORMALIZATION],
            ]
        except KeyError as exc:
            raise UnsupportedLanguageError(
                f"Cohere model does not support language: {language}"
            ) from exc

        cross_k, cross_v = backend.encoder.run(
            None,
            {"audio": audio.reshape(1, -1)},
        )
        cache_shape = (8, 1, 8, 1024, 128)
        cache_k = np.zeros(cache_shape, dtype=np.float32)
        cache_v = np.zeros(cache_shape, dtype=np.float32)
        current_tokens = np.asarray([prompt], dtype=np.int64)
        offset = 0
        generated_token_ids: list[int] = []
        end_token_id = backend.token_to_id[END_OF_TEXT]

        for _ in range(self._config.cohere_max_new_tokens):
            logits, cache_k, cache_v = backend.decoder.run(
                None,
                {
                    "tokens": current_tokens,
                    "in_n_layer_self_k_cache": cache_k,
                    "in_n_layer_self_v_cache": cache_v,
                    "n_layer_cross_k": cross_k,
                    "n_layer_cross_v": cross_v,
                    "offset": np.asarray(offset, dtype=np.int64),
                },
            )
            next_token_id = int(np.argmax(logits[0, -1, :]))
            if next_token_id == end_token_id:
                break
            generated_token_ids.append(next_token_id)
            offset += current_tokens.shape[1]
            current_tokens = np.asarray([[next_token_id]], dtype=np.int64)

        pieces = [
            backend.id_to_token[token_id]
            for token_id in generated_token_ids
            if token_id >= 256
            and token_id in backend.id_to_token
            and not backend.id_to_token[token_id].startswith("<|")
        ]
        return "".join(pieces).replace("▁", " ").strip()

    def _validate_request(self, language: str, model_id: str) -> None:
        if language not in self._supported_languages:
            raise UnsupportedLanguageError(f"Unsupported language: {language}")
        configured_model = self._model_map.get(model_id)
        if configured_model is None:
            raise UnsupportedModelSizeError(f"Unsupported model size: {model_id}")
        if (
            configured_model.language_codes is not None
            and language not in configured_model.language_codes
        ):
            raise UnsupportedLanguageError(
                f"Model {model_id} does not support language: {language}"
            )

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        model_size: str,
    ) -> TranscriptionResult:
        self._validate_request(language, model_size)

        try:
            backend = self._get_or_load_backend(model_size)
            inference_lock = self._get_inference_lock(model_size)
            if not inference_lock.acquire(blocking=False):
                raise TranscriptionBusyError(
                    "A transcription is already running for this model."
                )
            try:
                audio, sample_rate = librosa.load(
                    audio_path,
                    sr=16_000,
                    mono=True,
                    dtype=np.float32,
                )
                transcript = self._decode(backend, audio, language)
            finally:
                inference_lock.release()
        except (
            TranscriptionBusyError,
            TranscriptionError,
            UnsupportedLanguageError,
            UnsupportedModelSizeError,
        ):
            raise
        except Exception as exc:
            raise TranscriptionError(f"Failed to transcribe audio: {exc}") from exc

        return TranscriptionResult(
            transcript=transcript,
            language=language,
            model_size=model_size,
            duration_seconds=float(len(audio) / sample_rate),
        )
