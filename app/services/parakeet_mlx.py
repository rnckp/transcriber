import gc
from pathlib import Path
from threading import Lock
from typing import Any

import mlx.core as mx

from app.config import TranscriptionConfig
from app.services.transcription import (
    TranscriptionBusyError,
    TranscriptionError,
    TranscriptionResult,
    TranscriptionSegment,
    UnsupportedLanguageError,
    UnsupportedModelSizeError,
)


class ParakeetMlxService:
    """Run NVIDIA Parakeet checkpoints through MLX on Apple Silicon."""

    def __init__(self, config: TranscriptionConfig) -> None:
        self._config = config
        self._model_map = {
            model.id: model
            for model in config.supported_model_sizes
            if model.backend == "parakeet_mlx"
        }
        self._supported_languages = {
            language.code for language in config.supported_languages
        }
        self._models: dict[str, Any] = {}
        self._model_lock = Lock()
        self._inference_locks: dict[str, Lock] = {}
        self._inference_locks_guard = Lock()

    def release(self) -> None:
        with self._model_lock:
            self._models.clear()
        gc.collect()
        mx.clear_cache()

    def _get_inference_lock(self, model_id: str) -> Lock:
        with self._inference_locks_guard:
            return self._inference_locks.setdefault(model_id, Lock())

    def _load_model(self, model_name: str) -> Any:
        try:
            from parakeet_mlx import from_pretrained
        except ImportError as exc:
            raise TranscriptionError(
                "Parakeet MLX is unavailable. Run `uv sync` to install the "
                "declared Apple Silicon runtime."
            ) from exc

        return from_pretrained(
            model_name,
            cache_dir=self._config.cache_dir / "huggingface",
        )

    def _get_or_load_model(self, model_id: str) -> Any:
        configured_model = self._model_map.get(model_id)
        if configured_model is None:
            raise UnsupportedModelSizeError(f"Unsupported model size: {model_id}")

        model = self._models.get(model_id)
        if model is not None:
            return model

        with self._model_lock:
            model = self._models.get(model_id)
            if model is None:
                model = self._load_model(configured_model.model_name)
                self._models[model_id] = model
        return model

    def _validate_language(self, language: str, model_id: str) -> None:
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
        self._validate_language(language, model_size)

        try:
            model = self._get_or_load_model(model_size)
            inference_lock = self._get_inference_lock(model_size)
            if not inference_lock.acquire(blocking=False):
                raise TranscriptionBusyError(
                    "A transcription is already running for this model."
                )
            try:
                result = model.transcribe(
                    audio_path,
                    chunk_duration=self._config.parakeet_chunk_duration_seconds,
                    overlap_duration=self._config.parakeet_overlap_duration_seconds,
                )
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

        segments = [
            TranscriptionSegment(
                speaker="unknown",
                text=sentence.text.strip(),
                start_time=float(sentence.start),
                end_time=float(sentence.end),
            )
            for sentence in result.sentences
            if sentence.text.strip()
        ]
        duration_seconds = max(
            (segment.end_time for segment in segments if segment.end_time is not None),
            default=None,
        )
        return TranscriptionResult(
            transcript=result.text.strip(),
            language=language,
            model_size=model_size,
            duration_seconds=duration_seconds,
            segments=segments,
        )
