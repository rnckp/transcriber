from pathlib import Path
from threading import Lock

import mlx_whisper
from mlx_whisper import load_models

from app.config import TranscriptionConfig
from app.services.transcription import (
    TranscriptionError,
    TranscriptionResult,
    UnsupportedLanguageError,
    UnsupportedModelSizeError,
)


LEGACY_MODEL_REPO_IDS = {
    "mlx-community/whisper-tiny": "mlx-community/whisper-tiny-mlx",
    "mlx-community/whisper-base": "mlx-community/whisper-base-mlx",
    "mlx-community/whisper-small": "mlx-community/whisper-small-mlx",
    "mlx-community/whisper-medium": "mlx-community/whisper-medium-mlx",
}


class MlxWhisperService:
    def __init__(self, config: TranscriptionConfig) -> None:
        self._config = config
        self._models: dict[str, Path] = {}
        self._model_locks: dict[str, Lock] = {}
        self._model_locks_guard = Lock()
        self._model_map = {model.id: model for model in config.supported_model_sizes}
        self._supported_languages = {
            language.code for language in config.supported_languages
        }

    def _get_model_lock(self, model_size: str) -> Lock:
        with self._model_locks_guard:
            return self._model_locks.setdefault(model_size, Lock())

    def _has_local_model_files(self, model_path: Path) -> bool:
        return (model_path / "config.json").exists() and (
            (model_path / "weights.safetensors").exists()
            or (model_path / "weights.npz").exists()
        )

    def _resolve_repo_id(self, repo_id: str) -> str:
        return LEGACY_MODEL_REPO_IDS.get(repo_id, repo_id)

    def _get_or_load_model(self, model_size: str) -> Path:
        if model_size in self._models:
            return self._models[model_size]

        model_lock = self._get_model_lock(model_size)
        with model_lock:
            if model_size in self._models:
                return self._models[model_size]

            model = self._model_map.get(model_size)
            if model is None:
                raise UnsupportedModelSizeError(f"Unsupported model size: {model_size}")

            self._config.cache_dir.mkdir(parents=True, exist_ok=True)
            model_path = self._config.cache_dir / model.id
            if not self._has_local_model_files(model_path):
                model_path = Path(
                    load_models.snapshot_download(
                        repo_id=self._resolve_repo_id(model.model_name),
                        local_dir=model_path,
                        token=False,
                    )
                )
            self._models[model_size] = model_path
            return self._models[model_size]

    def _extract_duration_seconds(self, result: dict[str, object]) -> float | None:
        direct_duration = result.get("duration_seconds")
        if direct_duration is None:
            direct_duration = result.get("duration")
        if isinstance(direct_duration, int | float):
            return float(direct_duration)

        segments = result.get("segments")
        if not isinstance(segments, list) or not segments:
            return None

        end_times = [
            segment.get("end")
            for segment in segments
            if isinstance(segment, dict) and isinstance(segment.get("end"), int | float)
        ]
        if not end_times:
            return None
        return float(max(end_times))

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        model_size: str,
    ) -> TranscriptionResult:
        try:
            if language not in self._supported_languages:
                raise UnsupportedLanguageError(f"Unsupported language: {language}")

            model_path = self._get_or_load_model(model_size)
            result = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=str(model_path),
                language=language,
            )
            duration_seconds = self._extract_duration_seconds(result)
        except UnsupportedLanguageError:
            raise
        except UnsupportedModelSizeError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Failed to transcribe audio: {exc}") from exc

        return TranscriptionResult(
            transcript=result["text"].strip(),
            language=language,
            model_size=model_size,
            duration_seconds=duration_seconds,
        )
