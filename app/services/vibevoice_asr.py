from pathlib import Path
from threading import Lock
from typing import Any

from app.config import TranscriptionConfig
from app.services.transcription import (
    TranscriptionError,
    TranscriptionResult,
    TranscriptionSegment,
    UnsupportedLanguageError,
    UnsupportedModelSizeError,
)


class VibeVoiceASRService:
    def __init__(self, config: TranscriptionConfig) -> None:
        self._config = config
        self._model_map = {
            model.id: model
            for model in config.supported_model_sizes
            if model.backend == "vibevoice_asr"
        }
        self._supported_languages = {
            language.code for language in config.supported_languages
        }
        self._backends: dict[str, Any] = {}
        self._backend_lock = Lock()

    def _detect_device(self) -> str:
        if self._config.vibevoice_device != "auto":
            return self._config.vibevoice_device

        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if hasattr(torch.backends, "xpu") and torch.backends.xpu.is_available():
            return "xpu"
        return "cpu"

    def _resolve_dtype(self, device: str) -> Any:
        import torch

        dtype = self._config.vibevoice_dtype
        if dtype == "auto":
            if device in {"cuda", "xpu"}:
                return torch.bfloat16
            if device == "mps":
                return torch.float16
            return torch.float32

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        try:
            return dtype_map[dtype]
        except KeyError as exc:
            raise TranscriptionError(f"Unsupported VibeVoice dtype: {dtype}") from exc

    def _resolve_attention(self, device: str) -> str:
        if self._config.vibevoice_attention != "auto":
            return self._config.vibevoice_attention
        if device == "cuda":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                return "sdpa"
            return "flash_attention_2"
        return "sdpa"

    def _load_backend(self, model_name: str) -> Any:
        try:
            from app.services.vibevoice_inference import VibeVoiceASRInference
        except ImportError as exc:
            raise TranscriptionError(
                "VibeVoice imports failed. Run `uv sync` to install the declared "
                "VibeVoice runtime dependency stack."
            ) from exc

        device = self._detect_device()
        return VibeVoiceASRInference(
            model_path=model_name,
            device=device,
            dtype=self._resolve_dtype(device),
            attn_implementation=self._resolve_attention(device),
        )

    def _get_or_load_backend(self, model_size: str) -> Any:
        model = self._model_map.get(model_size)
        if model is None:
            raise UnsupportedModelSizeError(f"Unsupported model size: {model_size}")

        backend = self._backends.get(model_size)
        if backend is not None:
            return backend

        with self._backend_lock:
            backend = self._backends.get(model_size)
            if backend is None:
                backend = self._load_backend(model.model_name)
                self._backends[model_size] = backend
            return backend

    def _coerce_time(self, value: object) -> float | None:
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def _normalize_segments(
        self,
        raw_segments: object,
    ) -> list[TranscriptionSegment]:
        if not isinstance(raw_segments, list):
            return []

        segments: list[TranscriptionSegment] = []
        for raw_segment in raw_segments:
            if not isinstance(raw_segment, dict):
                continue

            text = str(raw_segment.get("text", "")).strip()
            if not text:
                continue

            speaker = str(raw_segment.get("speaker_id", "unknown")).strip() or "unknown"
            segments.append(
                TranscriptionSegment(
                    speaker=speaker,
                    text=text,
                    start_time=self._coerce_time(raw_segment.get("start_time")),
                    end_time=self._coerce_time(raw_segment.get("end_time")),
                )
            )

        return segments

    def _format_segment(self, segment: TranscriptionSegment) -> str:
        return f"Speaker {segment.speaker}: {segment.text}"

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        model_size: str,
    ) -> TranscriptionResult:
        if language not in self._supported_languages:
            raise UnsupportedLanguageError(f"Unsupported language: {language}")

        try:
            backend = self._get_or_load_backend(model_size)
            result = backend.transcribe(
                audio_path=str(audio_path),
                max_new_tokens=self._config.vibevoice_max_new_tokens,
                temperature=self._config.vibevoice_temperature,
                top_p=self._config.vibevoice_top_p,
                do_sample=self._config.vibevoice_do_sample,
                num_beams=self._config.vibevoice_num_beams,
                repetition_penalty=self._config.vibevoice_repetition_penalty,
            )
            segments = self._normalize_segments(result.get("segments"))
            transcript = "\n\n".join(
                self._format_segment(segment) for segment in segments
            )
            if not transcript:
                transcript = str(result.get("raw_text", "")).strip()
            duration_seconds = max(
                (
                    segment.end_time
                    for segment in segments
                    if segment.end_time is not None
                ),
                default=None,
            )
        except UnsupportedLanguageError:
            raise
        except UnsupportedModelSizeError:
            raise
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Failed to transcribe audio: {exc}") from exc

        return TranscriptionResult(
            transcript=transcript,
            language=language,
            model_size=model_size,
            duration_seconds=duration_seconds,
            segments=segments,
        )
