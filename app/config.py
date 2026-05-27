from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

LoggingLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
TranscriptionBackend = Literal["mlx_whisper", "vibevoice_asr"]


class SupportedLanguage(BaseModel):
    code: str
    label: str


class SupportedModelSize(BaseModel):
    id: str
    label: str
    model_name: str
    backend: TranscriptionBackend = "mlx_whisper"


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000

    @field_validator("port")
    @classmethod
    def validate_port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            msg = "port must be a valid TCP port between 1 and 65535"
            raise ValueError(msg)
        return value


class TranscriptionConfig(BaseModel):
    cache_dir: Path = Path(".cache/models")
    max_upload_size_mb: int = 25
    upload_chunk_size_mb: int = 1
    default_upload_filename: str = "recording.webm"
    supported_languages: list[SupportedLanguage] = Field(default_factory=list)
    supported_model_sizes: list[SupportedModelSize] = Field(default_factory=list)
    vibevoice_repo_path: Path | None = None
    vibevoice_device: str = "auto"
    vibevoice_dtype: str = "auto"
    vibevoice_attention: str = "auto"
    vibevoice_max_new_tokens: int = 4096
    vibevoice_temperature: float = 0.0
    vibevoice_top_p: float = 1.0
    vibevoice_do_sample: bool = False
    vibevoice_num_beams: int = 1
    vibevoice_repetition_penalty: float = 1.0

    @field_validator("max_upload_size_mb")
    @classmethod
    def validate_max_upload_size_mb(cls, value: int) -> int:
        if value <= 0:
            msg = "max_upload_size_mb must be greater than 0"
            raise ValueError(msg)
        return value

    @field_validator("upload_chunk_size_mb")
    @classmethod
    def validate_upload_chunk_size_mb(cls, value: int) -> int:
        if value <= 0:
            msg = "upload_chunk_size_mb must be greater than 0"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def validate_upload_limits(self) -> "TranscriptionConfig":
        if self.upload_chunk_size_mb > self.max_upload_size_mb:
            msg = (
                "upload_chunk_size_mb must be less than or equal to max_upload_size_mb"
            )
            raise ValueError(msg)
        return self

    @field_validator("default_upload_filename")
    @classmethod
    def validate_default_upload_filename(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            msg = "default_upload_filename must not be empty"
            raise ValueError(msg)
        return cleaned

    @field_validator("supported_languages")
    @classmethod
    def validate_supported_languages(
        cls,
        value: list[SupportedLanguage],
    ) -> list[SupportedLanguage]:
        if not value:
            msg = "supported_languages must include at least one item"
            raise ValueError(msg)
        codes = [language.code for language in value]
        if len(codes) != len(set(codes)):
            msg = "supported language codes must be unique"
            raise ValueError(msg)
        return value

    @field_validator("supported_model_sizes")
    @classmethod
    def validate_supported_model_sizes(
        cls,
        value: list[SupportedModelSize],
    ) -> list[SupportedModelSize]:
        if not value:
            msg = "supported_model_sizes must include at least one item"
            raise ValueError(msg)
        model_ids = [model.id for model in value]
        if len(model_ids) != len(set(model_ids)):
            msg = "supported model ids must be unique"
            raise ValueError(msg)
        return value

    @field_validator(
        "vibevoice_device",
        "vibevoice_dtype",
        "vibevoice_attention",
    )
    @classmethod
    def validate_vibevoice_setting(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            msg = "VibeVoice settings must not be empty"
            raise ValueError(msg)
        return cleaned

    @field_validator("vibevoice_max_new_tokens")
    @classmethod
    def validate_vibevoice_max_new_tokens(cls, value: int) -> int:
        if value <= 0:
            msg = "vibevoice_max_new_tokens must be greater than 0"
            raise ValueError(msg)
        return value

    @field_validator("vibevoice_temperature")
    @classmethod
    def validate_vibevoice_temperature(cls, value: float) -> float:
        if value < 0:
            msg = "vibevoice_temperature must be greater than or equal to 0"
            raise ValueError(msg)
        return value

    @field_validator("vibevoice_top_p")
    @classmethod
    def validate_vibevoice_top_p(cls, value: float) -> float:
        if not 0 < value <= 1:
            msg = "vibevoice_top_p must be greater than 0 and less than or equal to 1"
            raise ValueError(msg)
        return value

    @field_validator("vibevoice_num_beams")
    @classmethod
    def validate_vibevoice_num_beams(cls, value: int) -> int:
        if value <= 0:
            msg = "vibevoice_num_beams must be greater than 0"
            raise ValueError(msg)
        return value

    @field_validator("vibevoice_repetition_penalty")
    @classmethod
    def validate_vibevoice_repetition_penalty(cls, value: float) -> float:
        if value <= 0:
            msg = "vibevoice_repetition_penalty must be greater than 0"
            raise ValueError(msg)
        return value


class LoggingConfig(BaseModel):
    level: LoggingLevel = "INFO"


class UiConfig(BaseModel):
    copy_feedback_ms: int = 1400
    progress_model_factors: dict[str, float] = Field(
        default_factory=lambda: {
            "tiny": 0.35,
            "base": 0.5,
            "small": 0.75,
            "medium": 1.1,
            "large": 1.6,
            "vibevoice-7b": 2.6,
        }
    )
    progress_default_model_factor: float = 1.0
    progress_default_audio_seconds: float = 30.0
    progress_min_processing_seconds: float = 8.0
    progress_metadata_timeout_ms: int = 1500
    progress_update_interval_ms: int = 1000
    progress_upload_fraction: float = 0.2
    progress_transcription_fraction: float = 0.75
    progress_max_transcription_fraction: float = 0.95
    progress_min_visible_percent: int = 4
    processing_status_message: str = "Transcribing audio locally..."
    upload_status_message: str = "Processing audio file locally..."
    upload_ready_status_message: str = "Ready to process {filename}."
    empty_recording_message: str = "Record a little audio before stopping."
    missing_upload_message: str = "Choose an audio file before processing."
    ready_status_message: str = "Ready to record or process audio."
    recording_status_message: str = "Recording..."
    completion_status_message: str = "Transcription complete."
    processing_network_error_message: str = (
        "Processing failed before the server returned a response."
    )
    microphone_blocked_message: str = "Microphone access is blocked. Allow access in your browser settings and try again."
    recording_start_error_message: str = (
        "Recording could not start. Check your microphone and try again."
    )
    clipboard_unavailable_message: str = (
        "Clipboard access is unavailable in this browser."
    )
    save_success_message: str = "Saved transcript as {filename}."
    save_unavailable_message: str = (
        "Transcript download is unavailable in this browser."
    )
    audio_file_button_label: str = "Choose Audio File"
    audio_file_empty_label: str = "No file selected"
    copy_button_label: str = "Copy"
    copy_success_label: str = "Copied"
    record_button_label: str = "Record"
    stop_button_label: str = "Stop"
    timer_idle_label: str = "Idle"
    timer_ready_label: str = "Ready"
    timer_recording_label: str = "Recording"
    timer_processing_label: str = "Processing"
    progress_waiting_message: str = "Waiting for audio."
    progress_preparing_message: str = "Preparing audio..."
    progress_complete_message: str = "Complete"
    progress_transcribing_message: str = (
        "Transcribing audio locally, about {remaining} time left"
    )
    progress_uploading_message: str = "Uploading audio, {percent} complete"

    @field_validator("copy_feedback_ms")
    @classmethod
    def validate_copy_feedback_ms(cls, value: int) -> int:
        if value <= 0:
            msg = "copy_feedback_ms must be greater than 0"
            raise ValueError(msg)
        return value

    @field_validator(
        "progress_default_model_factor",
        "progress_default_audio_seconds",
        "progress_min_processing_seconds",
    )
    @classmethod
    def validate_positive_progress_float(cls, value: float) -> float:
        if value <= 0:
            msg = "progress timing values must be greater than 0"
            raise ValueError(msg)
        return value

    @field_validator(
        "progress_metadata_timeout_ms",
        "progress_update_interval_ms",
        "progress_min_visible_percent",
    )
    @classmethod
    def validate_positive_progress_int(cls, value: int) -> int:
        if value <= 0:
            msg = "progress integer values must be greater than 0"
            raise ValueError(msg)
        return value

    @field_validator(
        "progress_upload_fraction",
        "progress_transcription_fraction",
        "progress_max_transcription_fraction",
    )
    @classmethod
    def validate_progress_fraction(cls, value: float) -> float:
        if not 0 < value <= 1:
            msg = (
                "progress fractions must be greater than 0 and less than or equal to 1"
            )
            raise ValueError(msg)
        return value

    @field_validator("progress_model_factors")
    @classmethod
    def validate_progress_model_factors(
        cls,
        value: dict[str, float],
    ) -> dict[str, float]:
        if not value:
            msg = "progress_model_factors must include at least one item"
            raise ValueError(msg)
        for model_id, factor in value.items():
            if not model_id.strip():
                msg = "progress_model_factors keys must not be empty"
                raise ValueError(msg)
            if factor <= 0:
                msg = "progress_model_factors values must be greater than 0"
                raise ValueError(msg)
        return value

    @field_validator(
        "processing_status_message",
        "upload_status_message",
        "upload_ready_status_message",
        "empty_recording_message",
        "missing_upload_message",
        "ready_status_message",
        "recording_status_message",
        "completion_status_message",
        "processing_network_error_message",
        "microphone_blocked_message",
        "recording_start_error_message",
        "clipboard_unavailable_message",
        "save_success_message",
        "save_unavailable_message",
        "audio_file_button_label",
        "audio_file_empty_label",
        "copy_button_label",
        "copy_success_label",
        "record_button_label",
        "stop_button_label",
        "timer_idle_label",
        "timer_ready_label",
        "timer_recording_label",
        "timer_processing_label",
        "progress_waiting_message",
        "progress_preparing_message",
        "progress_complete_message",
        "progress_transcribing_message",
        "progress_uploading_message",
    )
    @classmethod
    def validate_ui_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            msg = "UI messages must not be empty"
            raise ValueError(msg)
        return cleaned


class AppConfig(BaseModel):
    server: ServerConfig
    transcription: TranscriptionConfig
    logging: LoggingConfig
    ui: UiConfig = Field(default_factory=UiConfig)


def load_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    config = AppConfig.model_validate(data)
    config_dir = path.resolve().parent
    config.transcription.cache_dir = _resolve_config_path(
        config_dir,
        config.transcription.cache_dir,
    )
    if config.transcription.vibevoice_repo_path is not None:
        config.transcription.vibevoice_repo_path = _resolve_config_path(
            config_dir,
            config.transcription.vibevoice_repo_path,
        )
    return config


def _resolve_config_path(config_dir: Path, value: Path) -> Path:
    expanded = value.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (config_dir / expanded).resolve()
