from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

LoggingLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class SupportedLanguage(BaseModel):
    code: str
    label: str


class SupportedModelSize(BaseModel):
    id: str
    label: str
    model_name: str


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
            msg = "upload_chunk_size_mb must be less than or equal to max_upload_size_mb"
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


class LoggingConfig(BaseModel):
    level: LoggingLevel = "INFO"


class UiConfig(BaseModel):
    copy_feedback_ms: int = 1400
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
    microphone_blocked_message: str = (
        "Microphone access is blocked. Allow access in your browser settings and try again."
    )
    recording_start_error_message: str = (
        "Recording could not start. Check your microphone and try again."
    )
    clipboard_unavailable_message: str = (
        "Clipboard access is unavailable in this browser."
    )
    save_success_message: str = "Saved transcript as {filename}."
    save_unavailable_message: str = "Transcript download is unavailable in this browser."
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

    @field_validator("copy_feedback_ms")
    @classmethod
    def validate_copy_feedback_ms(cls, value: int) -> int:
        if value <= 0:
            msg = "copy_feedback_ms must be greater than 0"
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
    return AppConfig.model_validate(data)
