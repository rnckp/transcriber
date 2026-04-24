from pathlib import Path

import logging

import pytest
from pydantic import ValidationError

from app.config import AppConfig, load_config
from app.logging import JsonFormatter, configure_logging


def test_load_config_reads_languages_and_model_sizes(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
server:
  host: "127.0.0.1"
  port: 8000
transcription:
  cache_dir: ".cache/models"
  max_upload_size_mb: 25
  upload_chunk_size_mb: 2
  default_upload_filename: "capture.webm"
  supported_languages:
    - code: "de"
      label: "German"
    - code: "en"
      label: "English"
  supported_model_sizes:
    - id: "tiny"
      label: "Tiny"
      model_name: "mlx-community/whisper-tiny-mlx"
    - id: "large"
      label: "Large"
      model_name: "mlx-community/whisper-large-v3-turbo"
logging:
  level: "INFO"
ui:
  copy_feedback_ms: 1200
  processing_status_message: "Working..."
  empty_recording_message: "Need more audio."
  save_success_message: "Saved {filename}."
  save_unavailable_message: "Download not available."
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert isinstance(config, AppConfig)
    assert [language.code for language in config.transcription.supported_languages] == [
        "de",
        "en",
    ]
    assert [model.id for model in config.transcription.supported_model_sizes] == [
        "tiny",
        "large",
    ]
    assert config.transcription.upload_chunk_size_mb == 2
    assert config.transcription.default_upload_filename == "capture.webm"
    assert config.ui.copy_feedback_ms == 1200
    assert config.ui.save_success_message == "Saved {filename}."
    assert config.ui.save_unavailable_message == "Download not available."


def test_load_config_rejects_chunk_size_larger_than_max(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
server:
  host: "127.0.0.1"
  port: 8000
transcription:
  cache_dir: ".cache/models"
  max_upload_size_mb: 1
  upload_chunk_size_mb: 2
  default_upload_filename: "recording.webm"
  supported_languages: []
  supported_model_sizes: []
logging:
  level: "INFO"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="upload_chunk_size_mb must be less than or equal to max_upload_size_mb"):
        load_config(config_file)


@pytest.mark.parametrize(
    ("yaml_text", "expected_loc"),
    [
        (
            """
server:
  host: "127.0.0.1"
  port: 0
transcription:
  cache_dir: ".cache/models"
  max_upload_size_mb: 25
  upload_chunk_size_mb: 1
  default_upload_filename: "recording.webm"
  supported_languages: []
  supported_model_sizes: []
logging:
  level: "INFO"
""".strip(),
            ("server", "port"),
        ),
        (
            """
server:
  host: "127.0.0.1"
  port: 8000
transcription:
  cache_dir: ".cache/models"
  max_upload_size_mb: 0
  upload_chunk_size_mb: 1
  default_upload_filename: "recording.webm"
  supported_languages: []
  supported_model_sizes: []
logging:
  level: "INFO"
""".strip(),
            ("transcription", "max_upload_size_mb"),
        ),
        (
            """
server:
  host: "127.0.0.1"
  port: 8000
transcription:
  cache_dir: ".cache/models"
  max_upload_size_mb: 25
  upload_chunk_size_mb: 0
  default_upload_filename: "recording.webm"
  supported_languages: []
  supported_model_sizes: []
logging:
  level: "INFO"
""".strip(),
            ("transcription", "upload_chunk_size_mb"),
        ),
        (
            """
server:
  host: "127.0.0.1"
  port: 8000
transcription:
  cache_dir: ".cache/models"
  max_upload_size_mb: 25
  upload_chunk_size_mb: 1
  default_upload_filename: "recording.webm"
  supported_languages: []
  supported_model_sizes: []
logging:
  level: "verbose"
""".strip(),
            ("logging", "level"),
        ),
    ],
)
def test_load_config_rejects_invalid_values(
    tmp_path: Path,
    yaml_text: str,
    expected_loc: tuple[str, str],
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValidationError) as exc_info:
        load_config(config_file)

    assert exc_info.value.errors()[0]["loc"] == expected_loc


def test_load_config_rejects_malformed_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
server:
  host: "127.0.0.1"
  port: 8000
logging:
  level: "INFO"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError) as exc_info:
        load_config(config_file)

    assert exc_info.value.errors()[0]["loc"] == ("transcription",)


def test_configure_logging_normalizes_level_and_sets_json_formatter() -> None:
    configure_logging("warning")

    root_logger = logging.getLogger()
    assert root_logger.level == logging.WARNING
    assert len(root_logger.handlers) == 1
    assert isinstance(root_logger.handlers[0].formatter, JsonFormatter)


def test_configure_logging_falls_back_for_unexpected_input() -> None:
    configure_logging("not-a-real-level")

    assert logging.getLogger().level == logging.INFO
