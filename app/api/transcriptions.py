import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.models import (
    ApiError,
    ErrorResponse,
    TranscriptionResponse,
    TranscriptionSegmentResponse,
)
from app.services.transcription import (
    TranscriptionError,
    UnsupportedLanguageError,
    UnsupportedModelSizeError,
)

router = APIRouter(prefix="/api/transcriptions", tags=["transcriptions"])
_BYTES_PER_MEGABYTE = 1024 * 1024


class UploadTooLargeError(Exception):
    """Raised when the uploaded file exceeds the configured size limit."""


def _error_response(
    code: str,
    message: str,
    status_code: int = status.HTTP_400_BAD_REQUEST,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=ApiError(code=code, message=message),
        ).model_dump(),
    )


def _build_supported_choice_message(labels: list[str], fallback: str) -> str:
    if not labels:
        return fallback
    if len(labels) == 1:
        return f"Choose {labels[0]}."
    if len(labels) == 2:
        return f"Choose {labels[0]} or {labels[1]}."
    return f"Choose {', '.join(labels[:-1])}, or {labels[-1]}."


async def _write_upload_to_temp_file(
    audio: UploadFile,
    suffix: str,
    max_upload_size_bytes: int,
    chunk_size_bytes: int,
) -> Path:
    temp_path: Path | None = None
    bytes_written = 0

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            temp_path = Path(handle.name)

            while chunk := await audio.read(chunk_size_bytes):
                bytes_written += len(chunk)
                if bytes_written > max_upload_size_bytes:
                    raise UploadTooLargeError
                await run_in_threadpool(handle.write, chunk)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise

    return temp_path


@router.post(
    "",
    response_model=TranscriptionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def create_transcription(
    request: Request,
    audio: Annotated[UploadFile, File(...)],
    language: Annotated[str, Form(...)],
    model_size: Annotated[str, Form(...)],
) -> TranscriptionResponse:
    config = request.app.state.config
    service = request.app.state.service

    supported_languages = {
        item.code for item in config.transcription.supported_languages
    }
    supported_language_labels = [
        item.label for item in config.transcription.supported_languages
    ]
    supported_models = {item.id for item in config.transcription.supported_model_sizes}
    supported_model_labels = [
        item.label for item in config.transcription.supported_model_sizes
    ]

    if language not in supported_languages:
        return _error_response(
            code="unsupported_language",
            message=_build_supported_choice_message(
                supported_language_labels,
                "Choose a supported language.",
            ),
        )

    if model_size not in supported_models:
        return _error_response(
            code="unsupported_model_size",
            message=_build_supported_choice_message(
                supported_model_labels,
                "Choose a supported model size.",
            ),
        )

    suffix = (
        Path(audio.filename or config.transcription.default_upload_filename).suffix
        or Path(config.transcription.default_upload_filename).suffix
        or ".webm"
    )
    temp_path: Path | None = None
    max_upload_size_bytes = (
        config.transcription.max_upload_size_mb * _BYTES_PER_MEGABYTE
    )
    chunk_size_bytes = config.transcription.upload_chunk_size_mb * _BYTES_PER_MEGABYTE

    try:
        temp_path = await _write_upload_to_temp_file(
            audio,
            suffix,
            max_upload_size_bytes,
            chunk_size_bytes,
        )
        result = await run_in_threadpool(
            service.transcribe,
            temp_path,
            language,
            model_size,
        )
    except UploadTooLargeError:
        return _error_response(
            code="upload_too_large",
            message=(
                "Upload exceeds the configured "
                f"{config.transcription.max_upload_size_mb} MB limit."
            ),
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        )
    except UnsupportedLanguageError as exc:
        return _error_response("unsupported_language", str(exc))
    except UnsupportedModelSizeError as exc:
        return _error_response("unsupported_model_size", str(exc))
    except TranscriptionError as exc:
        return _error_response(
            code="transcription_failed",
            message=str(exc),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    finally:
        await audio.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return TranscriptionResponse(
        transcript=result.transcript,
        language=result.language,
        model_size=result.model_size,
        duration_seconds=result.duration_seconds,
        segments=[
            TranscriptionSegmentResponse(
                speaker=segment.speaker,
                text=segment.text,
                start_time=segment.start_time,
                end_time=segment.end_time,
            )
            for segment in (result.segments or [])
        ],
    )
