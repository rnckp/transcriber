import json
from html import escape
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.transcriptions import router as transcriptions_router
from app.config import load_config
from app.logging import configure_logging
from app.models import ApiError, ErrorResponse
from app.services.mlx_whisper import MlxWhisperService

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
CONFIG_PATH = BASE_DIR.parent / "config.yaml"
INDEX_TEMPLATE = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

config = load_config(CONFIG_PATH)
configure_logging(config.logging.level)
service = MlxWhisperService(config.transcription)
app = FastAPI(title="Transcriber")
app.state.config = config
app.state.service = service


app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")
app.include_router(transcriptions_router)


def _render_select_options(options: list[tuple[str, str]]) -> str:
    return "".join(
        f'<option value="{escape(value)}">{escape(label)}</option>'
        for value, label in options
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=ApiError(
                code="invalid_request",
                message="Provide an audio recording, language, and model size before transcribing.",
            )
        ).model_dump(),
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    runtime_config = request.app.state.config
    ui_config = json.dumps(
        {
            "copyFeedbackMs": runtime_config.ui.copy_feedback_ms,
            "processingStatusMessage": runtime_config.ui.processing_status_message,
            "uploadStatusMessage": runtime_config.ui.upload_status_message,
            "uploadReadyStatusMessage": runtime_config.ui.upload_ready_status_message,
            "emptyRecordingMessage": runtime_config.ui.empty_recording_message,
            "missingUploadMessage": runtime_config.ui.missing_upload_message,
            "readyStatusMessage": runtime_config.ui.ready_status_message,
            "recordingStatusMessage": runtime_config.ui.recording_status_message,
            "completionStatusMessage": runtime_config.ui.completion_status_message,
            "microphoneBlockedMessage": runtime_config.ui.microphone_blocked_message,
            "recordingStartErrorMessage": runtime_config.ui.recording_start_error_message,
            "clipboardUnavailableMessage": runtime_config.ui.clipboard_unavailable_message,
            "saveSuccessMessage": runtime_config.ui.save_success_message,
            "saveUnavailableMessage": runtime_config.ui.save_unavailable_message,
            "audioFileButtonLabel": runtime_config.ui.audio_file_button_label,
            "audioFileEmptyLabel": runtime_config.ui.audio_file_empty_label,
            "copyButtonLabel": runtime_config.ui.copy_button_label,
            "copySuccessLabel": runtime_config.ui.copy_success_label,
            "recordButtonLabel": runtime_config.ui.record_button_label,
            "stopButtonLabel": runtime_config.ui.stop_button_label,
            "timerIdleLabel": runtime_config.ui.timer_idle_label,
            "timerReadyLabel": runtime_config.ui.timer_ready_label,
            "timerRecordingLabel": runtime_config.ui.timer_recording_label,
            "timerProcessingLabel": runtime_config.ui.timer_processing_label,
            "defaultUploadFilename": runtime_config.transcription.default_upload_filename,
        }
    ).replace("</", "<\\/")
    language_options = _render_select_options(
        [
            (language.code, language.label)
            for language in runtime_config.transcription.supported_languages
        ]
    )
    model_options = _render_select_options(
        [
            (model.id, model.label)
            for model in runtime_config.transcription.supported_model_sizes
        ]
    )
    html = INDEX_TEMPLATE
    html = html.replace("<!--LANGUAGE_OPTIONS-->", language_options)
    html = html.replace("<!--MODEL_OPTIONS-->", model_options)
    html = html.replace('"__APP_CONFIG__"', ui_config)
    return HTMLResponse(html)
