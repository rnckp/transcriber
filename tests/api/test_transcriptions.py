from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.config import SupportedLanguage, SupportedModelSize, load_config
from app.main import app
from app.services.transcription import TranscriptionError, TranscriptionResult
from tests.stubs import StubService


class FailingService:
    def transcribe(
        self, audio_path: Path, language: str, model_size: str
    ) -> TranscriptionResult:
        raise TranscriptionError("Failed to transcribe audio: backend unavailable")


def test_post_transcriptions_rejects_unknown_model_size() -> None:
    app.state.config = load_config(Path("config.yaml"))
    app.state.service = StubService()
    client = TestClient(app)

    response = client.post(
        "/api/transcriptions",
        data={"language": "de", "model_size": "giant"},
        files={"audio": ("sample.webm", b"audio", "audio/webm")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_model_size"


def test_post_transcriptions_returns_transcript(client: TestClient) -> None:
    response = client.post(
        "/api/transcriptions",
        data={"language": "de", "model_size": "small"},
        files={"audio": ("sample.webm", b"audio", "audio/webm")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "transcript": "Hallo Welt",
        "language": "de",
        "model_size": "small",
        "duration_seconds": 1.2,
    }


def test_post_transcriptions_uses_threadpool_for_transcription(
    client: TestClient, stub_service: StubService
) -> None:
    mocked_result = TranscriptionResult(
        transcript="Threadpooled",
        language="de",
        model_size="small",
        duration_seconds=2.4,
    )

    with patch(
        "app.api.transcriptions.run_in_threadpool",
        new=AsyncMock(return_value=mocked_result),
    ) as mocked_threadpool:
        response = client.post(
            "/api/transcriptions",
            data={"language": "de", "model_size": "small"},
            files={"audio": ("sample.webm", b"audio", "audio/webm")},
        )

    assert response.status_code == 200
    assert response.json()["transcript"] == "Threadpooled"
    service_call = None
    for awaited_call in mocked_threadpool.await_args_list:
        callback, *args = awaited_call.args
        if getattr(callback, "__self__", None) is stub_service:
            service_call = (callback, *args)
            break

    assert service_call is not None
    callback, audio_path, language, model_size = service_call
    assert callback.__self__ is stub_service
    assert callback.__func__.__name__ == "transcribe"
    assert audio_path.suffix == ".webm"
    assert language == "de"
    assert model_size == "small"


def test_post_transcriptions_rejects_oversized_upload() -> None:
    config = load_config(Path("config.yaml"))
    config.transcription.max_upload_size_mb = 1
    config.transcription.upload_chunk_size_mb = 1
    app.state.config = config
    stub_service = StubService()
    app.state.service = stub_service
    client = TestClient(app)

    response = client.post(
        "/api/transcriptions",
        data={"language": "de", "model_size": "small"},
        files={"audio": ("sample.webm", b"a" * (1024 * 1024 + 1), "audio/webm")},
    )

    assert response.status_code == 413
    assert response.json() == {
        "error": {
            "code": "upload_too_large",
            "message": "Upload exceeds the configured 1 MB limit.",
        }
    }
    assert stub_service.calls == []


def test_post_transcriptions_returns_structured_error_for_transcription_failures() -> (
    None
):
    app.state.config = load_config(Path("config.yaml"))
    app.state.service = FailingService()
    client = TestClient(app)

    response = client.post(
        "/api/transcriptions",
        data={"language": "de", "model_size": "small"},
        files={"audio": ("sample.webm", b"audio", "audio/webm")},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "transcription_failed",
            "message": "Failed to transcribe audio: backend unavailable",
        }
    }


def test_post_transcriptions_returns_plain_language_error_for_missing_audio(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/transcriptions",
        data={"language": "de", "model_size": "small"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "invalid_request",
            "message": "Provide an audio recording, language, and model size before transcribing.",
        }
    }


def test_post_transcriptions_uses_configured_language_labels_in_error_message() -> None:
    config = load_config(Path("config.yaml"))
    config.transcription.supported_languages = [
        SupportedLanguage(code="fr", label="French"),
        SupportedLanguage(code="it", label="Italian"),
    ]
    app.state.config = config
    app.state.service = StubService()
    client = TestClient(app)

    response = client.post(
        "/api/transcriptions",
        data={"language": "de", "model_size": "small"},
        files={"audio": ("sample.webm", b"audio", "audio/webm")},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "unsupported_language",
            "message": "Choose French or Italian.",
        }
    }


def test_index_renders_languages_and_models_from_app_state_config() -> None:
    config = load_config(Path("config.yaml"))
    config.ui.ready_status_message = "</script><script>window.injected=true</script>"
    config.transcription.supported_languages = [
        SupportedLanguage(code="fr", label="French"),
    ]
    config.transcription.supported_model_sizes = [
        SupportedModelSize(
            id="custom",
            label="Custom",
            model_name="mlx-community/whisper-custom",
        )
    ]
    app.state.config = config
    app.state.service = StubService()
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert '<option value="fr">French</option>' in response.text
    assert '<option value="custom">Custom</option>' in response.text
    assert '<option value="de">German</option>' not in response.text
    assert "record or upload audio to transcribe locally" in response.text
    assert "Copy or save the finished transcript as a text file." in response.text
    assert "Ready to record or upload audio." in response.text
    assert "Your transcript appears here after you record or upload audio." in response.text
    assert 'defaultUploadFilename' in response.text
    assert 'readyStatusMessage' in response.text
    assert '<\\/script><script>window.injected=true<\\/script>' in response.text
    assert 'saveSuccessMessage' in response.text
    assert 'id="audio-file"' in response.text
    assert 'id="upload-button"' in response.text
    assert 'id="save-button"' in response.text
