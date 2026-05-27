import asyncio

from app.middleware import UploadSizeLimitMiddleware


async def _run_middleware_request(
    *,
    body_parts: list[tuple[bytes, bool]],
    headers: list[tuple[bytes, bytes]] | None = None,
) -> list[dict[str, object]]:
    app_called = False

    async def downstream_app(scope, receive, send) -> None:  # type: ignore[no-untyped-def]
        nonlocal app_called
        app_called = True
        while True:
            message = await receive()
            if message["type"] == "http.request" and not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = UploadSizeLimitMiddleware(
        downstream_app,
        max_upload_size_bytes=3,
        paths={"/api/transcriptions"},
        error_message_factory=lambda scope: "Too much audio.",
    )
    pending_messages = [
        {
            "type": "http.request",
            "body": body,
            "more_body": more_body,
        }
        for body, more_body in body_parts
    ]
    sent_messages: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return pending_messages.pop(0)

    async def send(message: dict[str, object]) -> None:
        sent_messages.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/transcriptions",
        "headers": headers or [],
    }

    await middleware(scope, receive, send)

    assert app_called is True
    return sent_messages


def test_upload_limit_middleware_rejects_stream_without_content_length() -> None:
    sent_messages = asyncio.run(
        _run_middleware_request(
            body_parts=[
                (b"ab", True),
                (b"cd", False),
            ],
        )
    )

    assert sent_messages[0]["status"] == 413
    assert b"Too much audio." in sent_messages[1]["body"]


def test_upload_limit_middleware_rejects_content_length_before_downstream_app() -> None:
    app_called = False

    async def downstream_app(scope, receive, send) -> None:  # type: ignore[no-untyped-def]
        nonlocal app_called
        del scope, receive, send
        app_called = True

    middleware = UploadSizeLimitMiddleware(
        downstream_app,
        max_upload_size_bytes=3,
        paths={"/api/transcriptions"},
        error_message_factory=lambda scope: "Too much audio.",
    )
    sent_messages: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, object]) -> None:
        sent_messages.append(message)

    async def run_request() -> None:
        await middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/transcriptions",
                "headers": [(b"content-length", b"4")],
            },
            receive,
            send,
        )

    asyncio.run(run_request())

    assert app_called is False
    assert sent_messages[0]["status"] == 413
