import json
from collections.abc import Awaitable, Callable, Collection
from typing import Any

Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]
MaxUploadSizeProvider = int | Callable[[Scope], int]
ErrorMessageFactory = Callable[[Scope], str]


class UploadTooLargeDuringReceiveError(Exception):
    """Raised when a streamed request body exceeds the configured limit."""


class UploadSizeLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        max_upload_size_bytes: MaxUploadSizeProvider,
        paths: Collection[str],
        error_message_factory: ErrorMessageFactory,
    ) -> None:
        self.app = app
        self._max_upload_size_bytes = max_upload_size_bytes
        self._paths = set(paths)
        self._error_message_factory = error_message_factory

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or scope.get("path") not in self._paths:
            await self.app(scope, receive, send)
            return

        max_upload_size_bytes = self._resolve_max_upload_size_bytes(scope)
        content_length = self._content_length(scope)
        if content_length is not None and content_length > max_upload_size_bytes:
            await self._send_upload_too_large(scope, send)
            return

        bytes_received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal bytes_received
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                if isinstance(body, bytes):
                    bytes_received += len(body)
                if bytes_received > max_upload_size_bytes:
                    raise UploadTooLargeDuringReceiveError
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except UploadTooLargeDuringReceiveError:
            if not response_started:
                await self._send_upload_too_large(scope, send)

    def _resolve_max_upload_size_bytes(self, scope: Scope) -> int:
        if callable(self._max_upload_size_bytes):
            return self._max_upload_size_bytes(scope)
        return self._max_upload_size_bytes

    def _content_length(self, scope: Scope) -> int | None:
        for name, value in scope.get("headers", []):
            if name.lower() != b"content-length":
                continue
            try:
                return int(value.decode("ascii"))
            except ValueError:
                return None
        return None

    async def _send_upload_too_large(self, scope: Scope, send: Send) -> None:
        body = json.dumps(
            {
                "error": {
                    "code": "upload_too_large",
                    "message": self._error_message_factory(scope),
                }
            }
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
