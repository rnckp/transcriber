from pydantic import BaseModel


class ApiError(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ApiError


class TranscriptionResponse(BaseModel):
    transcript: str
    language: str
    model_size: str
    duration_seconds: float | None = None
