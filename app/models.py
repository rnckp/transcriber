from pydantic import BaseModel, Field


class ApiError(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ApiError


class TranscriptionSegmentResponse(BaseModel):
    speaker: str
    text: str
    start_time: float | None = None
    end_time: float | None = None


class TranscriptionResponse(BaseModel):
    transcript: str
    language: str
    model_size: str
    duration_seconds: float | None = None
    segments: list[TranscriptionSegmentResponse] = Field(default_factory=list)
