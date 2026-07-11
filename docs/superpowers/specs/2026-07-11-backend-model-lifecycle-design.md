# Backend Model Lifecycle Design

## Problem

`mlx-whisper` retains its loaded model in a process-global `ModelHolder` after a
transcription. MLX may also retain allocations in its Metal cache. When the next
request switches to VibeVoice, PyTorch sees those unified-memory allocations as
memory owned by another framework. Loading and running the 7B VibeVoice model can
then exhaust MPS memory.

The application needs to keep same-backend requests efficient while ensuring
that models from different accelerator frameworks do not coexist after a backend
switch.

## Selected Approach

The transcription router will own backend lifecycle coordination. It will track
the active backend and serialize transcription across backends with a lifecycle
lock. Before dispatching a request to a different backend, it will release the
previous backend and then run the requested backend.

This preserves in-memory model reuse for consecutive requests using the same
backend. It intentionally prevents concurrent accelerator work across backends,
because switching while another request is using the previous backend could
release live model state and because both frameworks compete for the same Apple
unified memory.

## Service Contract

Each transcription backend will expose `release() -> None` in addition to
`transcribe(...)`.

`MlxWhisperService.release()` will:

- remove the model reference and path held by `mlx-whisper`'s process-global
  `ModelHolder`;
- clear the MLX Metal cache; and
- leave downloaded model files and the service's resolved local paths intact.

`VibeVoiceASRService.release()` will:

- remove all cached VibeVoice inference backend instances;
- run Python garbage collection after references have been removed; and
- clear the PyTorch MPS cache when MPS is available.

Release operations will be safe to call when no model is loaded. Cache cleanup
is a best-effort runtime operation, but programming errors will not be silently
hidden.

## Request Flow

For each transcription request, the router will:

1. Validate the model id and resolve its backend.
2. Acquire the router lifecycle lock.
3. If the selected backend differs from the active backend, call `release()` on
   the previous backend.
4. Mark the selected backend active and dispatch the transcription while still
   holding the lock.
5. Release the lock after success or failure.

The selected backend remains active after a failed transcription because it may
have loaded model state before failing. A later switch will still release it.
The previous backend is marked inactive only after its release succeeds, so a
cleanup failure does not falsely report a completed switch.

## Error Handling

Backend release failures will be surfaced as `TranscriptionError` with context
identifying the backend that could not be released. The router will not start the
new backend when cleanup of the previous backend fails, avoiding a likely
out-of-memory condition.

Existing validation, busy, and transcription errors remain unchanged. The API
continues mapping transcription failures to its existing error response.

## Testing

Focused unit tests will verify:

- switching Whisper to VibeVoice releases Whisper before VibeVoice runs;
- switching VibeVoice to Whisper releases VibeVoice before Whisper runs;
- repeated requests to the same backend reuse it without releasing;
- a failed transcription still leaves that backend eligible for release on the
  next switch;
- Whisper release clears its global holder and MLX cache;
- VibeVoice release drops cached instances and clears the MPS cache safely; and
- release failures prevent dispatch to the new backend and produce a clear
  transcription error.

The full Ruff and pytest quality suite will run after the focused tests pass.

## Documentation

The README will state that switching backend families unloads the previous
in-memory model to protect unified memory, and that switching back later incurs a
model reload. Downloaded model files remain cached on disk.
