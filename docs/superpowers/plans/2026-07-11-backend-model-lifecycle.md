# Backend Model Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically unload the active transcription backend when switching between MLX Whisper and PyTorch VibeVoice so their models do not compete for Apple unified memory.

**Architecture:** Extend the transcription backend protocol with an explicit `release()` operation. Each backend owns cleanup of its framework-specific retained state, while the router owns serialized switching, active-backend tracking, release ordering, and cleanup error propagation.

**Tech Stack:** Python 3.13, FastAPI service layer, `mlx-whisper`/MLX Metal, PyTorch MPS, pytest, Ruff, uv.

## Global Constraints

- Consecutive requests using the same backend must retain and reuse the loaded model.
- Switching backend families must release the previous backend before dispatching the new request.
- The router must serialize transcription across backend families so no live request is released concurrently.
- Downloaded model files and resolved local model paths must remain cached on disk.
- No dependency or configuration changes.
- Use `uv run` for all Python, pytest, and Ruff commands.

## File Structure

- `app/services/transcription.py`: shared transcription backend protocol.
- `app/services/mlx_whisper.py`: MLX Whisper ownership and MLX cache cleanup.
- `app/services/vibevoice_asr.py`: VibeVoice backend ownership and PyTorch MPS cleanup.
- `app/services/transcription_router.py`: active-backend lifecycle coordination.
- `tests/services/test_mlx_whisper.py`: focused MLX release regression test.
- `tests/services/test_vibevoice_asr.py`: focused VibeVoice release regression tests.
- `tests/services/test_transcription_router.py`: switch ordering, reuse, failure, and serialization tests.
- `tests/stubs.py`: protocol-compatible test backends.
- `README.md`: user-visible reload and disk-cache behavior.
- `NOTES.md`: durable root-cause and lifecycle constraint.

---

### Task 1: Framework-Specific Backend Release Operations

**Files:**
- Modify: `app/services/transcription.py`
- Modify: `app/services/mlx_whisper.py`
- Modify: `app/services/vibevoice_asr.py`
- Modify: `tests/services/test_mlx_whisper.py`
- Modify: `tests/services/test_vibevoice_asr.py`

**Interfaces:**
- Consumes: existing `MlxWhisperService` and `VibeVoiceASRService` model caches.
- Produces: `TranscriptionService.release() -> None`, `MlxWhisperService.release() -> None`, and `VibeVoiceASRService.release() -> None`.

- [ ] **Step 1: Write the failing MLX release test**

Add imports and this test to `tests/services/test_mlx_whisper.py`:

```python
from mlx_whisper.transcribe import ModelHolder


def test_release_drops_loaded_whisper_model_and_clears_mlx_cache(
    tmp_path: Path,
) -> None:
    service = MlxWhisperService(TranscriptionConfig(cache_dir=tmp_path / "models"))
    retained_model = object()
    ModelHolder.model = retained_model
    ModelHolder.model_path = "retained-model"

    with patch("app.services.mlx_whisper.mx.clear_cache") as clear_cache:
        service.release()

    assert ModelHolder.model is None
    assert ModelHolder.model_path is None
    clear_cache.assert_called_once_with()
```

- [ ] **Step 2: Run the MLX test and verify RED**

Run: `UV_CACHE_DIR=/tmp/transcriber-uv-cache uv run --offline pytest tests/services/test_mlx_whisper.py::test_release_drops_loaded_whisper_model_and_clears_mlx_cache -v`

Expected: FAIL with `AttributeError: 'MlxWhisperService' object has no attribute 'release'`.

- [ ] **Step 3: Add the lifecycle protocol and minimal MLX cleanup**

Append the method to `TranscriptionService` in `app/services/transcription.py`:

```python
    def release(self) -> None: ...
```

Add imports to `app/services/mlx_whisper.py`:

```python
import mlx.core as mx
from mlx_whisper.transcribe import ModelHolder
```

Add this method to `MlxWhisperService`:

```python
    def release(self) -> None:
        ModelHolder.model = None
        ModelHolder.model_path = None
        mx.clear_cache()
```

Do not clear `self._models`; it contains disk paths rather than loaded weights.

- [ ] **Step 4: Run the MLX test and verify GREEN**

Run: `UV_CACHE_DIR=/tmp/transcriber-uv-cache uv run --offline pytest tests/services/test_mlx_whisper.py::test_release_drops_loaded_whisper_model_and_clears_mlx_cache -v`

Expected: PASS.

- [ ] **Step 5: Write failing VibeVoice release tests**

Add `import gc` to the implementation later, but first add these tests to `tests/services/test_vibevoice_asr.py`:

```python
def test_release_drops_vibevoice_backends_and_clears_available_mps_cache() -> None:
    service = VibeVoiceASRService(TranscriptionConfig())
    service._backends["vibevoice-7b"] = object()

    with (
        patch("gc.collect") as collect,
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.mps.empty_cache") as empty_cache,
    ):
        service.release()

    assert service._backends == {}
    collect.assert_called_once_with()
    empty_cache.assert_called_once_with()


def test_release_skips_mps_cache_when_mps_is_unavailable() -> None:
    service = VibeVoiceASRService(TranscriptionConfig())

    with (
        patch("gc.collect") as collect,
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.mps.empty_cache") as empty_cache,
    ):
        service.release()

    collect.assert_called_once_with()
    empty_cache.assert_not_called()
```

- [ ] **Step 6: Run the VibeVoice tests and verify RED**

Run: `UV_CACHE_DIR=/tmp/transcriber-uv-cache uv run --offline pytest tests/services/test_vibevoice_asr.py -k release -v`

Expected: FAIL with `AttributeError: 'VibeVoiceASRService' object has no attribute 'release'`.

- [ ] **Step 7: Implement minimal VibeVoice cleanup**

Add `import gc` to `app/services/vibevoice_asr.py`, then add:

```python
    def release(self) -> None:
        import torch

        with self._backend_lock:
            self._backends.clear()
        gc.collect()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
```

- [ ] **Step 8: Run focused backend tests and verify GREEN**

Run: `UV_CACHE_DIR=/tmp/transcriber-uv-cache uv run --offline pytest tests/services/test_mlx_whisper.py tests/services/test_vibevoice_asr.py -v`

Expected: all tests PASS.

- [ ] **Step 9: Commit backend release operations**

```bash
git add app/services/transcription.py app/services/mlx_whisper.py app/services/vibevoice_asr.py tests/services/test_mlx_whisper.py tests/services/test_vibevoice_asr.py
git commit -m "fix(transcription): release accelerator model caches"
```

---

### Task 2: Router-Managed Switching and Serialization

**Files:**
- Modify: `app/services/transcription_router.py`
- Modify: `tests/services/test_transcription_router.py`
- Modify: `tests/stubs.py`

**Interfaces:**
- Consumes: `TranscriptionService.release() -> None` from Task 1.
- Produces: router switching that releases the prior backend before dispatch and wraps release failures in `TranscriptionError`.

- [ ] **Step 1: Make router test doubles lifecycle-aware**

Update `RecordingBackend` in `tests/services/test_transcription_router.py`:

```python
class RecordingBackend:
    def __init__(self, transcript: str, events: list[str] | None = None) -> None:
        self.transcript = transcript
        self.calls: list[tuple[Path, str, str]] = []
        self.release_count = 0
        self.events = events

    def release(self) -> None:
        self.release_count += 1
        if self.events is not None:
            self.events.append(f"release:{self.transcript}")
```

Keep its existing `transcribe()` behavior and append
`self.events.append(f"transcribe:{self.transcript}")` when `events` is not
`None`.

Add no-op `release()` methods to both protocol stubs in `tests/stubs.py`:

```python
    def release(self) -> None:
        pass
```

- [ ] **Step 2: Write failing switch-order and same-backend reuse tests**

Add this helper using the same two-model configuration as the existing dispatch
test, then use it from that test and the new tests:

```python
def _build_router(
    whisper: RecordingBackend,
    vibevoice: RecordingBackend,
) -> TranscriptionRouterService:
    return TranscriptionRouterService(
        config=TranscriptionConfig(
            supported_languages=[SupportedLanguage(code="de", label="German")],
            supported_model_sizes=[
                SupportedModelSize(
                    id="small",
                    label="Small",
                    model_name="mlx-community/whisper-small-mlx",
                ),
                SupportedModelSize(
                    id="vibevoice-7b",
                    label="VibeVoice ASR 7B",
                    model_name="microsoft/VibeVoice-ASR",
                    backend="vibevoice_asr",
                ),
            ],
        ),
        backends={"mlx_whisper": whisper, "vibevoice_asr": vibevoice},
    )
```

Then add:

```python
def test_router_releases_previous_backend_before_switch(tmp_path: Path) -> None:
    events: list[str] = []
    whisper = RecordingBackend("whisper", events)
    vibevoice = RecordingBackend("vibevoice", events)
    router = _build_router(whisper, vibevoice)

    router.transcribe(tmp_path / "first.webm", "de", "small")
    router.transcribe(tmp_path / "second.webm", "de", "vibevoice-7b")
    router.transcribe(tmp_path / "third.webm", "de", "small")

    assert events == [
        "transcribe:whisper",
        "release:whisper",
        "transcribe:vibevoice",
        "release:vibevoice",
        "transcribe:whisper",
    ]


def test_router_reuses_same_backend_without_release(tmp_path: Path) -> None:
    whisper = RecordingBackend("whisper")
    router = _build_router(whisper, RecordingBackend("vibevoice"))

    router.transcribe(tmp_path / "first.webm", "de", "small")
    router.transcribe(tmp_path / "second.webm", "de", "small")

    assert whisper.release_count == 0
```

- [ ] **Step 3: Run switch tests and verify RED**

Run: `UV_CACHE_DIR=/tmp/transcriber-uv-cache uv run --offline pytest tests/services/test_transcription_router.py -k "releases_previous or reuses_same" -v`

Expected: switch-order test FAIL because no release event occurs.

- [ ] **Step 4: Implement minimal active-backend switching**

Import `Lock` and `TranscriptionError` in `app/services/transcription_router.py`.
Initialize lifecycle state in `__init__`:

```python
        self._active_backend_name: str | None = None
        self._lifecycle_lock = Lock()
```

Replace the final dispatch with:

```python
        with self._lifecycle_lock:
            if (
                self._active_backend_name is not None
                and self._active_backend_name != backend_name
            ):
                previous_backend = self._backends[self._active_backend_name]
                try:
                    previous_backend.release()
                except Exception as exc:
                    raise TranscriptionError(
                        "Failed to release transcription backend "
                        f"{self._active_backend_name}: {exc}"
                    ) from exc

            self._active_backend_name = backend_name
            return backend.transcribe(audio_path, language, model_size)
```

- [ ] **Step 5: Run router tests and verify GREEN**

Run: `UV_CACHE_DIR=/tmp/transcriber-uv-cache uv run --offline pytest tests/services/test_transcription_router.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Write failing release-error and failed-transcription lifecycle tests**

Add `TranscriptionError` to the test imports, then add:

```python
def test_router_does_not_dispatch_new_backend_when_release_fails(
    tmp_path: Path,
) -> None:
    whisper = RecordingBackend("whisper")
    vibevoice = RecordingBackend("vibevoice")
    router = _build_router(whisper, vibevoice)
    router.transcribe(tmp_path / "first.webm", "de", "small")
    whisper.release = MagicMock(side_effect=RuntimeError("cleanup failed"))

    with pytest.raises(
        TranscriptionError,
        match="Failed to release transcription backend mlx_whisper: cleanup failed",
    ):
        router.transcribe(tmp_path / "second.webm", "de", "vibevoice-7b")

    assert vibevoice.calls == []


def test_router_releases_failed_backend_on_later_switch(tmp_path: Path) -> None:
    whisper = RecordingBackend("whisper")
    vibevoice = RecordingBackend("vibevoice")
    whisper.transcribe = MagicMock(side_effect=TranscriptionError("decode failed"))
    router = _build_router(whisper, vibevoice)

    with pytest.raises(TranscriptionError, match="decode failed"):
        router.transcribe(tmp_path / "first.webm", "de", "small")
    router.transcribe(tmp_path / "second.webm", "de", "vibevoice-7b")

    assert whisper.release_count == 1
```

- [ ] **Step 7: Run lifecycle error tests**

Run: `UV_CACHE_DIR=/tmp/transcriber-uv-cache uv run --offline pytest tests/services/test_transcription_router.py -k "release_fails or failed_backend" -v`

Expected: PASS with the Task 2 implementation; if either fails, correct only router active-state ordering and rerun until both pass.

- [ ] **Step 8: Write and verify the cross-backend serialization test**

Add `ThreadPoolExecutor` and `Event` imports, then add:

```python
def test_router_serializes_transcription_across_backend_switches(
    tmp_path: Path,
) -> None:
    first_started = Event()
    release_first = Event()
    second_attempted = Event()
    whisper = RecordingBackend("whisper")
    vibevoice = RecordingBackend("vibevoice")
    original_transcribe = whisper.transcribe

    def blocking_transcribe(
        audio_path: Path,
        language: str,
        model_size: str,
    ) -> TranscriptionResult:
        first_started.set()
        assert release_first.wait(timeout=2)
        return original_transcribe(audio_path, language, model_size)

    whisper.transcribe = blocking_transcribe  # type: ignore[method-assign]
    router = _build_router(whisper, vibevoice)

    def run_vibevoice() -> TranscriptionResult:
        second_attempted.set()
        return router.transcribe(
            tmp_path / "second.webm",
            "de",
            "vibevoice-7b",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            router.transcribe,
            tmp_path / "first.webm",
            "de",
            "small",
        )
        assert first_started.wait(timeout=2)
        second = executor.submit(run_vibevoice)
        assert second_attempted.wait(timeout=2)
        assert vibevoice.calls == []

        release_first.set()
        assert first.result(timeout=2).transcript == "whisper"
        assert second.result(timeout=2).transcript == "vibevoice"
```

Run: `UV_CACHE_DIR=/tmp/transcriber-uv-cache uv run --offline pytest tests/services/test_transcription_router.py -k serializes -v`

Expected: PASS, demonstrating the lifecycle lock is held for the complete transcription call.

- [ ] **Step 9: Commit router lifecycle coordination**

```bash
git add app/services/transcription_router.py tests/services/test_transcription_router.py tests/stubs.py
git commit -m "fix(transcription): unload models when switching backends"
```

---

### Task 3: Document the Memory Lifecycle and Verify the Repository

**Files:**
- Modify: `README.md`
- Modify: `NOTES.md`

**Interfaces:**
- Consumes: completed automatic backend switching behavior from Task 2.
- Produces: user-facing performance expectation and durable maintainer context.

- [ ] **Step 1: Update README model-cache behavior**

Replace the first sentence after the Models list with:

```markdown
Models download on first use and remain cached on disk. Consecutive runs with the same backend reuse the loaded model in memory. Switching between MLX Whisper and VibeVoice automatically unloads the previous backend to protect Apple unified memory, so the first run after switching back must reload that model. VibeVoice also downloads its Hugging Face model on first real use and can require substantial memory; keep `transcription.vibevoice_max_new_tokens` conservative unless you know the target machine can handle more.
```

- [ ] **Step 2: Record the root cause in NOTES**

Append this bullet to `NOTES.md`:

```markdown
- `mlx-whisper` retains its loaded model in `transcribe.ModelHolder`, and MLX may retain Metal cache allocations. On Apple unified memory, PyTorch reports those allocations as “other allocations”; the router must release the previous framework before switching between MLX Whisper and VibeVoice, while same-backend requests may retain their model.
```

- [ ] **Step 3: Run formatting and lint verification**

Run: `UV_CACHE_DIR=/tmp/transcriber-uv-cache uv run --offline ruff format .`

Expected: formatter completes successfully.

Run: `UV_CACHE_DIR=/tmp/transcriber-uv-cache uv run --offline ruff check .`

Expected: `All checks passed!`

- [ ] **Step 4: Run the full test suite**

Run: `UV_CACHE_DIR=/tmp/transcriber-uv-cache uv run --offline pytest -v`

Expected: all tests PASS with no unexpected errors or warnings.

- [ ] **Step 5: Review the final diff and commit documentation**

Run: `git diff --check && git status --short && git diff --stat`

Expected: no whitespace errors; only the files listed in this plan are modified.

```bash
git add README.md NOTES.md
git commit -m "docs(transcription): explain backend model unloading"
```
