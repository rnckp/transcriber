const state = {
  mediaRecorder: null,
  chunks: [],
  stream: null,
  timerId: null,
  startedAt: null,
  isRecording: false,
  isProcessing: false,
  copyFeedbackId: null,
  lastResult: null,
};

const STORAGE_KEYS = {
  language: "transcriber.language",
  modelSize: "transcriber.modelSize",
};

const APP_CONFIG = Object.freeze(window.APP_CONFIG ?? {});

const elements = {
  language: document.querySelector("#language"),
  modelSize: document.querySelector("#model-size"),
  audioFile: document.querySelector("#audio-file"),
  audioFileButton: document.querySelector("#audio-file-button"),
  audioFileName: document.querySelector("#audio-file-name"),
  recordButton: document.querySelector("#record-button"),
  uploadButton: document.querySelector("#upload-button"),
  resetButton: document.querySelector("#reset-button"),
  copyButton: document.querySelector("#copy-button"),
  saveButton: document.querySelector("#save-button"),
  status: document.querySelector("#status"),
  timer: document.querySelector("#timer"),
  timerCard: document.querySelector("#timer-card"),
  timerState: document.querySelector("#timer-state"),
  transcript: document.querySelector("#transcript"),
  resultMeta: document.querySelector("#result-meta"),
  resultLanguage: document.querySelector("#result-language"),
  resultModel: document.querySelector("#result-model"),
  resultDuration: document.querySelector("#result-duration"),
};

function setStatus(message) {
  elements.status.textContent = message;
}

function setTimerState(label, className = "") {
  elements.timerState.textContent = label;
  elements.timerCard.classList.toggle("is-recording", className === "recording");
  elements.timerCard.classList.toggle("is-busy", className === "busy");
}

function setSelectorsDisabled(disabled) {
  elements.language.disabled = disabled;
  elements.modelSize.disabled = disabled;
}

function clearActiveRecorder({ cancelPendingStop = false } = {}) {
  if (state.mediaRecorder) {
    if (cancelPendingStop) {
      state.mediaRecorder.ondataavailable = null;
      state.mediaRecorder.onstop = null;
    }
    if (state.mediaRecorder.state !== "inactive") {
      state.mediaRecorder.stop();
    }
    state.mediaRecorder = null;
  }

  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
}

function hasSelectedUpload() {
  return Boolean(elements.audioFile.files && elements.audioFile.files.length > 0);
}

function syncUploadControls() {
  const disabled = state.isRecording || state.isProcessing;
  elements.audioFile.disabled = disabled;
  elements.audioFileButton.disabled = disabled;
  elements.uploadButton.disabled = disabled || !hasSelectedUpload();
}

function syncSelectedFileLabel() {
  const [file] = elements.audioFile.files ?? [];
  elements.audioFileName.textContent = file?.name || APP_CONFIG.audioFileEmptyLabel;
}

function syncResultActions() {
  const hasTranscript = Boolean(elements.transcript.value.trim());
  const disabled = state.isProcessing || !hasTranscript;

  elements.copyButton.disabled = disabled;
  elements.saveButton.disabled = disabled;
}

function clearResultMeta() {
  elements.resultMeta.hidden = true;
  elements.resultLanguage.textContent = "";
  elements.resultModel.textContent = "";
  elements.resultDuration.hidden = true;
  elements.resultDuration.textContent = "";
}

function setResultMeta(payload) {
  elements.resultLanguage.textContent = `Language: ${payload.language}`;
  elements.resultModel.textContent = `Model: ${payload.model_size}`;
  if (payload.duration_seconds == null) {
    elements.resultDuration.hidden = true;
    elements.resultDuration.textContent = "";
  } else {
    const durationSeconds = Number(payload.duration_seconds);
    elements.resultDuration.hidden = false;
    elements.resultDuration.textContent = `Duration: ${durationSeconds.toFixed(1)}s`;
  }
  elements.resultMeta.hidden = false;
}

function resetCopyButtonLabel() {
  if (state.copyFeedbackId !== null) {
    window.clearTimeout(state.copyFeedbackId);
    state.copyFeedbackId = null;
  }
  elements.copyButton.textContent = APP_CONFIG.copyButtonLabel;
}

function persistPreference(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Ignore storage failures in private browsing or locked-down contexts.
  }
}

function restorePreference(key, fallbackValue) {
  try {
    return window.localStorage.getItem(key) ?? fallbackValue;
  } catch {
    return fallbackValue;
  }
}

function resolveSelectPreference(element, storageKey) {
  const fallbackValue = element.value;
  const storedValue = restorePreference(storageKey, fallbackValue);
  const hasOption = Array.from(element.options).some((option) => option.value === storedValue);

  if (!hasOption) {
    persistPreference(storageKey, fallbackValue);
    return fallbackValue;
  }

  return storedValue;
}

function clearTimer() {
  if (state.timerId !== null) {
    window.clearInterval(state.timerId);
    state.timerId = null;
  }
}

function restoreInteractiveState(
  statusMessage,
  timerLabel,
  { resetButtonDisabled = false } = {},
) {
  state.isRecording = false;
  state.isProcessing = false;
  elements.recordButton.disabled = false;
  elements.recordButton.textContent = APP_CONFIG.recordButtonLabel;
  elements.resetButton.disabled = resetButtonDisabled;
  setSelectorsDisabled(false);
  syncUploadControls();
  syncResultActions();
  setTimerState(timerLabel);
  setStatus(statusMessage);
}

function restoreAfterFailedSubmission(message) {
  restoreInteractiveState(message, APP_CONFIG.timerIdleLabel);
  elements.resetButton.disabled = false;
}

function formatDuration(totalSeconds) {
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function sanitizeFilenamePart(value, fallback) {
  const normalized = String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

  return normalized || fallback;
}

function buildTranscriptFilename() {
  const language = sanitizeFilenamePart(state.lastResult?.language, "transcript");
  const modelSize = sanitizeFilenamePart(state.lastResult?.model_size, "model");
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");

  return `transcription-${language}-${modelSize}-${timestamp}.txt`;
}

function updateTimer() {
  if (!state.startedAt) {
    elements.timer.textContent = "00:00";
    return;
  }

  const elapsedSeconds = Math.floor((Date.now() - state.startedAt) / 1000);
  elements.timer.textContent = formatDuration(elapsedSeconds);
}

function resetUi() {
  if (state.isProcessing) {
    return;
  }

  clearTimer();
  clearActiveRecorder({ cancelPendingStop: true });
  state.chunks = [];
  state.startedAt = null;
  state.lastResult = null;
  elements.transcript.value = "";
  elements.audioFile.value = "";
  syncSelectedFileLabel();
  elements.timer.textContent = "00:00";
  resetCopyButtonLabel();
  clearResultMeta();
  restoreInteractiveState(APP_CONFIG.readyStatusMessage, APP_CONFIG.timerIdleLabel, {
    resetButtonDisabled: true,
  });
}

function setProcessingState(statusMessage) {
  state.isProcessing = true;
  elements.recordButton.disabled = true;
  elements.resetButton.disabled = true;
  setSelectorsDisabled(true);
  syncUploadControls();
  syncResultActions();
  setTimerState(APP_CONFIG.timerProcessingLabel, "busy");
  setStatus(statusMessage);
}

async function submitAudio(audio, filename, statusMessage) {
  setProcessingState(statusMessage);

  const formData = new FormData();
  formData.append("audio", audio, filename);
  formData.append("language", elements.language.value);
  formData.append("model_size", elements.modelSize.value);

  const response = await fetch("/api/transcriptions", {
    method: "POST",
    body: formData,
  });
  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.error.message);
  }

  state.isRecording = false;
  state.isProcessing = false;
  state.lastResult = payload;
  elements.transcript.value = payload.transcript;
  setResultMeta(payload);
  restoreInteractiveState(APP_CONFIG.completionStatusMessage, APP_CONFIG.timerReadyLabel);
}

async function startRecording() {
  elements.transcript.value = "";
  state.lastResult = null;
  resetCopyButtonLabel();
  clearResultMeta();
  syncResultActions();

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.mediaRecorder = new MediaRecorder(state.stream);
    state.chunks = [];
    state.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        state.chunks.push(event.data);
      }
    };
    state.mediaRecorder.onstop = () => {
      void handleRecordingStop();
    };
    state.mediaRecorder.start();
    state.isRecording = true;
    state.startedAt = Date.now();
    clearTimer();
    updateTimer();
    state.timerId = window.setInterval(updateTimer, 1000);
    elements.recordButton.textContent = APP_CONFIG.stopButtonLabel;
    elements.resetButton.disabled = false;
    setSelectorsDisabled(true);
    syncUploadControls();
    setTimerState(APP_CONFIG.timerRecordingLabel, "recording");
    setStatus(APP_CONFIG.recordingStatusMessage);
  } catch (error) {
    const microphoneErrorNames = new Set(["NotAllowedError", "SecurityError", "PermissionDeniedError"]);
    const message = microphoneErrorNames.has(error.name)
      ? APP_CONFIG.microphoneBlockedMessage
      : APP_CONFIG.recordingStartErrorMessage;
    clearActiveRecorder({ cancelPendingStop: true });
    state.startedAt = null;
    restoreInteractiveState(message, APP_CONFIG.timerIdleLabel, {
      resetButtonDisabled: true,
    });
  }
}

async function stopRecording() {
  clearTimer();
  clearActiveRecorder();
}

async function handleRecordingStop() {
  try {
    const blob = new Blob(state.chunks, { type: "audio/webm" });
    if (blob.size === 0) {
      throw new Error(APP_CONFIG.emptyRecordingMessage);
    }
    await submitAudio(
      blob,
      APP_CONFIG.defaultUploadFilename,
      APP_CONFIG.processingStatusMessage,
    );
  } catch (error) {
    restoreAfterFailedSubmission(error.message);
  }
}

async function handleFileUpload() {
  if (state.isRecording || state.isProcessing) {
    return;
  }

  const [file] = elements.audioFile.files ?? [];
  if (!file) {
    setStatus(APP_CONFIG.missingUploadMessage);
    syncUploadControls();
    return;
  }

  elements.transcript.value = "";
  state.lastResult = null;
  resetCopyButtonLabel();
  clearResultMeta();
  syncResultActions();

  try {
    await submitAudio(file, file.name, APP_CONFIG.uploadStatusMessage);
  } catch (error) {
    restoreAfterFailedSubmission(error.message);
  }
}

function initializePreferences() {
  elements.language.value = resolveSelectPreference(elements.language, STORAGE_KEYS.language);
  elements.modelSize.value = resolveSelectPreference(elements.modelSize, STORAGE_KEYS.modelSize);
}

elements.language.addEventListener("change", () => {
  persistPreference(STORAGE_KEYS.language, elements.language.value);
});

elements.modelSize.addEventListener("change", () => {
  persistPreference(STORAGE_KEYS.modelSize, elements.modelSize.value);
});

elements.audioFile.addEventListener("change", () => {
  syncSelectedFileLabel();
  syncUploadControls();

  const [file] = elements.audioFile.files ?? [];
  if (file && !state.isProcessing && !state.isRecording) {
    setStatus(APP_CONFIG.uploadReadyStatusMessage.replace("{filename}", file.name));
  } else if (!file && !state.isProcessing && !state.isRecording) {
    setStatus(APP_CONFIG.readyStatusMessage);
  }
});

elements.audioFileButton.addEventListener("click", () => {
  if (elements.audioFileButton.disabled) {
    return;
  }

  elements.audioFile.click();
});

elements.recordButton.addEventListener("click", async () => {
  if (state.isProcessing) {
    return;
  }

  if (state.isRecording) {
    state.isRecording = false;
    await stopRecording();
    return;
  }

  await startRecording();
});

elements.uploadButton.addEventListener("click", async () => {
  await handleFileUpload();
});

elements.resetButton.addEventListener("click", resetUi);
elements.copyButton.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(elements.transcript.value);
    resetCopyButtonLabel();
    elements.copyButton.textContent = APP_CONFIG.copySuccessLabel;
    state.copyFeedbackId = window.setTimeout(() => {
      elements.copyButton.textContent = APP_CONFIG.copyButtonLabel;
      state.copyFeedbackId = null;
    }, APP_CONFIG.copyFeedbackMs);
  } catch {
    setStatus(APP_CONFIG.clipboardUnavailableMessage);
  }
});

elements.saveButton.addEventListener("click", () => {
  if (!elements.transcript.value.trim()) {
    return;
  }

  try {
    const filename = buildTranscriptFilename();
    const fileUrl = window.URL.createObjectURL(new Blob([elements.transcript.value], { type: "text/plain;charset=utf-8" }));
    const link = document.createElement("a");

    link.href = fileUrl;
    link.download = filename;
    link.click();
    window.URL.revokeObjectURL(fileUrl);
    setStatus(APP_CONFIG.saveSuccessMessage.replace("{filename}", filename));
  } catch {
    setStatus(APP_CONFIG.saveUnavailableMessage);
  }
});

initializePreferences();
resetCopyButtonLabel();
elements.audioFileButton.textContent = APP_CONFIG.audioFileButtonLabel;
syncSelectedFileLabel();
setTimerState(APP_CONFIG.timerIdleLabel);
clearResultMeta();
syncUploadControls();
syncResultActions();
