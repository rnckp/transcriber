const state = {
  mediaRecorder: null,
  chunks: [],
  stream: null,
  timerId: null,
  progressTimerId: null,
  progressStartedAt: null,
  estimatedProcessingSeconds: null,
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
  processingProgress: document.querySelector("#processing-progress"),
  progressFill: document.querySelector("#progress-fill"),
  progressEta: document.querySelector("#progress-eta"),
  progressPercent: document.querySelector("#progress-percent"),
  progressTrack: document.querySelector(".progress-track"),
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

function clearProgressTimer() {
  if (state.progressTimerId !== null) {
    window.clearInterval(state.progressTimerId);
    state.progressTimerId = null;
  }
}

function formatRemainingTime(totalSeconds) {
  const seconds = Math.max(0, Math.ceil(totalSeconds));
  if (seconds < 60) {
    return `${seconds}s`;
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${String(remainingSeconds).padStart(2, "0")}s`;
}

function setProgress(value, label) {
  const progress = Math.min(1, Math.max(0, value));
  const percent = Math.round(progress * 100);
  const visiblePercent = Math.max(4, percent);

  elements.processingProgress.hidden = false;
  elements.progressFill.style.width = `${visiblePercent}%`;
  elements.progressPercent.textContent = `${percent}%`;
  elements.progressTrack.setAttribute("aria-valuenow", String(percent));
  elements.progressEta.textContent = label;
}

function resetProgress() {
  clearProgressTimer();
  state.progressStartedAt = null;
  state.estimatedProcessingSeconds = null;
  elements.processingProgress.hidden = false;
  setProgress(0, "Waiting for audio.");
}

function estimateProcessingSeconds(audioSeconds, modelSize) {
  const modelFactors = {
    tiny: 0.35,
    base: 0.5,
    small: 0.75,
    medium: 1.1,
    large: 1.6,
    "vibevoice-7b": 2.6,
  };
  const safeAudioSeconds = Number.isFinite(audioSeconds) && audioSeconds > 0 ? audioSeconds : 30;
  const factor = modelFactors[modelSize] ?? 1;

  return Math.max(8, Math.ceil(safeAudioSeconds * factor));
}

function estimateAudioSeconds(audio) {
  if (!(audio instanceof Blob) || audio.size === 0) {
    return Promise.resolve(null);
  }

  return new Promise((resolve) => {
    const audioElement = document.createElement("audio");
    const url = window.URL.createObjectURL(audio);
    let resolved = false;

    const finish = (value) => {
      if (resolved) {
        return;
      }
      resolved = true;
      window.clearTimeout(timeoutId);
      window.URL.revokeObjectURL(url);
      resolve(value);
    };

    const timeoutId = window.setTimeout(() => finish(null), 1500);

    audioElement.preload = "metadata";
    audioElement.onloadedmetadata = () => {
      finish(Number.isFinite(audioElement.duration) ? audioElement.duration : null);
    };
    audioElement.onerror = () => finish(null);
    audioElement.src = url;
  });
}

function startEstimatedProcessingProgress(estimatedSeconds) {
  clearProgressTimer();
  state.progressStartedAt = Date.now();
  state.estimatedProcessingSeconds = estimatedSeconds;

  const updateProgress = () => {
    const elapsedSeconds = (Date.now() - state.progressStartedAt) / 1000;
    const phaseProgress = Math.min(0.95, elapsedSeconds / estimatedSeconds);
    const progress = 0.2 + phaseProgress * 0.75;
    const remainingSeconds = Math.max(0, estimatedSeconds - elapsedSeconds);

    setProgress(
      progress,
      `Transcribing audio locally, about ${formatRemainingTime(remainingSeconds)} time left`,
    );
  };

  updateProgress();
  state.progressTimerId = window.setInterval(updateProgress, 1000);
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

function formatPercent(value) {
  return `${Math.min(100, Math.max(0, Math.round(value * 100)))}%`;
}

function waitForNextFrame() {
  return new Promise((resolve) => {
    window.requestAnimationFrame(resolve);
  });
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

function formatDiarizedTranscript(payload) {
  if (!Array.isArray(payload.segments) || payload.segments.length === 0) {
    return payload.transcript ?? "";
  }

  return payload.segments
    .map((segment) => `Speaker ${segment.speaker}: ${segment.text}`)
    .join("\n\n");
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
  resetProgress();
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
  setProgress(0, "Preparing audio...");
}

async function submitAudio(audio, filename, statusMessage, recordedDurationSeconds = null) {
  const audioSeconds = recordedDurationSeconds ?? await estimateAudioSeconds(audio);
  const estimatedProcessingSeconds = estimateProcessingSeconds(audioSeconds, elements.modelSize.value);

  setProcessingState(statusMessage);
  await waitForNextFrame();

  const formData = new FormData();
  formData.append("audio", audio, filename);
  formData.append("language", elements.language.value);
  formData.append("model_size", elements.modelSize.value);

  const payload = await new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();

    request.open("POST", "/api/transcriptions");
    request.upload.onprogress = (event) => {
      if (event.lengthComputable && event.total > 0) {
        const uploadProgress = event.loaded / event.total;
        const percent = formatPercent(uploadProgress);

        setStatus(`${statusMessage} ${percent}`);
        setProgress(
          uploadProgress * 0.2,
          `Uploading audio, ${percent} complete`,
        );
      }
    };
    request.upload.onload = () => {
      setStatus(APP_CONFIG.processingStatusMessage);
      startEstimatedProcessingProgress(estimatedProcessingSeconds);
    };
    request.onload = () => {
      let responsePayload;
      try {
        responsePayload = JSON.parse(request.responseText);
      } catch {
        reject(new Error(APP_CONFIG.processingNetworkErrorMessage));
        return;
      }
      if (request.status < 200 || request.status >= 300) {
        reject(new Error(responsePayload.error?.message ?? APP_CONFIG.processingNetworkErrorMessage));
        return;
      }
      resolve(responsePayload);
    };
    request.onerror = () => {
      reject(new Error(APP_CONFIG.processingNetworkErrorMessage));
    };
    request.send(formData);
  });

  state.isRecording = false;
  state.isProcessing = false;
  state.lastResult = payload;
  elements.transcript.value = formatDiarizedTranscript(payload);
  setResultMeta(payload);
  setProgress(1, "Complete");
  clearProgressTimer();
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
    const recordedDurationSeconds = state.startedAt
      ? Math.max(1, (Date.now() - state.startedAt) / 1000)
      : null;

    if (blob.size === 0) {
      throw new Error(APP_CONFIG.emptyRecordingMessage);
    }
    await submitAudio(
      blob,
      APP_CONFIG.defaultUploadFilename,
      APP_CONFIG.processingStatusMessage,
      recordedDurationSeconds,
    );
  } catch (error) {
    resetProgress();
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
    resetProgress();
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
