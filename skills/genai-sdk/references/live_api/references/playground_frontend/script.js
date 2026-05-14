/**
 * @fileoverview Front-end controller for the Live API testing playground.
 *
 * Responsibilities:
 *   - Modal-based session settings (open/close, fully closable).
 *   - Build a `BidiGenerateContentSetup` JSON from form fields (or a
 *     user-provided JSON override).
 *   - Bridge browser audio/video/text to the server via a WebSocket using
 *     serialized `BidiGenerateContent*` protos.
 *   - Render incoming model audio + transcriptions.
 *
 * NOTE: This reference uses Closure-style imports for proto types. When
 * adapting to your project, replace these with the appropriate import
 * mechanism for your build system (e.g. ES modules, script tags, etc.).
 */

// -- Proto imports (adapt to your project's module system) --
// These are Closure-style imports for reference. Replace with your proto
// library's import mechanism.
//
// import { BidiGenerateContentRealtimeInput } from '...';
// import { Blob } from '...';
// import { BidiGenerateContentClientMessage as ClientMessage } from '...';
// import { BidiGenerateContentServerMessage as ServerMessage } from '...';

// ---- Audio / video constants ----
const AUDIO_INPUT_SAMPLE_RATE = 16000;
const AUDIO_OUTPUT_SAMPLE_RATE = 24000;
const AUDIO_CHUNK_INTERVAL_MS = 20;
const AUDIO_CHANNEL_COUNT = 1;
const IDEAL_AUDIO_BUFFER_SIZE = (AUDIO_CHUNK_INTERVAL_MS / 1000) *
    AUDIO_INPUT_SAMPLE_RATE * AUDIO_CHANNEL_COUNT;
const AUDIO_BUFFER_SIZE =
    Math.pow(2, Math.ceil(Math.log2(IDEAL_AUDIO_BUFFER_SIZE)));
const VIDEO_FRAME_INTERVAL_MS = 1000;
const VIDEO_MAX_DIMENSION = 768;

/**
 * Host suffix per environment, used to compute the WebSocket endpoint URL.
 * Pattern: wss://{location}-{suffix}/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent
 * When location is "global", the location prefix is omitted.
 */
const HOST_SUFFIX_BY_ENV = {
  'prod': 'aiplatform.googleapis.com',
  'staging': 'staging-aiplatform.sandbox.googleapis.com',
  'autopush': 'autopush-aiplatform.sandbox.googleapis.com',
};

const WS_PATH =
    '/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent';

const DEFAULT_ENV = 'prod';
const DEFAULT_LOCATION = 'us-central1';
const DEFAULT_MODEL = 'gemini-live-2.5-flash-native-audio';

/**
 * Computes the WebSocket endpoint URL from environment and location.
 */
function computeEndpointUrl(env, location) {
  const suffix = HOST_SUFFIX_BY_ENV[env] || HOST_SUFFIX_BY_ENV['prod'];
  const host = location === 'global' ? suffix : `${location}-${suffix}`;
  return `wss://${host}${WS_PATH}`;
}

document.addEventListener('DOMContentLoaded', () => {
  // ===========================================================================
  // DOM lookup
  // ===========================================================================
  const $ = (id) => document.getElementById(id);

  const sessionButton = $('session-button');
  const openSettingsButton = $('open-settings');
  const settingsDoneButton = $('settings-done');
  const settingsModal = $('settings-modal');

  const statusIndicator = $('status-indicator');
  const statusDot = statusIndicator.querySelector('.status-dot');
  const statusLabel = statusIndicator.querySelector('.status-label');

  const conversation = $('conversation');
  const emptyState = $('empty-state');
  const chatInput = $('chat-input');
  const sendButton = $('send-button');
  const clearChatButton = $('clear-chat');

  const audioSelect = $('audio-source');
  const videoSelect = $('video-source');
  const videoElement = $('video-display');
  const videoEmpty = $('video-empty');
  const statusMessage = $('status-message');

  const kvEnvironment = $('kv-environment');
  const kvLocation = $('kv-location');
  const kvEndpoint = $('kv-endpoint');
  const kvModel = $('kv-model');
  const kvVoice = $('kv-voice');
  const kvLanguage = $('kv-language');

  // Settings form — environment / location / endpoint
  const envSegmented = $('env-segmented');
  const locationSelect = $('location');
  const endpointDisplay = $('endpoint-display');
  const modelInput = $('model-id');
  const voiceInput = $('voice-name');
  const langInput = $('language-code');
  const systemInstructionTextarea = $('system-instruction');
  const modalityAudio = $('modality-audio');
  const modalityText = $('modality-text');
  const genTemperature = $('gen-temperature');
  const genTopP = $('gen-top-p');
  const genTopK = $('gen-top-k');
  const genMaxTokens = $('gen-max-tokens');
  const genThinkingBudget = $('gen-thinking-budget');
  const genMediaResolution = $('gen-media-resolution');
  const inputTransCheckbox = $('input-transcription');
  const outputTransCheckbox = $('output-transcription');
  const activityHandlingSelect = $('activity-handling');
  const turnCoverageSelect = $('turn-coverage');
  const disableAadCheckbox = $('disable-automatic-activity-detection');
  const aadStartSensitivity = $('aad-start-sensitivity');
  const aadEndSensitivity = $('aad-end-sensitivity');
  const aadPrefixPadding = $('aad-prefix-padding');
  const aadSilenceDuration = $('aad-silence-duration');
  const ctxCompressionCheckbox = $('ctx-compression');
  const ctxTriggerInput = $('ctx-trigger-tokens');
  const ctxTargetInput = $('ctx-target-tokens');
  const proactiveAudioCheckbox = $('proactive-audio');
  const setupOverrideTextarea = $('setup-json-override');

  // ===========================================================================
  // State
  // ===========================================================================
  let playbackAudioContext = null;
  let nextBufferStartTime = 0;
  let currentBubbles = {};
  let localStream = null;
  let screenStream = null;
  let websocket = null;
  let audioStream = null;
  let audioContext = null;
  let scriptProcessor = null;
  let playbackBuffer = [];  // Simple array used as a FIFO queue
  let videoFrameIntervalId = null;
  let sessionStarted = false;
  let statusMessageTimeoutId = 0;

  const sessionId = crypto.randomUUID();
  console.log('Session ID:', sessionId);

  // Current selections (mutable; updated by the segmented control / selects)
  let currentEnv = DEFAULT_ENV;
  let currentLocation = DEFAULT_LOCATION;

  // Set defaults
  modelInput.value = DEFAULT_MODEL;
  locationSelect.value = DEFAULT_LOCATION;

  // ===========================================================================
  // Environment segmented control
  // ===========================================================================
  function setEnvironment(value) {
    currentEnv = value;
    envSegmented.querySelectorAll('.segmented-option').forEach((opt) => {
      opt.classList.toggle(
          'is-active', opt.getAttribute('data-value') === value);
    });
    recomputeDerivedFields();
  }
  envSegmented.querySelectorAll('.segmented-option').forEach((opt) => {
    opt.addEventListener('click', () => {
      setEnvironment(opt.getAttribute('data-value'));
    });
  });
  setEnvironment(DEFAULT_ENV);

  locationSelect.addEventListener('change', () => {
    currentLocation = locationSelect.value;
    recomputeDerivedFields();
  });

  // Recompute endpoint whenever env or location changes.
  function recomputeDerivedFields() {
    const endpoint = computeEndpointUrl(currentEnv, currentLocation);
    endpointDisplay.textContent = endpoint;
    refreshConfigSummary();
  }

  modelInput.addEventListener('input', refreshConfigSummary);

  // ===========================================================================
  // Playback buffer helpers (simple FIFO queue)
  // ===========================================================================
  function enqueue(item) { playbackBuffer.push(item); }
  function dequeue() { return playbackBuffer.shift(); }
  function peek() { return playbackBuffer[0]; }
  function isQueueEmpty() { return playbackBuffer.length === 0; }
  function clearQueue() { playbackBuffer = []; }

  // ===========================================================================
  // Status indicator + transient status messages
  // ===========================================================================
  function setSessionStatus(state) {
    statusDot.classList.remove('status-idle', 'status-active', 'status-error');
    if (state === 'active') {
      statusDot.classList.add('status-active');
      statusLabel.textContent = 'Session active';
    } else if (state === 'error') {
      statusDot.classList.add('status-error');
      statusLabel.textContent = 'Error';
    } else {
      statusDot.classList.add('status-idle');
      statusLabel.textContent = 'Idle';
    }
  }
  setSessionStatus('idle');

  function showStatus(message, isError = false) {
    statusMessage.textContent = message;
    statusMessage.className = 'status-message visible';
    statusMessage.classList.add(isError ? 'error' : 'success');
    if (statusMessageTimeoutId) clearTimeout(statusMessageTimeoutId);
    statusMessageTimeoutId = setTimeout(() => {
      statusMessage.textContent = '';
      statusMessage.className = 'status-message';
    }, 5000);
  }

  // ===========================================================================
  // Settings modal: open/close
  // ===========================================================================
  function openSettings() {
    settingsModal.hidden = false;
  }
  function closeSettings() {
    settingsModal.hidden = true;
  }

  openSettingsButton.addEventListener('click', openSettings);
  settingsDoneButton.addEventListener('click', () => {
    const override = setupOverrideTextarea.value.trim();
    if (override) {
      try {
        JSON.parse(override);
      } catch (e) {
        showStatus(`Invalid setup JSON override: ${e.message}`, true);
        return;
      }
    }
    refreshConfigSummary();
    closeSettings();
  });
  settingsModal.querySelectorAll('[data-close-modal]').forEach((el) => {
    el.addEventListener('click', closeSettings);
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !settingsModal.hidden) {
      closeSettings();
    }
  });

  // ===========================================================================
  // Save-recording modal (shown when the websocket closes after a session)
  // ===========================================================================
  const saveModal = $('save-recording-modal');
  const recordingFilenameInput = $('recording-filename');
  const recordingSaveButton = $('recording-save');
  const recordingDiscardButton = $('recording-discard');
  let pendingRecordingSessionId = '';

  function defaultRecordingFilename() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    const stamp =
        `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}` +
        `_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
    return `liveapi_session_${stamp}.pb`;
  }

  function openSaveRecordingModal(sid) {
    pendingRecordingSessionId = sid;
    recordingFilenameInput.value = defaultRecordingFilename();
    saveModal.hidden = false;
    setTimeout(() => recordingFilenameInput.focus(), 0);
  }
  function closeSaveModal() {
    saveModal.hidden = true;
    pendingRecordingSessionId = '';
  }
  saveModal.querySelectorAll('[data-close-save-modal]').forEach((el) => {
    el.addEventListener('click', closeSaveModal);
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !saveModal.hidden) closeSaveModal();
  });

  async function discardRecording(sid) {
    if (!sid) return;
    try {
      const fd = new FormData();
      fd.append('session_id', sid);
      await fetch('/recording/discard', {method: 'POST', body: fd});
    } catch (e) {
      console.warn('Failed to discard recording on server:', e);
    }
  }

  async function saveRecording() {
    const sid = pendingRecordingSessionId;
    if (!sid) return;
    const filename =
        recordingFilenameInput.value.trim() || defaultRecordingFilename();

    recordingSaveButton.disabled = true;
    recordingDiscardButton.disabled = true;

    try {
      if (typeof window.showSaveFilePicker === 'function') {
        let fileHandle;
        try {
          fileHandle = await window.showSaveFilePicker({
            suggestedName: filename,
            types: [{
              description: 'Protobuf recording (.pb)',
              accept: {'application/octet-stream': ['.pb']},
            }],
          });
        } catch (err) {
          if (err && err.name === 'AbortError') return;
          throw err;
        }
        const url = `/recording/download?session_id=${encodeURIComponent(sid)}`;
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }
        const writable = await fileHandle.createWritable();
        await response.body.pipeTo(writable);
      } else {
        const url =
            `/recording/download?session_id=${encodeURIComponent(sid)}` +
            `&filename=${encodeURIComponent(filename)}`;
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
      }
      await discardRecording(sid);
      showStatus('Recording saved.', false);
      closeSaveModal();
    } catch (e) {
      console.error('Failed to save recording:', e);
      showStatus(`Failed to save recording: ${e.message}`, true);
    } finally {
      recordingSaveButton.disabled = false;
      recordingDiscardButton.disabled = false;
    }
  }

  recordingSaveButton.addEventListener('click', saveRecording);
  recordingDiscardButton.addEventListener('click', async () => {
    const sid = pendingRecordingSessionId;
    closeSaveModal();
    await discardRecording(sid);
    showStatus('Recording discarded.', false);
  });

  // ===========================================================================
  // Sidebar config summary
  // ===========================================================================
  function refreshConfigSummary() {
    kvEnvironment.textContent = currentEnv || '—';
    kvLocation.textContent = currentLocation || '—';
    kvEndpoint.textContent = computeEndpointUrl(currentEnv, currentLocation);
    kvModel.textContent = modelInput.value || '—';
    kvVoice.textContent = voiceInput.value || '—';
    kvLanguage.textContent = langInput.value || '—';
  }
  [voiceInput, langInput].forEach((el) => {
    el.addEventListener('input', refreshConfigSummary);
  });
  refreshConfigSummary();

  // ===========================================================================
  // Build the setup JSON payload sent to /start.
  //
  // The output must conform to the BidiGenerateContentSetup proto message
  // (using proto3 JSON camelCase field names):
  //
  //   BidiGenerateContentSetup
  //     model                        (string)
  //     generationConfig             (GenerationConfig)
  //       responseModalities         (repeated Modality enum)
  //       temperature, topP, topK, maxOutputTokens
  //       speechConfig               (SpeechConfig)
  //         voiceConfig.prebuiltVoiceConfig.voiceName
  //         languageCode
  //       mediaResolution            (MediaResolution enum)
  //       thinkingConfig             (ThinkingConfig)
  //         thinkingBudget
  //     systemInstruction            (Content)
  //     realtimeInputConfig          (RealtimeInputConfig)
  //       activityHandling           (ActivityHandling enum)
  //       turnCoverage               (TurnCoverage enum)
  //       automaticActivityDetection (AutomaticActivityDetection)
  //         disabled, startOfSpeechSensitivity, endOfSpeechSensitivity
  //         prefixPaddingMs, silenceDurationMs
  //     inputAudioTranscription      (AudioTranscriptionConfig)
  //     outputAudioTranscription     (AudioTranscriptionConfig)
  //     contextWindowCompression     (ContextWindowCompressionConfig)
  //       triggerTokens, slidingWindow.targetTokens
  //     proactivity                  (ProactivityConfig)
  //       proactiveAudio
  //
  // Fields not surfaced in the UI (managed server-side or via JSON override):
  //     tools, sessionResumption
  // ===========================================================================
  function buildSetupJson() {
    const overrideText = setupOverrideTextarea.value.trim();
    if (overrideText) {
      return JSON.parse(overrideText);
    }

    const responseModalities = [];
    if (modalityAudio.checked) responseModalities.push('AUDIO');
    if (modalityText.checked) responseModalities.push('TEXT');

    const generationConfig = {
      responseModalities,
      speechConfig: {
        voiceConfig: {prebuiltVoiceConfig: {voiceName: voiceInput.value}},
        languageCode: langInput.value,
      },
    };
    if (genTemperature.value !== '') {
      generationConfig.temperature = Number(genTemperature.value);
    }
    if (genTopP.value !== '') generationConfig.topP = Number(genTopP.value);
    if (genTopK.value !== '') generationConfig.topK = Number(genTopK.value);
    if (genMaxTokens.value !== '') {
      generationConfig.maxOutputTokens = Number(genMaxTokens.value);
    }
    if (genThinkingBudget.value !== '') {
      generationConfig.thinkingConfig = {
        thinkingBudget: Number(genThinkingBudget.value),
      };
    }
    if (genMediaResolution.value) {
      generationConfig.mediaResolution = genMediaResolution.value;
    }

    // The frontend sends the short model ID; the backend server
    // reconstructs the fully-qualified model resource name using its
    // --project_id flag and the location from the endpoint URL.
    const setup = {
      model: modelInput.value,
      generationConfig,
      systemInstruction: {parts: [{text: systemInstructionTextarea.value}]},
    };

    if (inputTransCheckbox.checked) setup.inputAudioTranscription = {};
    if (outputTransCheckbox.checked) setup.outputAudioTranscription = {};

    const realtimeInputConfig = {};
    if (activityHandlingSelect.value) {
      realtimeInputConfig.activityHandling = activityHandlingSelect.value;
    }
    if (turnCoverageSelect.value) {
      realtimeInputConfig.turnCoverage = turnCoverageSelect.value;
    }
    const aad = {};
    if (disableAadCheckbox.checked) aad.disabled = true;
    if (aadStartSensitivity.value) {
      aad.startOfSpeechSensitivity = aadStartSensitivity.value;
    }
    if (aadEndSensitivity.value) {
      aad.endOfSpeechSensitivity = aadEndSensitivity.value;
    }
    if (aadPrefixPadding.value !== '') {
      aad.prefixPaddingMs = Number(aadPrefixPadding.value);
    }
    if (aadSilenceDuration.value !== '') {
      aad.silenceDurationMs = Number(aadSilenceDuration.value);
    }
    if (Object.keys(aad).length > 0) {
      realtimeInputConfig.automaticActivityDetection = aad;
    }
    if (Object.keys(realtimeInputConfig).length > 0) {
      setup.realtimeInputConfig = realtimeInputConfig;
    }

    if (ctxCompressionCheckbox.checked) {
      setup.contextWindowCompression = {
        triggerTokens: Number(ctxTriggerInput.value) || 100000,
        slidingWindow: {targetTokens: Number(ctxTargetInput.value) || 4000},
      };
    }

    if (proactiveAudioCheckbox.checked) {
      setup.proactivity = {proactiveAudio: true};
    }

    return setup;
  }

  // ===========================================================================
  // Conversation rendering
  // ===========================================================================
  function ensureEmptyStateHidden() {
    if (emptyState && emptyState.parentElement === conversation) {
      conversation.removeChild(emptyState);
    }
  }

  function renderTranscription(role, text, finished) {
    if (!text && !finished) return;
    if (text) {
      ensureEmptyStateHidden();
      let bubble = currentBubbles[role];
      if (!bubble) {
        bubble = document.createElement('div');
        bubble.classList.add('bubble');
        bubble.classList.add(role === 'user' ? 'bubble-user' : 'bubble-model');

        const roleSpan = document.createElement('span');
        roleSpan.classList.add('bubble-role');
        roleSpan.textContent = role;
        bubble.appendChild(roleSpan);

        const textSpan = document.createElement('span');
        textSpan.classList.add('bubble-text');
        bubble.appendChild(textSpan);

        conversation.appendChild(bubble);
        currentBubbles[role] = bubble;
      }
      bubble.querySelector('.bubble-text').textContent += text;
      conversation.scrollTop = conversation.scrollHeight;
    }
    if (finished) {
      currentBubbles[role] = null;
    }
  }

  /**
   * Renders a tool_call from the model as a distinct bubble in the timeline.
   */
  function renderToolCall(name, id, args) {
    ensureEmptyStateHidden();
    const bubble = document.createElement('div');
    bubble.classList.add('bubble', 'bubble-tool');

    const header = document.createElement('div');
    header.classList.add('tool-header');
    const label = document.createElement('span');
    label.classList.add('tool-label');
    label.textContent = 'tool call';
    const nameEl = document.createElement('span');
    nameEl.classList.add('tool-name');
    nameEl.textContent = name || '(unnamed)';
    header.appendChild(label);
    header.appendChild(nameEl);
    if (id) {
      const idEl = document.createElement('span');
      idEl.classList.add('tool-id');
      idEl.textContent = `id=${id}`;
      header.appendChild(idEl);
    }
    bubble.appendChild(header);

    if (args !== null && args !== undefined) {
      const pre = document.createElement('pre');
      pre.classList.add('tool-args');
      try {
        pre.textContent = JSON.stringify(args, null, 2);
      } catch (_) {
        pre.textContent = String(args);
      }
      bubble.appendChild(pre);
    }

    conversation.appendChild(bubble);
    conversation.scrollTop = conversation.scrollHeight;
  }

  /**
   * Renders a tool_call_cancellation as a small inline event.
   */
  function renderToolCallCancellation(ids) {
    ensureEmptyStateHidden();
    const bubble = document.createElement('div');
    bubble.classList.add('bubble', 'bubble-tool', 'bubble-tool-cancel');
    const header = document.createElement('div');
    header.classList.add('tool-header');
    const label = document.createElement('span');
    label.classList.add('tool-label');
    label.textContent = 'tool cancelled';
    header.appendChild(label);
    if (ids && ids.length) {
      const idEl = document.createElement('span');
      idEl.classList.add('tool-id');
      idEl.textContent = `ids=${ids.join(', ')}`;
      header.appendChild(idEl);
    }
    bubble.appendChild(header);
    conversation.appendChild(bubble);
    conversation.scrollTop = conversation.scrollHeight;
  }

  /**
   * Renders a tool response bubble.
   */
  function renderToolResponse(name, id, response) {
    ensureEmptyStateHidden();
    const bubble = document.createElement('div');
    bubble.classList.add('bubble', 'bubble-tool', 'bubble-tool-response');

    const header = document.createElement('div');
    header.classList.add('tool-header');
    const label = document.createElement('span');
    label.classList.add('tool-label');
    label.textContent = 'tool response';
    const nameEl = document.createElement('span');
    nameEl.classList.add('tool-name');
    nameEl.textContent = name || '(unnamed)';
    header.appendChild(label);
    header.appendChild(nameEl);
    if (id) {
      const idEl = document.createElement('span');
      idEl.classList.add('tool-id');
      idEl.textContent = `id=${id}`;
      header.appendChild(idEl);
    }
    bubble.appendChild(header);

    if (response !== null && response !== undefined) {
      const pre = document.createElement('pre');
      pre.classList.add('tool-args');
      try {
        pre.textContent = JSON.stringify(response, null, 2);
      } catch (_) {
        pre.textContent = String(response);
      }
      bubble.appendChild(pre);
    }

    conversation.appendChild(bubble);
    conversation.scrollTop = conversation.scrollHeight;
  }

  clearChatButton.addEventListener('click', () => {
    conversation.innerHTML = '';
    currentBubbles = {};
    if (emptyState) conversation.appendChild(emptyState);
  });

  // ===========================================================================
  // Audio playback
  // ===========================================================================
  function pcm16ToFloat32(int16Array) {
    const float32Array = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
      float32Array[i] = int16Array[i] / 32768.0;
    }
    return float32Array;
  }

  function playAudioChunk(audioData) {
    if (!playbackAudioContext) return;
    // Adapt this line to your proto library's API for getting raw bytes.
    const pcmData = new Int16Array(audioData.buffer);
    const floatData = pcm16ToFloat32(pcmData);
    const audioBuffer = playbackAudioContext.createBuffer(
        1, floatData.length, AUDIO_OUTPUT_SAMPLE_RATE);
    audioBuffer.copyToChannel(floatData, 0);
    const source = playbackAudioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(playbackAudioContext.destination);
    const startTime =
        Math.max(playbackAudioContext.currentTime, nextBufferStartTime);
    source.start(startTime);
    nextBufferStartTime = startTime + audioBuffer.duration;
  }

  // Drain loop: processes the playback buffer every 10ms.
  // Uses a single FIFO for audio + transcription + tool events to preserve
  // the order in which they arrived from the server.
  setInterval(() => {
    while (!isQueueEmpty()) {
      const item = peek();
      if (item.type === 'audio') {
        if (playbackAudioContext &&
            nextBufferStartTime > playbackAudioContext.currentTime + 0.5) {
          return;  // Wait for playback to catch up
        }
        playAudioChunk(dequeue().data);
      } else if (item.type === 'transcription') {
        const t = dequeue();
        renderTranscription(t.role, t.text, t.finished);
      } else if (item.type === 'newTranscriptionSignal') {
        currentBubbles[dequeue().role] = null;
      } else if (item.type === 'toolCall') {
        const tc = dequeue();
        renderToolCall(tc.name, tc.id, tc.args);
      } else if (item.type === 'toolCallCancellation') {
        const tcc = dequeue();
        renderToolCallCancellation(tcc.ids);
      } else if (item.type === 'toolResponse') {
        const tr = dequeue();
        renderToolResponse(tr.name, tr.id, tr.response);
      } else {
        dequeue();
      }
    }
  }, 10);

  // ===========================================================================
  // WebSocket (browser <-> companion server)
  // ===========================================================================
  function connectWebSocket() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl =
        `${wsProtocol}//${window.location.host}/ws?session_id=${sessionId}`;
    websocket = new WebSocket(wsUrl);
    websocket.binaryType = 'arraybuffer';

    websocket.onopen = () => {
      setSessionStatus('active');
      showStatus('Session connected.', false);
    };

    websocket.onmessage = (event) => {
      // Text frames are out-of-band events from the server (e.g. tool
      // responses). Binary frames are protobuf ServerMessage payloads.
      if (typeof event.data === 'string') {
        try {
          const evt = JSON.parse(event.data);
          if (evt && evt.type === 'tool_response') {
            enqueue({
              type: 'toolResponse',
              name: evt.name,
              id: evt.id,
              response: evt.response,
            });
          }
        } catch (e) {
          console.error('Failed to parse text WS frame:', e);
        }
        return;
      }
      try {
        // Deserialize the binary protobuf ServerMessage.
        // Adapt this to your proto library (e.g. ServerMessage.decode(),
        // ServerMessage.deserializeBinary(), etc.)
        const message = ServerMessage.deserializeBinary(event.data);

        // Tool calls
        if (message.hasToolCall()) {
          const fcs = message.getToolCall().getFunctionCallsList();
          fcs.forEach((fc) => {
            enqueue({
              type: 'toolCall',
              name: fc.getName(),
              id: fc.getId(),
              args: fc.getArgs() ? fc.getArgs().toJavaScript() : null,
            });
          });
        }
        if (message.hasToolCallCancellation()) {
          const ids = message.getToolCallCancellation().getIdsList();
          enqueue({type: 'toolCallCancellation', ids});
        }

        if (!message.hasServerContent()) return;
        const sc = message.getServerContent();

        // Model audio
        if (sc.hasModelTurn()) {
          sc.getModelTurn().getPartsList().forEach((part) => {
            if (part.hasInlineData() &&
                part.getInlineData().getMimeType().startsWith('audio/')) {
              enqueue({type: 'audio', data: part.getInlineData()});
            }
          });
        }
        // Transcriptions
        if (sc.hasInputTranscription()) {
          const t = sc.getInputTranscription();
          enqueue({
            type: 'transcription',
            role: 'user',
            text: t.getText(),
            finished: t.getFinished(),
          });
        }
        if (sc.hasOutputTranscription()) {
          const t = sc.getOutputTranscription();
          enqueue({
            type: 'transcription',
            role: 'model',
            text: t.getText(),
            finished: t.getFinished(),
          });
        }
        // Interrupt: flush the playback queue
        if (sc.getInterrupted()) {
          clearQueue();
        }
        // Turn complete: close current bubbles for next turn
        if (sc.getTurnComplete()) {
          enqueue({type: 'newTranscriptionSignal', role: 'model'});
          enqueue({type: 'newTranscriptionSignal', role: 'user'});
        }
      } catch (e) {
        console.error('Failed to decode incoming message:', e);
      }
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      showStatus('WebSocket error.', true);
    };

    websocket.onclose = () => {
      if (sessionStarted && websocket) {
        showStatus('WebSocket closed unexpectedly.', true);
      }
      const hadActiveSession = sessionStarted;
      websocket = null;
      sessionStarted = false;
      sessionButton.textContent = 'Start session';
      sessionButton.disabled = false;
      openSettingsButton.disabled = false;
      setSessionStatus('idle');
      stopAudioStreaming();
      stopMediaStreams();
      if (hadActiveSession) {
        setTimeout(() => openSaveRecordingModal(sessionId), 800);
      }
    };
  }

  // ===========================================================================
  // Session start/stop
  // ===========================================================================
  sessionButton.addEventListener('click', async () => {
    sessionButton.disabled = true;

    audioSelect.value = 'none';
    videoSelect.value = 'none';
    stopAudioStreaming();
    stopMediaStreams();

    if (!sessionStarted) {
      try {
        let setupJson;
        try {
          setupJson = buildSetupJson();
        } catch (e) {
          throw new Error(`Invalid setup JSON override: ${e.message}`);
        }
        openSettingsButton.disabled = true;

        const body = {
          session_id: sessionId,
          endpoint_url: computeEndpointUrl(currentEnv, currentLocation),
          setup: setupJson,
        };
        const response = await fetch('/start', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body),
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }
        const result = await response.json();
        if (result.status !== 'started') {
          throw new Error('Backend did not confirm session start.');
        }

        if (!playbackAudioContext) {
          playbackAudioContext =
              new AudioContext({sampleRate: AUDIO_OUTPUT_SAMPLE_RATE});
        }
        connectWebSocket();
        sessionStarted = true;
        sessionButton.textContent = 'Stop session';
      } catch (error) {
        console.error('Failed to start session:', error);
        showStatus(`Failed to start session: ${error.message}`, true);
        setSessionStatus('error');
        sessionStarted = false;
        openSettingsButton.disabled = false;
      } finally {
        sessionButton.disabled = false;
      }
    } else {
      sessionButton.disabled = true;
      if (websocket) {
        websocket.close();
      } else {
        sessionStarted = false;
        sessionButton.textContent = 'Start session';
        sessionButton.disabled = false;
        openSettingsButton.disabled = false;
        setSessionStatus('idle');
      }
    }
  });

  // ===========================================================================
  // Media devices: enumeration + selection
  // ===========================================================================
  async function populateMediaDevices() {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      await navigator.mediaDevices.getUserMedia({audio: true, video: true});
      const devices = await navigator.mediaDevices.enumerateDevices();
      audioSelect.innerHTML = '<option value="none">None</option>';
      videoSelect.innerHTML = '<option value="none">None</option>';
      devices.forEach((device) => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.text =
            device.label || `Device ${device.deviceId.substring(0, 8)}`;
        if (device.kind === 'audioinput')
          audioSelect.appendChild(option);
        else if (device.kind === 'videoinput')
          videoSelect.appendChild(option);
      });
      const screenOption = document.createElement('option');
      screenOption.value = 'screen';
      screenOption.text = 'Screen sharing';
      videoSelect.appendChild(screenOption);
    } catch (err) {
      console.error('Error populating devices:', err);
      const screenOption = document.createElement('option');
      screenOption.value = 'screen';
      screenOption.text = 'Screen sharing';
      videoSelect.appendChild(screenOption);
    }
  }

  // ----- Video -----
  videoSelect.addEventListener('change', async () => {
    const value = videoSelect.value;
    stopMediaStreams();
    if (value === 'screen') {
      try {
        screenStream =
            await navigator.mediaDevices.getDisplayMedia({video: true});
        videoElement.srcObject = screenStream;
        videoEmpty.hidden = true;
        startVideoFrameCapture();
        screenStream.getTracks().forEach((track) => {
          track.onended = () => {
            if (videoSelect.value === 'screen') {
              videoSelect.value = 'none';
              stopMediaStreams();
            }
          };
        });
      } catch (err) {
        console.error('Screen sharing failed:', err);
        videoSelect.value = 'none';
      }
    } else if (value !== 'none') {
      try {
        localStream = await navigator.mediaDevices.getUserMedia(
            {video: {deviceId: {exact: value}}});
        videoElement.srcObject = localStream;
        videoEmpty.hidden = true;
        startVideoFrameCapture();
      } catch (err) {
        console.error('Failed to get camera stream:', err);
        videoSelect.value = 'none';
      }
    }
  });

  function stopMediaStreams() {
    stopVideoFrameCapture();
    if (localStream) {
      localStream.getTracks().forEach((t) => t.stop());
      localStream = null;
    }
    if (screenStream) {
      screenStream.getTracks().forEach((t) => t.stop());
      screenStream = null;
    }
    videoElement.srcObject = null;
    videoEmpty.hidden = false;
  }

  // ----- Audio capture -----
  function floatTo16BitPCM(input) {
    const output = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
      const s = Math.max(-1, Math.min(1, input[i]));
      output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return output;
  }

  function processAudioChunk(audioProcessingEvent) {
    if (!sessionStarted || !websocket ||
        websocket.readyState !== WebSocket.OPEN) {
      return;
    }
    const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
    const pcm16Data = floatTo16BitPCM(inputData);
    const audioBytes = new Uint8Array(pcm16Data.buffer);

    // Build a ClientMessage with realtimeInput.audio and send as binary.
    // Adapt the proto construction to your library:
    //   const blob = new Blob();
    //   blob.setMimeType(`audio/pcm;rate=${AUDIO_INPUT_SAMPLE_RATE}`);
    //   blob.setData(audioBytes);
    //   const realtimeInput = new BidiGenerateContentRealtimeInput();
    //   realtimeInput.setAudio(blob);
    //   const msg = new ClientMessage();
    //   msg.setRealtimeInput(realtimeInput);
    //   websocket.send(msg.serializeBinary());
    const audioBlob =
        new Blob()
            .setMimeType(`audio/pcm;rate=${AUDIO_INPUT_SAMPLE_RATE}`)
            .setData(audioBytes);
    const realtimeInput =
        new BidiGenerateContentRealtimeInput().setAudio(audioBlob);
    websocket.send(
        new ClientMessage().setRealtimeInput(realtimeInput).serializeBinary());
  }

  async function startAudioStreaming(deviceId) {
    try {
      audioStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: {exact: deviceId},
          sampleRate: AUDIO_INPUT_SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      audioContext = new AudioContext({sampleRate: AUDIO_INPUT_SAMPLE_RATE});
      scriptProcessor =
          audioContext.createScriptProcessor(AUDIO_BUFFER_SIZE, 1, 1);
      const source = audioContext.createMediaStreamSource(audioStream);
      source.connect(scriptProcessor);
      scriptProcessor.connect(audioContext.destination);
      scriptProcessor.onaudioprocess = processAudioChunk;
    } catch (err) {
      console.error('Failed to start audio streaming:', err);
      showStatus(`Audio error: ${err.message}`, true);
      stopAudioStreaming();
      audioSelect.value = 'none';
    }
  }

  function stopAudioStreaming() {
    if (scriptProcessor) {
      scriptProcessor.disconnect();
      scriptProcessor.onaudioprocess = null;
      scriptProcessor = null;
    }
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }
    if (audioStream) {
      audioStream.getTracks().forEach((t) => t.stop());
      audioStream = null;
    }
  }

  audioSelect.addEventListener('change', () => {
    stopAudioStreaming();
    if (audioSelect.value !== 'none') startAudioStreaming(audioSelect.value);
  });

  // ----- Video frame capture -----
  const videoCanvas = document.createElement('canvas');

  async function captureAndSendVideoFrame() {
    if (!sessionStarted || !websocket ||
        websocket.readyState !== WebSocket.OPEN || !videoElement.srcObject ||
        videoElement.paused || videoElement.ended ||
        videoElement.videoWidth === 0) {
      return;
    }
    const w = videoElement.videoWidth;
    const h = videoElement.videoHeight;
    const ratio = w / h;
    let tw, th;
    if (w >= h) {
      tw = VIDEO_MAX_DIMENSION;
      th = VIDEO_MAX_DIMENSION / ratio;
    } else {
      th = VIDEO_MAX_DIMENSION;
      tw = VIDEO_MAX_DIMENSION * ratio;
    }
    videoCanvas.width = Math.round(tw);
    videoCanvas.height = Math.round(th);
    const ctx = videoCanvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0, videoCanvas.width, videoCanvas.height);
    videoCanvas.toBlob(async (blob) => {
      if (blob && websocket && websocket.readyState === WebSocket.OPEN) {
        const imageBytes = new Uint8Array(await blob.arrayBuffer());
        // Build a ClientMessage with realtimeInput.video and send.
        // Adapt proto construction to your library.
        const imageBlob =
            new Blob().setMimeType('image/jpeg').setData(imageBytes);
        const realtimeInput =
            new BidiGenerateContentRealtimeInput().setVideo(imageBlob);
        websocket.send(new ClientMessage()
                           .setRealtimeInput(realtimeInput)
                           .serializeBinary());
      }
    }, 'image/jpeg', 1);
  }

  function startVideoFrameCapture() {
    stopVideoFrameCapture();
    videoFrameIntervalId =
        setInterval(captureAndSendVideoFrame, VIDEO_FRAME_INTERVAL_MS);
  }
  function stopVideoFrameCapture() {
    if (videoFrameIntervalId) {
      clearInterval(videoFrameIntervalId);
      videoFrameIntervalId = null;
    }
  }

  populateMediaDevices();

  // ===========================================================================
  // Chat input
  // ===========================================================================
  sendButton.addEventListener('click', () => {
    const text = chatInput.value;
    if (text && websocket && websocket.readyState === WebSocket.OPEN) {
      // Send text via realtimeInput.text
      const realtimeInput =
          new BidiGenerateContentRealtimeInput().setText(text);
      websocket.send(new ClientMessage()
                         .setRealtimeInput(realtimeInput)
                         .serializeBinary());
      renderTranscription('user', text, true);
      chatInput.value = '';
    }
  });
  chatInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendButton.click();
    }
  });
});
