---
name: live-api-interactive-ui
description: >-
  Specification for a polished, single-page browser playground for the Gemini
  Live API over WebSockets. Provides session lifecycle controls, a modal
  settings dialog that builds a `BidiGenerateContentSetup`, microphone /
  camera / screen-share streaming with PCM resampling and JPEG frame
  sampling, a chat-style transcript with tool-call bubbles, low-latency
  audio playback with interrupt handling, and an optional
  save-recording-on-exit dialog. Use when generating the test UI required by
  Step 6 of the LiveAPI service skill.
---

# Live API Interactive UI

This document describes a **fancy, production-quality single-page web
playground** for exercising a Live API client implementation that connects
to the Gemini Live API over **WebSockets** (the public wire protocol from
`client_server_messages.md`). Use it as the spec for the test UI required
by Step 6 of the parent skill.

The UI is intentionally polished — a thin, responsive shell with a modal
settings dialog, a chat-style transcript, a side rail for sources and live
config, and an optional save-recording dialog when a session ends. It is
designed to fit in a laptop viewport (≥ 1366×768) without page-level
scrolling.

---

## 1. High-level layout

```
┌──────────────────────────────────────────────────────────────────────┐
│ Header                                                               │
│  [logo]  Live API Playground   ● Idle / ● Active / ● Error           │
│                                              [Settings] [Start]      │
├──────────────────────────────────────────────────────────────────────┤
│ Conversation panel                            │ Side rail            │
│  ┌────────────────────────────────────┐ Clear │ ─ Configuration ─    │
│  │ user: Hello                         │      │   Endpoint  …        │
│  │ model: Hi! How can I help?          │      │   Model     …        │
│  │ [tool call] get_weather id=…        │      │   Voice     …        │
│  │ [tool resp] get_weather id=…        │      │   Language  …        │
│  │                                     │      │ ─ Sources ─          │
│  └────────────────────────────────────┘      │   Microphone [▾]     │
│  ┌─────────────────────────────────────┐     │   Video      [▾]     │
│  │ [textarea]                  [Send]  │     │ ─ Preview ─          │
│  └─────────────────────────────────────┘     │   ┌────────────┐     │
│                                               │   │ <video>    │     │
│                                               │   └────────────┘     │
│                                               │ [transient status]   │
└──────────────────────────────────────────────────────────────────────┘
```

Implementation:

- The page-level container is `display: flex; flex-direction: column;
  height: 100vh; overflow: hidden;` so the **page itself never scrolls**.
- Inner regions (the conversation list, the side rail, the modal body)
  scroll independently with `overflow: auto`.
- Main area uses `display: grid; grid-template-columns: minmax(0, 1fr)
  320px;` so the conversation grows and the side rail stays at a fixed
  width on desktop.
- Below ~900 px, collapse to a single column via a media query and cap
  the side rail at `max-height: 40vh`.

A polished design system (recommended):

- A small token set in `:root` for `--bg`, `--surface`, `--border`,
  `--text`, `--text-muted`, `--accent`, `--success`, `--warning`,
  `--danger`, plus radii and shadows.
- A primary accent (e.g. indigo) plus a soft variant for focus rings.
- Subtle drop shadows on panels and cards; rounded corners (`8–12 px`).
- Modal cards use a `backdrop-filter: blur(4px)` backdrop with a
  `pop-in` animation (small `translateY` + scale).

---

## 2. Header

- **Brand mark**: a small gradient square + product title (e.g. "Live API
  Playground").
- **Status indicator** (pill-shaped):
  - `● Idle` — gray dot, neutral pill.
  - `● Session active` — green dot with a subtle pulsing ring (CSS
    keyframe animation alternating opacity).
  - `● Error` — red dot.
- **Action buttons**:
  - `Settings` — opens the settings modal (§ 4).
  - `Start session` / `Stop session` — toggles session lifecycle
    (§ 6).

---

## 3. Main area

### 3.1 Conversation panel (left)

- Scrollable list of message bubbles. Auto-scroll to bottom on each new
  bubble or appended text.
- An **empty state** placeholder (icon + title + hint) is shown when
  there are no messages.
- A `Clear` link button in the panel header empties the conversation
  and re-shows the empty state.
- **Bubble types**:
  - **User bubble** (`bubble-user`): right-aligned, accent background,
    white text.
  - **Model bubble** (`bubble-model`): left-aligned, neutral surface
    background with border.
  - **Tool-call bubble** (`bubble-tool`): full-width, light-purple
    background. Header row shows a `tool call` label badge, the function
    name (monospace), and the call `id` aligned to the right. A
    `<pre>` block underneath shows pretty-printed `args` JSON.
  - **Tool-response bubble** (`bubble-tool-response`): same shape as
    tool-call but labelled `tool response` with the response JSON.
  - **Tool-call cancellation bubble** (`bubble-tool-cancel`): warning
    yellow tint, shows the `ids[]` of cancelled calls.
- **Composer** at the bottom:
  - `<textarea>` that auto-grows up to a max height (~140 px).
  - Pressing **Enter** sends; **Shift+Enter** inserts a newline.
  - Send button is disabled when there is no active session.

#### 3.1.1 Streaming transcripts

Per-role streaming bubble model:

- Maintain a `currentBubbles` map keyed by role (`user` / `model`).
- For each incoming transcription chunk:
  - If no current bubble for that role exists, create a new one and
    append it.
  - Append the chunk's text to the current bubble's text node.
- When a transcription chunk has `finished == true`, **clear** the
  current bubble for that role so the next chunk starts a fresh bubble.
- Also clear both `user` and `model` current bubbles on a `turnComplete`
  signal so the next turn starts fresh.

### 3.2 Side rail (right)

Three (or four) sections, each separated by a hairline divider.

#### 3.2.1 Configuration summary

A compact `<dl>` (key/value list) showing the current selections so the
user can verify them at a glance without re-opening the modal:

| Key         | Value source |
| ----------- | --- |
| Environment | `autopush` / `staging` / `prod` (see § 4.1). |
| Location    | Region code, e.g. `us-central1`, `europe-west4`, `global` (see § 4.1). |
| Endpoint    | **Computed** WebSocket URL the connection will use; read-only. Derived from Environment + Location per § 4.1. |
| Model       | **Computed** fully-qualified model name (`projects/<p>/locations/<l>/publishers/<m>`); read-only. Derived from project id + Location + the raw model id input per § 4.2. |
| Voice       | `prebuiltVoiceConfig.voiceName`. |
| Language    | `speechConfig.languageCode`. |

Long values truncate with ellipsis (`white-space: nowrap; overflow:
hidden; text-overflow: ellipsis`). Endpoint and Model update **live** as
the user changes Environment / Location / model id in the settings
modal so the user always sees what will actually be sent.

#### 3.2.2 Sources

Two `<select>` controls:

- **Microphone**: `None` + every `audioinput` device discovered by
  `navigator.mediaDevices.enumerateDevices()`. Changing the value starts
  / stops the audio capture pipeline (§ 5.1).
- **Video**: `None` + every `videoinput` device + a special
  `Screen sharing` option that uses `getDisplayMedia()`. Changing the
  value starts / stops the corresponding video pipeline (§ 5.2).

#### 3.2.3 Preview

A 16:9 placeholder showing the current `<video>` element (`autoplay`,
`muted`, `playsinline`). When no video stream is active, show a
"No video" placeholder centered inside the box.

#### 3.2.4 Transient status messages

A small banner area used for short-lived success / error messages
("Session connected", "WebSocket closed unexpectedly", etc.). Auto-fades
after ~5 s. Use color tints (light green for success, light red for
error) with matching text color.

---

## 4. Settings modal

Opened from the header `Settings` button. Closable via:

- The `×` icon button.
- A `Cancel` button in the footer.
- Clicking the backdrop.
- Pressing `Escape`.

A `Done` button in the footer validates and closes (see § 4.6).

The modal is a centered card with `max-width: 640px` and
`max-height: calc(100vh - 48px)`; the body scrolls when content overflows.

### 4.1 Environment, Location, and computed Endpoint

The user does **not** type the WebSocket endpoint URL directly. Instead
they pick an **Environment** and a **Location**; the UI computes the
endpoint from those two selections and **displays it as read-only**
(monospace, dimmed, with a small "computed" badge so it's clear it
cannot be edited).

#### 4.1.1 Environment

A segmented control or `<select>` with three options:

| Value        | Meaning |
| ------------ | --- |
| `autopush`   | Continuous-deployment / pre-staging environment. Use for the latest server-side changes. |
| `staging`    | Pre-production environment. Use for stability testing. |
| `prod`       | Production environment. Default. |

Default selection: `prod`.

#### 4.1.2 Location

A `<select>` of supported regions plus `global`. Suggested options
(adjust to whatever set the underlying client implementation supports):

`global`, `us-central1`, `us-east4`, `us-east5`, `us-west1`,
`us-west4`, `europe-west1`, `europe-west2`, `europe-west3`,
`europe-west4`, `europe-southwest1`, `asia-east1`,
`asia-northeast1`, `asia-southeast1`.

Default: `us-central1`.

#### 4.1.3 Computed endpoint

The endpoint is derived from `(environment, location)` using the
following rules. The exact host suffix per environment must match the
upstream service's documented hosts; the patterns below are the
templates the UI must follow:

```
host_suffix = {
  prod:     "aiplatform.googleapis.com",
  staging:  "staging-aiplatform.sandbox.googleapis.com",
  autopush: "autopush-aiplatform.sandbox.googleapis.com",
}[environment]

if location == "global":
    host = host_suffix
else:
    host = f"{location}-{host_suffix}"

endpoint = f"wss://{host}/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"
```

Examples:

| Environment | Location       | Computed endpoint |
| ----------- | -------------- | --- |
| `prod`      | `us-central1`  | `wss://us-central1-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent` |
| `prod`      | `global`       | `wss://aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent` |
| `staging`   | `europe-west4` | `wss://europe-west4-staging-aiplatform.sandbox.googleapis.com/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent` |
| `autopush`  | `us-central1`  | `wss://us-central1-autopush-aiplatform.sandbox.googleapis.com/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent` |

#### 4.1.4 Display widget

Render the computed endpoint as a **read-only** display (NOT an
`<input>`):

- A monospace, dimmed text block with `user-select: text` so the user
  can still copy it.
- A small "computed" badge or hint *"Computed from Environment +
  Location."* next to it.
- A copy-to-clipboard icon button (optional but recommended) for
  convenience.
- Recompute and re-render the value **immediately** on any change to
  Environment or Location, and propagate the new value to the side-rail
  Configuration summary (§ 3.2.1).

If the user needs a non-derivable endpoint (e.g. testing a private
deployment), they can still use the **Setup JSON override** (§ 4.5.6)
plus an out-of-band override on the underlying client implementation.
The UI deliberately does NOT expose a free-text endpoint field to keep
the common case bullet-proof.

### 4.2 Model ID and computed fully-qualified model name

The user supplies the **short** model id (e.g.
`gemini-live-2.5-flash-native-audio`); the UI computes the
fully-qualified name that is sent to the server, using the project id
(known to the page from the server-side configuration) and the
Location selected in § 4.1.2.

#### 4.2.1 Model id input

A plain text input for the short model id only. Default:
`gemini-live-2.5-flash-native-audio` (or whichever your implementation
prefers).

#### 4.2.2 Computed fully-qualified model name

Build the value sent on the wire as:

```
model = f"projects/{project_id}/locations/{location}/publishers/{model_id}"
```

- `project_id` is the page's bound project (from the server-side
  configuration; it is NOT a user-editable field on this page).
- `location` is § 4.1.2's selection. When `global` is selected, use the
  literal string `global` as the location segment.
- `model_id` is § 4.2.1's value, untrimmed of any user-supplied prefix
  (the input should accept just `gemini-live-...` style ids).

Examples:

| project_id      | location       | model_id                                  | Resulting `setup.model` |
| --------------- | -------------- | ----------------------------------------- | --- |
| `my-project`    | `us-central1`  | `gemini-live-2.5-flash-native-audio`      | `projects/my-project/locations/us-central1/publishers/gemini-live-2.5-flash-native-audio` |
| `my-project`    | `global`       | `gemini-live-2.5-flash-native-audio`      | `projects/my-project/locations/global/publishers/gemini-live-2.5-flash-native-audio` |

#### 4.2.3 Display widget

Below the short model id input, render the **computed** fully-qualified
model name as a read-only monospace block (same styling as the computed
endpoint in § 4.1.4) with a "computed" badge. Update live as the user
changes the model id input or the Location selector.

This computed value is the one written into `setup.model` when building
the setup payload (§ 4.6 / § 6.1).

### 4.3 Speech (split row)

Two side-by-side text inputs:

- **Voice**: name of a prebuilt voice (default `Puck`). See
  `client_server_messages.md` § "Voices supported" for the full list.
- **Language**: BCP-47 code (default `en-US`).

### 4.4 System instruction

A multi-line `<textarea>` (default rows: 4) seeded with a sensible
example (e.g. *"You are a helpful assistant…"*). Becomes
`setup.systemInstruction.parts[0].text`.

### 4.5 Advanced sections (collapsed by default)

Each of these is a `<details>` element with a chevron icon that rotates
when opened. Keep them collapsed by default to keep the modal scannable.

#### 4.5.1 Generation parameters

- **Response modalities**: two checkboxes — `AUDIO` (default checked)
  and `TEXT`. Becomes `generationConfig.responseModalities[]`.
- **Temperature** / **Top P** / **Top K** / **Max output tokens**:
  numeric inputs, all optional. Empty means "default"; non-empty values
  are written into `generationConfig`.

#### 4.5.2 Transcription

- **Input audio transcription** (default checked) → `setup.inputAudioTranscription = {}`.
- **Output audio transcription** (default checked) → `setup.outputAudioTranscription = {}`.

#### 4.5.3 Realtime input

- **Activity handling** select with options:
  - `Default (interrupts on activity)` (no field set)
  - `START_OF_ACTIVITY_INTERRUPTS`
  - `NO_INTERRUPTION`
- **Disable automatic activity detection** checkbox →
  `realtimeInputConfig.automaticActivityDetection.disabled = true`.

When any of these are non-default, write a `realtimeInputConfig` object
into `setup`; otherwise omit it entirely.

#### 4.5.4 Context window compression

- **Enable sliding window compression** checkbox (default checked).
  When checked, write:

  ```js
  setup.contextWindowCompression = {
    triggerTokens: <number, default 100000>,
    slidingWindow: { targetTokens: <number, default 4000> },
  }
  ```

- Two number inputs for `triggerTokens` and `targetTokens`.

#### 4.5.5 Proactivity

- **Proactive audio** checkbox → `setup.proactivity = { proactiveAudio: true }`
  (Google AI only — see `client_server_messages.md` § 15).

#### 4.5.6 Setup JSON override

A monospace `<textarea>` (rows: 8). When non-empty, it is parsed as a
`BidiGenerateContentSetup` JSON object and used **verbatim** — every
structured field above is ignored. A small inline help message states
this. Validate the JSON when the user clicks `Done` and refuse to close
the modal if parsing fails (show an inline error).

### 4.6 `Done` validation

On click:

1. If the override textarea is non-empty, attempt `JSON.parse`. On
   failure, show an error toast and keep the modal open.
2. Refresh the side-rail configuration summary (§ 3.2.1).
3. Close the modal.

---

## 5. Source pipelines

### 5.1 Microphone capture & streaming

Constants:

- Input sample rate: **16 kHz** (the model's input rate).
- Channels: **1** (mono).
- Chunk interval: **20 ms** (typical).
- Buffer size: round `chunkInterval * sampleRate` up to the next power
  of two (e.g. 16000 × 0.02 = 320 → 512).

Pipeline:

1. On microphone selection change, call:
   ```js
   navigator.mediaDevices.getUserMedia({
     audio: {
       deviceId: {exact: <id>},
       sampleRate: 16000,
       channelCount: 1,
       echoCancellation: true,
       noiseSuppression: true,
       autoGainControl: true,
     }
   })
   ```
2. Create an `AudioContext({sampleRate: 16000})`.
3. Create a `ScriptProcessorNode` (or `AudioWorkletNode` if you prefer
   the modern API) with the chosen buffer size.
4. Source `MediaStreamSource` → processor → `audioContext.destination`.
5. In the `onaudioprocess` callback (or worklet message), grab the
   first channel's `Float32Array`, clamp each sample to `[-1, 1]`, and
   convert to little-endian `int16` PCM:
   ```
   sample16 = (s < 0) ? s * 0x8000 : s * 0x7FFF
   ```
6. Wrap the bytes into a `Blob` proto with
   `mime_type = "audio/pcm;rate=16000"`, set it as
   `BidiGenerateContentRealtimeInput.audio`, wrap in a `ClientMessage`,
   serialize, and `websocket.send(...)` the bytes.

On microphone change to `None` or session stop, fully tear down the
processor, close the `AudioContext`, and stop all tracks.

### 5.2 Video capture & streaming

Two paths sharing one `<video>` element:

- **Camera**: `getUserMedia({video: {deviceId: {exact: <id>}}})`.
- **Screen sharing**: `getDisplayMedia({video: true})`. Listen for the
  track's `onended` (the user revoked sharing via the browser UI) and
  reset the select to `None`.

Frame sampling:

- A hidden `<canvas>` is used to draw the current `<video>` frame at a
  **max dimension of 768 px** (preserve aspect ratio).
- Every **1000 ms** (1 fps; tunable), if the session is active and the
  WebSocket is open, draw the current frame, encode as JPEG via
  `canvas.toBlob('image/jpeg', 1)`, wrap as a `Blob` proto with
  `mime_type = "image/jpeg"`, set as
  `BidiGenerateContentRealtimeInput.video`, and send.
- Stop the interval on source change / session stop and tear down all
  tracks.

Note: the model consumes **sampled image frames**, NOT a video
container. A single canvas per frame is fine.

### 5.3 Text composer

When the user presses Enter / clicks Send:

1. Build a `BidiGenerateContentRealtimeInput.text = <user input>` and
   send it via the WebSocket as a `ClientMessage`.
2. Immediately render a user bubble locally (don't wait for an echo from
   the server).
3. Clear the input.

(Text via `realtimeInput.text` is the simplest path on Google AI; on
Gemini Enterprise raw WebSocket, switch to `clientContent.turns[].parts[].text`
with `turnComplete: true`. The UI does not need to expose this choice;
the underlying client implementation handles it.)

---

## 6. Session lifecycle

### 6.1 Start

When the user clicks `Start session` and no session is active:

1. Disable the button and the Settings button.
2. Build the setup JSON (§ 4) — abort with an error toast if the
   override JSON is invalid.
3. POST a small JSON envelope to a server endpoint (e.g. `/start`)
   containing:
   ```json
   {
     "session_id": "<UUID>",
     "endpoint_url": "<computed from Environment + Location, see § 4.1.3>",
     "setup": <setup JSON, with setup.model = computed fully-qualified
               name from § 4.2.2>
   }
   ```
4. On success (`{"status": "started"}`), open a WebSocket to the server
   (e.g. `wss://<host>/ws?session_id=<UUID>`) and:
   - Set `binaryType = 'arraybuffer'`.
   - Wire `onopen`, `onmessage`, `onerror`, `onclose` (§ 6.3).
5. Lazily create a single `AudioContext({sampleRate: 24000})` for
   playback (the model's output rate). Reuse it across sessions.
6. Update the status pill to "Session active" and change the button
   label to `Stop session`.

> The session ID is generated client-side once with `crypto.randomUUID()`
> and stays constant for the page's lifetime.

### 6.2 Stop

Clicking `Stop session` while active simply calls `websocket.close()`.
Cleanup happens in `onclose` so the same path runs whether the user
stopped the session or the server / network terminated it.

### 6.3 WebSocket message handling

- **Text frames** (rare; some servers emit them as out-of-band events):
  parse as JSON and route to the appropriate renderer (e.g. tool
  responses executed locally on the server).
- **Binary frames**: deserialize as
  `BidiGenerateContentServerMessage`. Route the inner oneof:
  - `serverContent.modelTurn.parts[]`: for each `inlineData` part with
    `mime_type` starting with `audio/`, enqueue a playback chunk
    (§ 7).
  - `serverContent.inputTranscription`: enqueue a `user` transcription
    event with `text` and `finished` flag.
  - `serverContent.outputTranscription`: enqueue a `model`
    transcription event with `text` and `finished` flag.
  - `serverContent.interrupted == true`: **flush** the playback queue
    immediately (drop any unplayed audio + pending transcription) and
    start a new bubble (§ 7.3).
  - `serverContent.turnComplete == true`: enqueue events that close
    the current `model` and `user` bubbles so the next turn starts
    fresh.
  - `toolCall.functionCalls[]`: enqueue tool-call bubbles (one per
    function call).
  - `toolCallCancellation.ids[]`: enqueue a tool-cancellation bubble.

### 6.4 onclose cleanup

On `WebSocket.onclose`, regardless of cause:

1. Tear down the audio capture pipeline (§ 5.1).
2. Stop all media streams (§ 5.2).
3. Reset the status pill to "Idle".
4. Re-enable the `Start session` and `Settings` buttons.
5. Reset the session button label to `Start session`.
6. If the session had actually started (i.e. it was not a failed
   handshake), and the recorder feature is enabled, schedule the
   save-recording modal to open after a short delay (~800 ms) so the
   server has time to flush the recording (§ 8).

---

## 7. Audio playback (model output)

### 7.1 Playback queue

Maintain a single FIFO queue (`playbackBuffer`) of mixed event types:

- `audio` — a chunk of model audio (raw PCM bytes + mime).
- `transcription` — text + role + finished flag.
- `newTranscriptionSignal` — close the current bubble for a role.
- `toolCall`, `toolCallCancellation`, `toolResponse` — UI events.

Why one queue for everything: it preserves the **order** in which audio
and transcription chunks arrived so they stay roughly time-aligned
visually. Without a single queue, fast text rendering can race ahead of
slow audio scheduling.

### 7.2 Drain loop

A `setInterval` running every ~10 ms peeks at the queue head:

- If it's an `audio` event:
  - If the playback context's lookahead is already > ~500 ms ahead of
    `currentTime`, **wait** (don't dequeue yet) — this avoids building
    an unbounded backlog when the model produces faster than realtime.
  - Otherwise dequeue and schedule playback (§ 7.4).
- Otherwise, dequeue and dispatch to the appropriate renderer
  (transcription / bubble close / tool event).

### 7.3 Interrupt handling

When the server sends `serverContent.interrupted = true`:

1. Immediately call `playbackBuffer.clear()` — drops any queued audio
   AND any queued transcription / tool events that arrived after the
   interrupt-causing user activity.
2. Subsequent transcription chunks belong to a fresh turn, so they will
   start fresh bubbles automatically (the previous bubbles for both
   roles have effectively been abandoned by clearing the queue).
3. Already-scheduled (but not yet played) `AudioBufferSourceNode`s
   continue playing — the queue clear only stops *future* scheduling.
   For a more aggressive flush, also recreate the `AudioContext` or
   call `.stop()` on a tracked list of in-flight sources.

### 7.4 PCM scheduling

The model emits 24 kHz, 16-bit, mono PCM. Per chunk:

```js
const pcm16 = new Int16Array(chunkBytes.buffer);
const f32 = new Float32Array(pcm16.length);
for (let i = 0; i < pcm16.length; i++) f32[i] = pcm16[i] / 32768.0;
const buf = ctx.createBuffer(1, f32.length, 24000);
buf.copyToChannel(f32, 0);
const src = ctx.createBufferSource();
src.buffer = buf;
src.connect(ctx.destination);
const startAt = Math.max(ctx.currentTime, nextBufferStartTime);
src.start(startAt);
nextBufferStartTime = startAt + buf.duration;
```

`nextBufferStartTime` is a per-context cursor that grows as each chunk
is scheduled. It guarantees back-to-back, gap-free playback even when
chunks arrive in bursts.

---

## 8. Save-recording dialog (optional)

If the implementation includes the recorder feature, surface this dialog
when a session ends (triggered from `onclose` per § 6.4).

A second modal (narrower, `max-width: 480px`) titled
**"Save session recording?"** with:

- A subtitle explaining that a log of every message exchanged is
  available on the server.
- A **filename** monospace input pre-filled with a sensible default
  (e.g. `liveapi_session_<YYYYMMDD>_<HHMMSS>.<ext>` where `<ext>`
  matches the on-disk format chosen per `message_recorder.md`).
- A help line clarifying that on Chromium-based browsers the user
  will get a native save-as dialog (folder picker), and on others the
  file lands in the default downloads folder.

Footer buttons:

- **Discard** (left, danger-tinted ghost) — POSTs to a server endpoint
  (e.g. `/recording/discard?session_id=...`) so the server can clean up
  the temp file. Closes the modal.
- **Cancel** — just closes the modal (the server keeps the file; the
  user can re-trigger save manually if your UI exposes that).
- **Save…** (primary) — § 8.1.

### 8.1 Save flow

1. Disable both Save and Discard buttons while saving.
2. If `window.showSaveFilePicker` is available (File System Access
   API):
   - Open the picker with `suggestedName` and an appropriate `types`
     entry (e.g. `application/octet-stream` + the chosen extension).
   - On `AbortError` (user cancelled), re-enable the buttons and leave
     the modal open.
   - Otherwise `fetch('/recording/download?session_id=...')` and pipe
     `response.body` into `fileHandle.createWritable()`.
3. Else, fall back to a regular browser download:
   - Create a hidden `<a href="/recording/download?...&filename=..."
     download="...">` and `.click()` it.
4. On success, POST `/recording/discard` to clean the server-side temp
   file, show a "Recording saved." status toast, and close the modal.
5. On error, show the error in the status banner; leave the modal open.

### 8.2 Integration with the viewer

If the recording viewer (see `recording_viewer.md`) is also deployed,
add a third footer action between Discard and Save:

- **Open in viewer** — opens the viewer URL in a new tab with the
  session's recording id as a query parameter so the viewer can
  auto-load it (or instruct the user to upload it after download).

---

## 9. Polish details (the "fancy" part)

These are small touches that make the difference between a barebones
test page and a playground that feels production-quality. Ship as many
as time allows.

- **Status pill** with a pulsing green dot for "active". Use a CSS
  keyframe `@keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity:
  .55 } }` and a soft `box-shadow` halo.
- **Modal pop-in animation**: `opacity: 0 → 1` with a small
  `translateY(8px) scale(.985) → 0,1` over ~180 ms with a spring-y
  cubic-bezier (e.g. `cubic-bezier(0.16, 1, 0.3, 1)`).
- **Backdrop blur**: `backdrop-filter: blur(4px)` on the modal
  backdrop.
- **Segmented control** for any small enum (e.g. response modality
  toggle, future backend selection). Pills with an active "card" look
  using a subtle shadow.
- **Focus rings**: every focusable input/button gets a 3 px soft
  accent-tinted focus ring (`box-shadow: 0 0 0 3px var(--accent-soft)`)
  via `:focus-visible`.
- **Smooth transitions** on hover / focus state changes (~120 ms).
- **Tool-call bubbles** with a colored badge and a monospace
  pretty-printed args block — far more useful than a plain text dump.
- **Dark-on-light typography** (or full dark mode) with
  `-webkit-font-smoothing: antialiased`.
- **Auto-grow textarea** for the composer, capped at a sensible max
  height.
- **Persist nothing** in localStorage by default; let the user start
  fresh on every page load. (If you do persist anything, make it the
  Settings form values, not the conversation.)

---

## 10. Required server companion (brief)

The frontend assumes a small companion HTTP server on the same origin.
Minimal endpoints (names are conventions; pick what fits your stack):

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`  | `/` and `/static/...`     | Serve the SPA. |
| `POST` | `/start`                  | Body: `{session_id, endpoint_url, setup}`. The server constructs a Live API client (using the implementation generated by the parent skill) bound to that `session_id`. Responds `{"status": "started"}` or HTTP 4xx/5xx with an error message. |
| `WS`   | `/ws?session_id=<UUID>`   | Bidirectional bridge. Browser sends serialized `ClientMessage`s; server forwards them to the upstream Live API. Server forwards upstream `ServerMessage`s as binary frames; out-of-band server-side events (e.g. local tool execution results) may be sent as JSON text frames. |
| `GET`  | `/recording/download?session_id=...` | Streams the recording for the given session. (Only when the recorder feature is enabled.) |
| `POST` | `/recording/discard?session_id=...`  | Deletes the server-side temp file. (Only when the recorder feature is enabled.) |

The frontend treats the server as opaque infrastructure — it does not
care about authentication, token refresh, or upstream endpoint routing.
Those concerns belong to the underlying Live API client implementation
(see the parent skill).

---

## 11. Responsive sizing checklist

Verify before shipping:

- At **1366×768**: header + status + Start/Settings buttons + at least
  the conversation panel header + the side rail's Configuration and
  Sources sections must all be visible **without page-level
  scrolling**. The video preview may scroll into view inside the side
  rail but the panel itself does not push the rest of the page off-
  screen.
- At **1920×1080**: comfortable margins; no oversized white space.
- Below **900 px wide**: the two-column main grid collapses to a
  single column; the side rail is capped at `max-height: 40vh` and
  scrolls.
- The settings modal never exceeds `calc(100vh - 48px)`; its body
  scrolls when content overflows.
- The save-recording modal stays at `max-width: 480px` and remains
  fully visible.
