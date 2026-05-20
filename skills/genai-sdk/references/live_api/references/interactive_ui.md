---
name: live-api-interactive-ui
description: >-
  Reference implementation for a single-page browser playground for the Gemini
  Live API over WebSockets. The agent should adapt the provided reference
  frontend files to the target project rather than building from scratch.
---

# Live API Interactive UI

A complete reference frontend is provided in `playground_frontend/` alongside this file:

- **`playground_frontend/index.html`** — Page structure: header with status indicator,
  conversation panel with chat bubbles, side rail with config summary /
  media sources / video preview, settings modal, and an informational
  session-ended notice modal.
- **`playground_frontend/script.js`** — Session lifecycle, WebSocket messaging,
  audio/video capture pipelines, PCM playback with interrupt handling,
  conversation rendering (transcription + tool-call bubbles).
- **`playground_frontend/style.css`** — Design tokens, layout grid, bubble styles,
  modal animations, responsive breakpoints.

**The agent should use these reference files as the starting point and rewrite 
them to fit the target project's build system, module format, and proto
library.** Do not generate the UI from scratch — adapt the reference.

## Shared shell with the recording viewer

When this skill produces the testing chat UI, it always also produces
the recording viewer (see `recording_viewer.md`) and serves both from
the **same backend process and same port**. The chat UI described here
is therefore not a standalone page — it is one of two surfaces hosted
inside a shared application shell that renders a persistent **left
sidebar** with at least two entries: "Chat UI" and "Recording viewer".
Selecting a sidebar entry swaps the main content area to the
corresponding surface without a full page reload.

The shell, the sidebar, and the routing between the two surfaces are
the agent's responsibility. The reference frontend files here
(`playground_frontend/`) only cover the chat surface itself; they
must be embedded into the shared shell when adapted.

When the session ends, the chat UI shows an **informational
session-ended modal** (no save controls, no download button) that
includes an **"Open Recording viewer"** action. That action switches
the sidebar to the viewer entry and auto-loads the recording the
backend just persisted (no new tab, no second backend). Downloads
are exposed only inside the viewer's side panel.

---

## Computed endpoint URL (Environment + Location)

The user does **not** type the WebSocket endpoint URL directly. Instead
they pick an **Environment** and a **Location**; the UI computes the
endpoint and displays it as **read-only** so the user can see what will
be used without being able to accidentally break it.

### Endpoint computation logic

```
HOST_SUFFIX = {
  prod:     "aiplatform.googleapis.com",
  staging:  "staging-aiplatform.sandbox.googleapis.com",
  autopush: "autopush-aiplatform.sandbox.googleapis.com",
}

host = HOST_SUFFIX[environment]                          # if location == "global"
host = f"{location}-{HOST_SUFFIX[environment]}"          # otherwise

endpoint = f"wss://{host}/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"
```

Examples:

| Environment | Location       | Computed endpoint |
| ----------- | -------------- | --- |
| `prod`      | `us-central1`  | `wss://us-central1-aiplatform.googleapis.com/ws/...` |
| `prod`      | `global`       | `wss://aiplatform.googleapis.com/ws/...` |
| `staging`   | `europe-west4` | `wss://europe-west4-staging-aiplatform.sandbox.googleapis.com/ws/...` |
| `autopush`  | `us-central1`  | `wss://us-central1-autopush-aiplatform.sandbox.googleapis.com/ws/...` |

The endpoint updates **live** as the user changes Environment or Location
in the settings modal and is shown in both the modal and the sidebar
Configuration summary.

---

## Model name reconstruction (backend responsibility)

The frontend only collects the **short** model ID from the user (e.g.
`gemini-live-2.5-flash-native-audio`) and sends it as `setup.model` in
the `/start` POST body. **The backend server is responsible for
reconstructing the fully-qualified model resource name** using its own
project ID and the location parsed from the endpoint URL.

### Backend reconstruction logic

The backend service should accept a `--project_id` (or equivalent)
startup argument and build the full model name before forwarding the
setup to the upstream Live API:

```
model = f"projects/{project_id}/locations/{location}/publishers/google/models/{setup.model}"
```

- `project_id` — a startup argument to the backend service (NOT a
  user-facing UI field).
- `location` — extracted from the `endpoint_url` sent by the frontend
  (the first segment before the host suffix), or from a separate field
  if the server design prefers that.
- `setup.model` — the short model ID received from the frontend.

This keeps the project ID out of the browser and avoids exposing it in
the UI.

---

## What to adapt

When integrating the reference into a new project, the agent should adjust:

1. **Module system** — The reference uses inline `<script>` with proto
   classes assumed global. Replace with the project's module system
   (ES modules, Closure, bundler, etc.) and import the correct proto
   definitions for `BidiGenerateContentClientMessage`,
   `BidiGenerateContentServerMessage`, `BidiGenerateContentRealtimeInput`,
   and `Blob`.

2. **Environment / Location defaults** — Update `DEFAULT_ENV` and
   `DEFAULT_LOCATION` to match the target deployment. The
   `HOST_SUFFIX_BY_ENV` map and `computeEndpointUrl()` function
   contain the derivation logic.

3. **Model ID** — Update `DEFAULT_MODEL` to the desired short model
   name. The backend server reconstructs the fully-qualified resource
   path (`projects/{project}/locations/{location}/publishers/{model_id}`)
   using its `--project_id` startup argument.

4. **Server companion endpoints** — The frontend assumes a companion HTTP
   server on the same origin with these endpoints:

   | Method | Path | Purpose |
   | --- | --- | --- |
   | `POST` | `/start` | Body: `{session_id, endpoint_url, setup}`. The `endpoint_url` is the computed WebSocket endpoint. `setup.model` is the **short** model ID; the backend reconstructs the fully-qualified name using its `--project_id` argument. Returns `{"status": "started"}`. |
   | `WS` | `/ws?session_id=<UUID>` | Bidirectional bridge. Browser sends serialized `ClientMessage`s; server forwards to upstream. Server forwards `ServerMessage`s as binary frames. |
   | `POST` | `/recording/finalize` | Multipart `session_id=...`. Called by the chat UI when the websocket has closed. The backend **automatically persists** the session's recording into the recordings directory under a server-chosen filename (e.g. `liveapi_session_<timestamp>.pb`) and returns `{"name": "<basename>"}`. If the session produced no recording (recorder disabled, zero frames, etc.) the response is `{"name": ""}`. |
   | `POST` | `/recording/discard` | Multipart `session_id=...`. Deletes the server-side temp recording file. Used when the user explicitly declines to save. |

   The chat UI **does NOT** provide a download endpoint or a download
   button. Downloads are owned exclusively by the Recording viewer (see
   `recording_viewer.md`'s `GET /api/recordings/download`).

   The **same** backend process MUST also expose the recording-viewer
   endpoints documented in `recording_viewer.md`
   (`GET /api/agents`, `GET /api/audio/<idx>.wav`,
   `GET /api/recordings`, `GET /api/recordings/download`,
   `POST /api/load`, `POST /api/upload`) so the sidebar can swap
   between the chat and viewer surfaces without leaving the origin or
   hitting a second process. Sub-routing the static assets (e.g.
   `/chat/*` vs `/viewer/*`) is encouraged to keep the two surfaces'
   files organized.

   **Recordings directory alignment (REQUIRED).** The directory
   `POST /recording/finalize` writes to MUST be the same directory
   `GET /api/recordings` lists from and `POST /api/load` reads from.
   The user therefore never types a filesystem path in the viewer —
   the chat UI's auto-save and the viewer's recording picker share
   the same on-disk location. The backend takes the directory as a
   startup argument (e.g. `--recordings_dir`).

5. **Proto serialization** — The reference calls
   `ServerMessage.deserializeBinary()` and `msg.serializeBinary()`. Adapt
   to whatever proto library the project uses.

6. **Session-ended notice modal** — Always present when this skill
   produces the chat UI (because the recorder is bundled with the
   chat UI in that case). The modal is **purely informational** — it
   has no filename input, no "save" action, and no "download" action.

   On websocket close, the chat UI:

   1.  Calls `POST /recording/finalize`; the backend auto-saves the
       recording into the recordings directory under a server-chosen
       filename and returns the resulting `name`.
   2.  Opens a modal that says the session ended, displays the
       server-assigned filename, and tells the user to use the
       **Recording viewer** (sidebar) to review the conversation and
       to download the raw `.pb` if needed.

   The modal MUST include a primary **"Open Recording viewer"**
   button that switches the sidebar to the viewer entry with the
   just-saved recording preselected (navigate to
   `#/viewer?recording=<name>`; the viewer's initial load honors that
   query and auto-loads the named recording). A secondary
   **"Stay here"** button just closes the modal.

   Downloads are not exposed on the chat surface — they live in the
   viewer (see `recording_viewer.md`).

---

## Proto conformance check (IMPORTANT)

The reference `buildSetupJson()` in `script.js` constructs a JSON object
that must match the `BidiGenerateContentSetup` proto message (using
proto3 JSON camelCase field names). A comment block above the function
lists the full field hierarchy.

**Before shipping, the agent MUST compare the setup JSON structure
against the actual proto definition being used in the project.** The
proto may have evolved since this reference was written -- fields may
have been added, renamed, restructured, or deprecated.

Specifically, the agent should:

1. Read the project's proto definition for `BidiGenerateContentSetup`
   and all nested messages (`GenerationConfig`, `SpeechConfig`,
   `RealtimeInputConfig`, `AutomaticActivityDetection`,
   `ContextWindowCompressionConfig`, `ThinkingConfig`, etc.).
2. Compare every field name and nesting level against the reference
   `buildSetupJson()` output. Verify that camelCase JSON names match
   the proto's snake_case field names per proto3 JSON encoding rules.
3. Check for **new proto fields** not present in the reference UI. If
   a new field is useful for a testing playground, add a corresponding
   UI control in the settings modal and wire it into `buildSetupJson()`.
4. Check for **removed or renamed fields** in the proto. Remove or
   rename the corresponding UI controls and JSON keys.
5. Check enum values (e.g. `ActivityHandling`, `TurnCoverage`,
   `MediaResolution`, `StartSensitivity`, `EndSensitivity`) -- the
   set of valid values may have changed.

If mismatches are found, the settings modal HTML and the
`buildSetupJson()` function in `script.js` must be updated to match
the proto.

---

## Key architecture patterns in the reference

These patterns are intentional and should be preserved when adapting:

- **Single playback FIFO queue** — Audio chunks, transcription events, and
  tool-call events share one queue to preserve arrival order and keep audio
  and text roughly time-aligned.

- **Drain loop with backpressure** — A 10ms `setInterval` peeks at the
  queue head; audio chunks are held if the playback lookahead exceeds
  ~500ms to avoid unbounded backlog.

- **Interrupt handling** — On `serverContent.interrupted`, the queue is
  flushed immediately so stale audio/text is dropped.

- **Streaming transcription bubbles** — A `currentBubbles` map keyed by
  role tracks the active bubble. Text is appended incrementally; when receiving
  `finished` or `turnComplete`, the server bubble should be closed immediately. 
  The user bubble should close when receiving `finished` or a text message is sent.


- **PCM scheduling** — Model audio is 24kHz/16-bit/mono. Chunks are
  converted to Float32, scheduled back-to-back via
  `AudioBufferSourceNode.start(nextBufferStartTime)` for gap-free playback.

- **Audio capture** — Microphone input is captured at 16kHz/mono via
  `ScriptProcessorNode`, converted to 16-bit PCM, wrapped in a
  `BidiGenerateContentRealtimeInput.audio` proto, and sent as binary.

- **Video frame capture** — Camera or screen frames are sampled at 1fps,
  drawn to a hidden canvas (max 768px), encoded as JPEG, and sent as
  `realtimeInput.video`.
