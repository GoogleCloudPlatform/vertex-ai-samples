---
name: liveapi-cross-cutting-requirements
description: >-
  Cross-cutting requirements for the LiveAPI service skill. These rules
  apply to every step of the workflow (class implementation, backend,
  frontend, recorder, viewer) and must be re-read before declaring any
  deliverable complete.
---

# Cross-cutting requirements

These rules apply at every step of the LiveAPI service skill workflow.
The agent should re-read this file before declaring any step complete.

## Reuse the proto types

All client / server traffic — in the service class, the backend bridge,
and the frontend — MUST use the `ClientMessage` and `ServerMessage`
defined in `client_server_messages.proto`. No ad-hoc parallel structs,
no hand-rolled JSON shapes for messages already defined in the proto.

## Optional features integration

### Feature bundling

The testing chat UI, the recorder, and the recording viewer are a
**single bundled option**, not three independent toggles. The Step 1
interview asks one question: "do you want the testing chat UI?".

-   **User answers yes:** the agent MUST produce the chat UI, the
    recorder, and the viewer together. The viewer is served from the
    same backend process and same port as the chat UI, behind a shared
    sidebar (see "Unified backend & sidebar" below).
-   **User answers no:** the agent MUST NOT produce any of the three.
    The deliverable is just the service class plus the smoke check.

There is no supported configuration where the chat UI exists without
the recorder, or the viewer exists without the recorder, or the
recorder/viewer exist without the chat UI.

### Unified backend & sidebar (when the chat UI is enabled)

-   A **single** backend process serves both the chat UI and the
    viewer on the same port. Do not spin up a second process or a
    second port for the viewer.
-   The frontend is a shared shell with a persistent left sidebar
    containing at least two entries — "Chat UI" and "Recording
    viewer" — that swap the main content area without a full page
    reload. The active entry is visually marked.
-   When the chat session ends, the chat UI shows an
    **informational session-ended modal** (no filename input, no
    save action, no download action) that includes an
    **"Open Recording viewer"** button. That button switches the
    sidebar to the viewer entry and auto-loads the recording the
    backend just saved (via `#/viewer?recording=<name>`).

### Recordings directory alignment

-   The backend takes a single `--recordings_dir` startup argument.
-   On websocket close the chat UI calls
    `POST /recording/finalize`; the backend auto-saves the session's
    recording into `--recordings_dir` under a server-chosen filename
    and returns the resulting `name`. The chat UI does not expose a
    user-chosen filename, a save button, or a download action.
-   The viewer's `GET /api/recordings`,
    `GET /api/recordings/download`, and `POST /api/load` all read
    from the same directory. **Downloads are exposed only through
    the viewer.**
-   The viewer frontend MUST present a left **side panel** populated
    from `GET /api/recordings`, with each row supporting load +
    download, plus a refresh button and an upload action. It MUST
    NOT expose a free-form filesystem path input.
-   `POST /api/load`, `GET /api/recordings/download`, and
    `POST /recording/finalize` MUST reject any name containing path
    separators or `..` to prevent directory traversal.

### Recorder (per `message_recorder.md`)

-   The service class accepts the recorder as an **optional**
    constructor argument. When omitted, recording is disabled with zero
    runtime overhead. (Per the bundling rule above, the recorder is
    omitted exactly when the chat UI is not requested.)
-   The class owns building each record before calling
    `recorder.record(...)`: set the appropriate `payload` oneof arm
    (client or server message), the `timestamp`, and the `agent_name`.
-   The class MUST NOT call `recorder.start()` or `recorder.close()`.
    The recorder's lifecycle belongs to the caller so a single recorder
    can be shared across multiple sessions.
-   Recorder errors are best-effort and must never interrupt the
    session.
-   On-disk format: length-prefixed serialized protobuf (recommended)
    or JSON Lines. Do **not** use Google-internal formats (e.g.
    recordio). Write **one record per message** — never batch.

### Viewer (per `recording_viewer.md`)

-   Always produced together with the chat UI per the bundling rule
    above; never produced standalone.
-   Served from the same backend process and same port as the chat UI,
    reachable via the shared sidebar.

## Audio / transcription playback

Follow the public best-practices guide:
https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/live-api/best-practices

For interrupt handling and streaming transcription bubbles, follow the
spec in `interactive_ui.md` (single playback FIFO queue, flush on
`serverContent.interrupted`, close bubble on `finished` /
`turnComplete`).

## Responsive layout

All web frontends produced by this skill (test UI, recording viewer)
MUST fit within the browser viewport at common laptop resolutions
(≥ 1366×768) without page-level scrolling.

-   Page-level layout sized to `100vh` / `100vw`; `body` is not
    scrollable.
-   Inner panels (chat transcript, timeline, log panel) use
    `overflow: auto`; long content scrolls **inside** its panel rather
    than pushing the rest of the UI off-screen.
-   Top-level containers use relative units (`vh`, `vw`, `%`, `fr`),
    not fixed pixels.
-   Verify at 1366×768 and 1920×1080 — header, primary action buttons,
    preview, and at least part of the transcript / timeline must be
    visible without scrolling the page.

This rule applies even when there are multiple panels — split the
viewport between them with scrollable inner regions instead of
stacking them into a tall page.

## Verification (run during Step 8)

For the unified backend produced when the chat UI is enabled, the
agent MUST start it, open it in a (headless) browser, and verify all
of the following. (When the chat UI is not enabled, only the service
class smoke check from Step 5 applies.)

### Smoke checks (every service)

1.  Process starts without error and binds to the expected port.
2.  Page loads (HTTP 200) and primary content renders.
3.  Core controls are visible without page-level scrolling at
    1366×768.
4.  The shared sidebar renders with at least the "Chat UI" and
    "Recording viewer" entries; clicking each entry swaps the main
    content area to the corresponding surface without a full page
    reload, and the active entry is visually marked.
5.  On the Recording viewer surface, the left side panel of
    recordings, the refresh button, and the upload action are all
    visible without page-level scrolling at 1366×768.

### End-to-end interactions

-   **Test UI**: open a connection → send a text message → receive a
    response. Trigger an interrupt scenario and confirm playback is
    flushed.
-   **Recorder** (if enabled): run a session that records → close the
    recorder → the resulting file is non-empty and parseable as the
    chosen on-disk format.
-   **Viewer** (if enabled):
    1.  Drop the shipped fixture `references/sample_recording.pb`
        into the backend's `--recordings_dir`; verify it appears as
        a row in the left side panel (sourced from
        `GET /api/recordings`), that clicking the row loads it, and
        that clicking the row's Download button streams the same
        bytes back. Then load it again via Upload.
    2.  Switch the global toggle and at least one per-agent toggle
        between Playback and Message modes.
    3.  Click a server audio bar and pin its tooltip; confirm the
        proto detail renders.
    4.  **Playback ≠ Message assertion** — fetch `/api/agents` for the
        loaded fixture and verify, programmatically or by direct
        inspection, that **all** of these hold:
        -   For every agent, `total_ms(playback) >= total_ms(message)`.
        -   At least one server-audio message satisfies
            `end_ms - start_ms >= 50` ms (i.e., it renders as a wide
            duration bar in playback mode, not as a 2 px instant
            marker).
        -   For the fixture's interrupt section, at least one
            server-audio message has `start_ms != wire_ms` (the
            interrupt rewind shifted at least one subsequent chunk).
        -   No message with `modality == "audio"` has
            `end_ms == start_ms` (the audio-duration invariant from
            `recording_viewer.md` § Audio duration).
    5.  If any of these fails, the bug is in the viewer server's
        reconstruction pipeline, not the frontend. See
        `recording_viewer.md` § Diagnostics for the five known
        failure modes.
-   **Recorder + Viewer end-to-end** (when the chat UI is enabled):
    record a live session in the chat UI → end the session and
    confirm the informational session-ended modal appears with the
    server-assigned filename and **no** download or save controls →
    click "Open Recording viewer" → the sidebar switches to the
    viewer entry **in the same tab and same backend**, the
    just-saved recording is the active row in the left side panel,
    and it is auto-loaded → the row's Download button streams the
    `.pb` back to the user → server audio bars appear in playback
    mode as **wide duration bars** (not instant markers); the same
    bars in message mode collapse to the 4 px fixed-width form.

Report any failures and fix them before declaring the implementation
complete.

## Silent failures to defend against

These are the failure modes that have repeatedly bitten previous
implementations of this skill in new languages or new proto stacks.
Each one shares the same shape: a layer assumes an environment the
next layer doesn't provide, and the result is a *successful-looking*
empty payload rather than a loud error. Re-read this list before
declaring **any** of Steps 4 – 8 complete.

### 1. Permissive proto-JSON parsing of an outer envelope

**Trigger.** A backend uses `discard_unknown=true` (or any
unknown-tolerant decoder) on the `/start` body's `setup` field and
treats the result as a `ClientMessage`. The browser POSTs a *bare*
`BidiGenerateContentSetup` (per `interactive_ui.md` § "Wire shapes"),
which has no field that matches any `ClientMessage` arm; the decoder
quietly produces an empty `ClientMessage{}` and the backend opens an
upstream session with no setup, which Vertex AI then closes with a
cryptic policy violation.

**Defense.** Always validate the parsed message *after* the
permissive parse: assert that the expected oneof arm (`setup`) is
populated and that `setup.model` is non-empty after model-name
reconstruction. Reject with HTTP 400 + a human-readable message
before opening the upstream WebSocket. (See `interactive_ui.md` §
"Strict-parse rule for `/start`".)

### 2. Lazy proto-type resolution in reflective JS bindings

**Trigger.** A frontend uses `protobufjs` (or any other reflective
runtime) to build proxy classes whose setters branch on
`field.resolvedType` (`isMessage`-style checks). Protobufjs resolves
message-typed references **lazily**, so iterating `Type.fields`
immediately after `protobuf.load()` returns fields whose
`resolvedType` is still `null`. Every message-typed setter then falls
through to the primitive branch, storing the wrapper *itself* on the
parent and producing zero-byte payloads on the wire.

**Defense.** Call `root.resolveAll()` (or the equivalent in your
runtime) *before* iterating fields to build accessors. Add a JS
round-trip test that serializes a populated `ClientMessage` through
the shim and asserts the encoded byte length is greater than the
single-arm-tag-plus-zero-length stub (~2–3 bytes); a passing test
proves at least one message-typed field was actually written. (See
`proto-shim.md`.)

### 3. Wire-format ambiguity between binary and proto-JSON

**Trigger.** The Live API's WebSocket bridge silently rejects raw
binary `ClientMessage` frames (close code 1007 with text "Invalid
JSON payload received…"), but only when the JSON-encoded form
*happens* to start with bytes that look like JSON. Implementations
that send `proto.Marshal(...)` straight onto the wire see the
session close right after `setup` with a misleading error pointing at
the model resource name.

**Defense.** Send WebSocket TEXT frames containing proto-JSON
(`protojson.Marshal` / `protobuf.util.toJSON` / equivalents) for
every frame sent to Vertex. Accept TEXT proto-JSON for receive (with
a binary-protobuf fallback for forward compat). Document the
encoding choice in the session manager's comments so a future change
isn't accidental.

### 4. Coarse `try/catch` around event handlers

**Trigger.** Reference frontends commonly wrap the entire
`ws.onmessage` body in one `try { ... } catch (e) { console.error(...) }`.
A single bug in any concern (e.g. a missing `Transcription.finished`
accessor) throws on the first frame and silences every subsequent
concern — transcription, audio playback, *and* tool calls all stop
rendering together. The user sees a UI that does nothing while frames
flow normally on the wire.

**Defense.** Scope each `try/catch` to one concern (tool calls,
audio, transcription, turn lifecycle). On the first error in any
concern, render a **visible** banner at the top of the page so the
failure is obvious without opening dev-tools. Let unaffected
concerns keep running. The reference `playground_frontend/script.js`
implements this pattern; adaptations MUST preserve it.

### 5. Accessor-return-type drift between proto runtimes

**Trigger.** The reference script expects `Blob.getData()` to return
an object with a `.buffer` ArrayBuffer (legacy google-protobuf JS
API). A different runtime (e.g. protobufjs) returns a `Uint8Array`
directly. The audio code path computes `new Int16Array(audioData.buffer)`
on what's actually `Uint8Array.buffer` — which happens to work in some
shapes and silently produces nothing in others, depending on
`byteOffset` alignment.

**Defense.** Wrap the audio decoder in a shape adapter that handles
`Uint8Array`, `ArrayBuffer`, generic `ArrayBuffer.isView` results,
and the legacy `{buffer: ArrayBuffer}` shape. Bail with a visible
banner on unknown shapes rather than computing on garbage. The
reference `playAudioChunk` in `playground_frontend/script.js` is the
template.

### 6. CDN dependencies in the browser

**Trigger.** Frontend pulls protobufjs (or any other runtime) from a
public CDN. The CDN is blocked by a corporate proxy or just slow,
`window.protoReady` never resolves, and the page presents as a blank
spinner with no error message.

**Defense.** Vendor every runtime dependency under the static-asset
tree (`web/proto/protobuf.min.js`, etc.). Add a visible banner on
any `await window.protoReady` rejection so the failure surfaces in
the UI even if dev-tools are closed. Never depend on
`https://cdn.example.com/...` at runtime for the chat UI to load.

### 7. Server-message oneof violations

**Trigger.** The Live API public docs describe `ServerMessage` as a
strict oneof, but Vertex AI in practice bundles multiple top-level
fields into a single end-of-turn frame (typically `serverContent` +
`usageMetadata`). A strict-oneof decoder rejects these frames with
"oneof is already set"; a strict-arm `switch` on `msg.MessageType`
silently ignores the second arm.

**Defense.** Model server-side top-level fields as independently
optional in your generated bindings. Process each populated field
with `if msg.GetServerContent() != nil { ... }` style checks rather
than a `switch` over a oneof tag. Mirror the same shape in your
generated proto. (See `client_server_messages.md` § 2 and the
`ServerMessage` definition in `client_server_messages.proto`.)

### Verification: must-pass before declaring Step 8 complete

A frontend implementation MUST satisfy all of the following or it is
not done:

1. Loading the chat UI with the browser console closed produces
   either a working UI or a visible red banner. There is no third
   state.
2. Killing the network mid-load (or pointing the CDN URL at an
   unreachable host) reproduces the banner, not a blank page.
3. Sending a text message via the composer produces a populated
   `clientContent` record in `/api/recordings` (NOT
   `clientContent {}`). The record's `text_preview` contains the
   typed text.
4. Receiving a model response renders a model bubble in the
   conversation panel (TEXT modality) or plays audible audio (AUDIO
   modality) — never both blank.
5. Triggering an error in one concern (e.g. by sending malformed
   bytes through the WS) does not stop subsequent frames from
   rendering through the other concerns.

Failures of any of (1)–(5) almost always trace back to one of the
seven patterns above; consult the matching subsection before
attempting a fix.
