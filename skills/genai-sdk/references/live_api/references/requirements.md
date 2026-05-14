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

### Recorder (per `message_recorder.md`)

-   The service class accepts the recorder as an **optional**
    constructor argument. When omitted, recording is disabled with zero
    runtime overhead.
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

-   Only useful when the recorder is also enabled.
-   If the user picks the viewer without the recorder, warn them they
    will have nothing to load.

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

For every web service produced, the agent MUST start it, open it in a
(headless) browser, and verify all of the following:

### Smoke checks (every service)

1.  Process starts without error and binds to the expected port.
2.  Page loads (HTTP 200) and primary content renders.
3.  Core controls are visible without page-level scrolling at
    1366×768.

### End-to-end interactions

-   **Test UI**: open a connection → send a text message → receive a
    response. Trigger an interrupt scenario and confirm playback is
    flushed.
-   **Recorder** (if enabled): run a session that records → close the
    recorder → the resulting file is non-empty and parseable as the
    chosen on-disk format.
-   **Viewer** (if enabled):
    1.  Load the shipped fixture
        `references/sample_recording.jsonl` by path, then load it
        again by upload.
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
-   **Recorder + Viewer end-to-end** (if both enabled): record a live
    session in the test UI → use the save dialog to open it in the
    viewer → server audio bars appear in playback mode as **wide
    duration bars** (not instant markers); the same bars in message
    mode collapse to the 4 px fixed-width form.

Report any failures and fix them before declaring the implementation
complete.
