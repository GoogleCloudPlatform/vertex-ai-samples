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
-   The chat UI's save-recording dialog MUST expose an
    **"Open in viewer"** action that switches the sidebar to the
    viewer entry and loads the just-saved recording via the shared
    backend's viewer endpoints (no new tab, no second backend).

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
-   **Recorder + Viewer end-to-end** (when the chat UI is enabled):
    record a live session in the chat UI → click "Open in viewer" in
    the save-recording dialog → the sidebar switches to the viewer
    entry **in the same tab and same backend** and the just-saved
    recording is already loaded → server audio bars appear in
    playback mode as **wide duration bars** (not instant markers);
    the same bars in message mode collapse to the 4 px fixed-width
    form.

Report any failures and fix them before declaring the implementation
complete.
