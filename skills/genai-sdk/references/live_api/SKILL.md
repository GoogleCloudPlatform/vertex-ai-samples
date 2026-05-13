---
name: liveapi-service
description: Generates a LiveAPI client service class in the user's chosen programming language. Use when the user wants to build, scaffold, or integrate a client that connects to the Gemini LiveAPI websocket endpoint (Gemini Enterprise or non-Gemini Enterprise), handles session setup/resumption, bearer token refresh, and sending/receiving `ClientMessage`/`ServerMessage` protos.
---

# LiveAPI Service Skill

Provided files in `references`:

-   `client_server_messages.md`: The public document of protos used for LiveAPI.
-   `client_server_messages.proto`: The proto generated based on the
    `client_server_messages.md`.
-   `session_manager.md`: Describes how to correctly handle the sessions.
-   `message_recorder.md`: **Optional feature.** Spec for an asynchronous,
    non-blocking recorder that captures the bidirectional traffic
    (`ClientMessage` sent + `ServerMessage` received) of a Live API session
    to durable storage.
-   `recording_viewer.md`: **Optional feature.** Spec for a single-page web
    viewer that renders a recording produced by the recorder, with one
    interactive timeline per agent and two view modes (Playback vs.
    Message).
-   `interactive_ui.md`: Spec for the polished single-page browser
    playground used by Step 6's test UI — modal settings, mic/camera/
    screen-share streaming, chat-style transcript with tool-call bubbles,
    low-latency audio playback with interrupt handling, and an optional
    save-recording dialog. Follow this when implementing the frontend in
    Step 6.

What you should do:

Step 1:

Copy existing reference files to user provided destination folder.

The recorder/viewer reference files (`message_recorder.md`,
`recording_viewer.md`) only need to be copied if the user opted into those
optional features (see Step 2).

Step 2:

Examine the public documents mentioned in `client_server_messages.md`. Checking if
there are any discrepancies between the public documents and the created
markdown / proto as `client_server_messages`. If yes, update these file in the
destination folder

Step 3:

Implement a class in the user wanted coding language that work as a LiveAPI
service, it should import the existing proto file, build the connection to the
LiveAPI endpoint, expose functions to user and let user able to send and receive
data to / from the model.

If a language need a specific environment, such as python, you should create the
environment in the output folder and provide a bash file, by executing which,
the user can recreate the correct environment, do not use or modify the existing
system environment.

Wanted behavior:

The user will provide the following information to the class for initialization:

-   project_id
-   location
-   model_id
-   config, should be a `ClientMessage` with `setup` field.
-   use_gemini_enterprise, should be a boolean telling if using Gemini Enterprise or not
-   api_key, if not using Gemini Enterprise, an api_key should be provided.
-   **agent_name** (optional): a string identifier for this session, used by
    the recorder to disambiguate frames in multi-agent scenarios. Required
    when the recorder is enabled and a single recorder is shared across
    multiple service instances.
-   **recorder** (optional): a `MessageRecorder` instance. Only present if
    the user opted into the recorder feature in Step 2.

There are optional features such as recorder and recorder viewer. Ask the
user if they want to have these features. Concretely:

-   **Recorder**: implement per `message_recorder.md`. The session manager
    must accept the recorder as an **optional** constructor argument. When
    omitted, recording is disabled with zero runtime overhead. When
    supplied, on every successfully sent client frame and every received
    server frame, the session manager MUST build a fully-populated record
    (set the `payload` oneof arm, the `timestamp`, and the `agent_name`)
    **before** calling `recorder.record(...)` — the recorder treats the
    record as opaque. The session manager MUST NOT call `recorder.start()`
    or `recorder.close()`; the recorder's lifecycle is owned by the caller
    so a single recorder can be shared across multiple sessions. Recorder
    errors are best-effort and must never interrupt the session.

    To implement the recorder you also need to define a small wrapper proto
    for the on-disk record:

    ```proto
    message RecordedEvent {
      oneof payload {
        ClientMessage client_message = 1;  // sent by application
        ServerMessage server_message = 2;  // received from model
      }
      int64 timestamp_nanos = 3;
      string agent_name = 4;
    }
    ```

    Pick a portable on-disk format. Length-prefixed serialized protobuf
    is the recommended default; JSON Lines is acceptable for
    debuggability. **Do not** use Google-internal formats (e.g. recordio).
    Write **one record per message** — never batch multiple frames into a
    single wrapper proto (see `message_recorder.md` § "Per-record size").

-   **Recording viewer**: implement per `recording_viewer.md`. This is a
    thin HTTP server + a single-page frontend that loads a recording
    produced by the recorder and renders one interactive timeline per
    agent. The viewer MUST support both view modes specified in
    `recording_viewer.md` § 2.3:
    - **Playback mode** — when audio will start/end playing
      (cursor-pushed reconstruction).
    - **Message mode** — when each frame was actually sent / received on
      the wire (raw observation time).
    The viewer is most useful when the recorder is also enabled; if the
    user picks the viewer without the recorder, warn them that they will
    have nothing to load.

If using Gemini Enterprise, you should get a bearer token, refresh it when needed, and send it with
each websocket connection (including session resumption).

The class should expose the following functions to the user:

-   [async] send_realtime_data(data): allow the user to send realtime_data to
    the model. The `data` should be a `ClientMessage` in the proto file.
-   [async] send_client_content(data): allow the user to send non_realtime data
    to the model, allow the user to add context. The `data` should be a
    `ClientMessage` in the proto file.
-   [async] receive(): Allow the user to receive data from the model. The data
    received should be a `ServerMessage` in the proto file.

Step 4:

Once the code implemented, you should implement a test file, initialize the
connection and try to send `text`, `audio`, `video` data and receive the
response.

Ask the user for necessary information.

Step 5:

You should finally provide a markdown file with name `how_to_run.md`, describe
how to correctly use the class you just created. You should provide full example
about how to correctly build clientmessage for all kinds of support modalities
and how to send them. Also you should describe how to correctly fetch data from
the model.

Step 6:

You should create scripts to deploy your implementation as a service, it should
contains both frontend UI and backend service [You can use whatever coding
language you want]. In these service, the user can use the frontend UI to test
your implementation, it should allow the user to:

-   Start new connection / close current connection.
-   Select models to use.
-   Select input sources (audio or / and video [camera or screenshot]) and
    streaming data to model.
-   Send text message to model.
-   Heard the audio sound from model and see the model and user transcription
    and conversation history.

The frontend UI MUST follow the design and feature spec in
`interactive_ui.md` (polished header with status pill, modal settings
dialog that builds a `BidiGenerateContentSetup`, sidebar with config
summary + source selectors + video preview, chat-style conversation with
tool-call/response/cancellation bubbles, low-latency PCM playback with
interrupt-driven flush, and the responsive layout rules).

If the user selected to use the recorder, pop up a window and let the user
decide if they want to download the saved message file, which can be used
later for visualization.

If the user **also** selected the recording viewer, deploy the viewer
service alongside the test UI (separate port or sub-route is fine). The
download dialog should additionally expose an "Open in viewer" action that
loads the just-saved recording into the viewer in a new browser tab.

**Attention**

The service should reuse the `ServerMessage` and `ClientMessage` defined in the
proto for sending and receiving messages.

**Responsive layout (IMPORTANT).** All web frontends produced in this skill
(the test UI in Step 6 and the recording viewer in Step 8) MUST fit within
the browser viewport at common laptop resolutions (≥ 1366×768) without
requiring the user to scroll the *page* to see core controls and primary
content. Specifically:

-   The page-level layout must size to `100vh` / `100vw`; do not let the
    body itself become scrollable.
-   Use flexbox / grid with `overflow: auto` on **inner** panels (chat
    transcript, timeline, log panel, etc.) so that long content scrolls
    *inside* its panel rather than pushing the rest of the UI off-screen.
-   Avoid fixed pixel sizes on top-level containers; prefer relative units
    (`vh`, `vw`, `%`, `fr`) so the layout adapts to the actual window
    size.
-   Test at a few viewport sizes (e.g. 1366×768 and 1920×1080); the
    header, primary action buttons, video/audio preview, and at least
    part of the transcript / timeline must all be visible without
    scrolling the page.

This rule applies even when there are multiple panels (input controls,
transcript, audio waveform, debug log, etc.) — split the viewport between
them with scrollable inner regions instead of stacking them into a tall
page.

**Test every web service after implementation.** This skill may produce
multiple web services (the test UI in Step 6, and optionally the recording
viewer in Step 8 — possibly running on different ports / sub-routes).
After implementation, the coding agent MUST start each service, open it in
a (headless) browser, and verify that:

1.  The service starts without error and binds to the expected port.
2.  The page loads (HTTP 200) and the primary content renders.
3.  Core interactive controls are visible without scrolling the page at
    1366×768.
4.  At least one end-to-end interaction works (e.g. for the test UI:
    open a connection, send a text message, receive a response; for the
    viewer: load a sample recording, switch between Playback and
    Message modes).

Report any failures and fix them before declaring the implementation
complete.

While implementing the audio / transcription playback logic, please follow the
instruction in
https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/live-api/best-practices.

Make sure you correctly handle the `interrupt` signal from `ServerMessage`,
which should:

-   You'll receive audio and transcription interleaved. The played audio and
    corresponding transcription should be time aligned.
-   Immediately stop the playing for audio and transcription.
-   Clear the playback buffer to dump unsent audio / transcription.
-   Start new chat bubbles for model / user.

Make sure you correctly handle the `finished` signal from `input_transcription`
or `output_transcription`, which should start a new bubble after concatenating the
data.

Step 7: Implement a description file `how_to_test_with_ui.md` and tell how to
start the services, which URL should the user use and how to interactive with
the model.

Step 8: If the user opted into the recording viewer, write a brief
`how_to_use_viewer.md` that describes:

-   How to launch the viewer service and which URL to open.
-   How to load a recording (either by path or by uploading a downloaded
    file).
-   How to switch between the **Playback** and **Message** view modes
    (global toggle in the top header, plus per-agent overrides), and what
    each mode shows (Playback = when audio will start/end playing;
    Message = exact send/receive wire times — does NOT show when the
    message will be played).
-   How to inspect a single message (hover for a quick look, click to pin
    the panel) and how to play a single audio chunk vs. the full
    timeline.
