---
name: liveapi-service
description: Generates a LiveAPI client service class in the user's chosen programming language. Use when the user wants to build, scaffold, or integrate a client that connects to the Gemini LiveAPI websocket endpoint (Gemini Enterprise or non-Gemini Enterprise), handles session setup/resumption, bearer token refresh, and sending/receiving `ClientMessage`/`ServerMessage` protos.
---

# LiveAPI Service Skill

## References

Files in `references/`:

-   `client_server_messages.md` + `client_server_messages.proto` —
    Public LiveAPI proto definitions and the generated proto file.
-   `session_manager.md` — Session handling spec (setup, resumption,
    bearer token refresh).
-   `interactive_ui.md` — Spec + reference frontend for the test UI
    (Step 7).
-   `requirements.md` — **Cross-cutting requirements that apply at
    every step.** Re-read before declaring any step complete.
-   `message_recorder.md` — **Optional.** Spec for the async,
    non-blocking recorder of bidirectional Live API traffic.
-   `recording_viewer.md` — **Optional.** Spec for the single-page web
    viewer of recordings produced by the recorder.


## Class contract

The generated service class is initialized with:

| Field | Notes |
| --- | --- |
| `project_id` | Runtime input. |
| `location` | Runtime input. |
| `model_id` | Runtime input. |
| `config` | A `ClientMessage` whose `setup` field is populated. |
| `use_gemini_enterprise` | Boolean. Selects auth + endpoint. |
| `api_key` | Required when `use_gemini_enterprise = false`. |
| `agent_name` (optional) | String identifier; **required** when a recorder is shared across multiple service instances. |
| `recorder` (optional) | A `MessageRecorder` instance. Only when the recorder feature was selected. |

The class exposes:

-   `[async] send_realtime_data(data: ClientMessage)` — send realtime
    input.
-   `[async] send_client_content(data: ClientMessage)` — send
    non-realtime context / turns.
-   `[async] receive() -> ServerMessage` — receive one server frame.

When `use_gemini_enterprise = true`, fetch and refresh the bearer token
and attach it to every websocket connection (including session
resumption).

## Workflow

Every step ends with a **Definition of done** — explicit criteria that
must be true before the agent moves on. The cross-cutting rules in
`references/requirements.md` apply throughout.

### Step 1 — Interview the user

Collect upfront, do not assume:

-   Destination folder for the generated project.
-   Target programming language.
-   Whether to enable the **testing chat UI** (optional). When the user
    says **yes**, the **recorder** and the **recording viewer** are
    enabled automatically as part of the same deliverable — they are
    not separately optional in this mode. Confirm this bundling with
    the user so they know what they are getting. When the user says
    **no**, skip the chat UI, the recorder, and the viewer entirely.
-   Whether the deployment will use **Gemini Enterprise** (affects
    auth + endpoint).
-   Whether the user can provide an project_id for testing purpose.

**Definition of done:** every choice above is captured in the agent's
plan; no implicit defaults remain. If the chat UI was selected, the
plan explicitly records that the recorder and viewer will be built
alongside it.

### Step 2 — Copy references

Copy the the whole references folder to the destination. 

**Definition of done:** All files and folders are copied to the destination

### Step 3 — Sync the proto

Reconcile `client_server_messages.md` against the public source
documents it cites. If there is drift:

1.  Update the markdown in the copied folder.
2.  Update `client_server_messages.proto` to match.
3.  Regenerate the language-specific bindings the service class will
    import.

**Definition of done:** markdown, `.proto`, and generated bindings all
agree, and the bindings compile in the target language.

### Step 4 — Implement the LiveAPI service class

Implement per the **Class contract** section, importing the generated
proto. If the language requires an isolated environment (Python, Node,
etc.), provision it inside the destination folder and provide a
one-line activation script. Do **not** modify the user's system
environment.

If the recorder feature was selected, wire the recorder hook into both
send paths and the receive path per the recorder integration rules in
`references/requirements.md`.

**Definition of done:** the class compiles / imports, exposes all
methods listed in the contract, accepts the optional `recorder` /
`agent_name` arguments, and raises a clear error when required
credentials are missing.

### Step 5 — Smoke check

Write a minimal end-to-end script that opens a session, sends one text
turn, receives one response, and exits cleanly.

-   Reads credentials / project_id / location / model from environment
    variables or CLI flags so it is runnable without code edits.
-   Prints a clear pass / fail line and exits non-zero on failure.
-   No baked-in secrets.

**Definition of done:** the script runs to completion against a real
endpoint, prints `PASS`, and exits 0.

### Step 6 — Backend service

Build a **single** HTTP + WebSocket backend that exposes both the
testing chat UI and the recording viewer from the **same process and
same port**. The backend MUST serve:

-   The chat UI endpoints from `interactive_ui.md` (`POST /start`,
    `WS /ws`, plus the optional recording download / discard
    endpoints).
-   The recording viewer endpoints from `recording_viewer.md`
    (`GET /api/agents`, `GET /api/audio/<idx>.wav`, `POST /api/load`,
    `POST /api/upload`).
-   The static assets for both frontends.
-   Reuse the service implemented as user required.

Use sub-routes to disambiguate the two surfaces (e.g.
`/chat/*` for the chat UI assets and `/viewer/*` for the viewer
assets, with a shared `/` shell that renders the sidebar described in
Step 7). Do **not** spin up a second process or a second port for the
viewer.

**Definition of done:** a single backend process starts on the
configured port; the chat WebSocket endpoint accepts a connection;
the viewer JSON / WAV / load / upload endpoints all respond; both
frontends are reachable from the same origin.

### Step 7 — Frontend UI

Adapt `interactive_ui.md`'s reference frontend for the chat surface
and `recording_viewer.md`'s reference frontend for the viewer
surface (do not generate either from scratch). Both surfaces are
served by the single backend from Step 6 and share a common shell.

**Sidebar navigation (REQUIRED).** The shared shell MUST render a
persistent left sidebar with at least two entries:

-   **Chat UI** — routes to the chat playground.
-   **Recording viewer** — routes to the viewer.

Selecting a sidebar entry swaps the main content area to the
corresponding surface without a full page reload (client-side route
or equivalent). The currently active entry MUST be visually marked.
The sidebar is always visible and follows the responsive layout rules
in `references/requirements.md` (no page-level scrolling at
≥ 1366×768; sidebar + content split the viewport).

The **Chat UI** surface MUST allow the user to:

-   Start a new connection / close the current connection.
-   Select the model.
-   Select input sources (audio and / or video — camera or screenshot)
    and stream them to the model.
-   Send text messages.
-   Hear model audio and see model + user transcription / conversation
    history.

Because the chat UI is only generated when the user also opted in to
the recorder and viewer (per Step 1's bundling rule), the chat UI
MUST show a save-recording dialog when the session ends, and that
dialog MUST expose an **"Open in viewer"** action that navigates to
the viewer sidebar entry with the just-saved recording loaded.

**Definition of done:** the page renders the sidebar plus both
surfaces; switching sidebar entries swaps the content area; clicking
through the chat UI exercises every required chat interaction; the
viewer entry loads recordings via the shared backend; the
"Open in viewer" action from the save-recording dialog lands the user
on the viewer surface with the recording already loaded.

### Step 8 — Verify

Run the verification protocol in `references/requirements.md` (smoke
checks + end-to-end interactions for every produced service).

**Definition of done:** every applicable verification passed; failures
have been fixed and re-verified, not merely reported.

### Step 9 — Documentation

Produce in the destination folder:

-   `README.md` — orientation + quick start: install, run the smoke
    check from Step 5, programmatic usage of the service class with
    full examples of building a `ClientMessage` for each modality and
    consuming `ServerMessage`s.
-   `how_to_test_with_ui.md` — only if the chat UI was enabled. Covers
    starting the unified backend, the URL to open, sidebar
    navigation between the chat surface and the viewer surface, how
    to interact with the model through the chat UI, and how to open
    a just-recorded session in the viewer via the save-recording
    dialog. Defer to `recording_viewer.md` for what the viewer modes
    show.

**Definition of done:** all expected docs exist, the commands they
print actually run, and the URLs they cite resolve.
