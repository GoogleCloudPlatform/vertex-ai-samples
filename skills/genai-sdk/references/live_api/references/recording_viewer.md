---
name: live-api-recording-viewer
description: >-
  Implements a single-page web viewer for a Live API session log produced by
  the message recorder (see `message_recorder.md`). Renders one interactive
  timeline per agent showing every client/server frame as a colored bar,
  supports per-message inspection of the full proto, and plays back a stereo
  audio mix of the conversation. Use when building a tool to debug, audit, or
  share a captured Live API session.
---

# Live API Recording Viewer

A web app that loads a recorded Live API session (the on-disk log written by a
`MessageRecorder` — see `message_recorder.md`) and renders it as one
interactive timeline per agent. It is intended as a debugging / auditing /
sharing tool: a developer points the viewer at a recording and immediately
sees the bidirectional sequence of frames, can hover or click any frame to
inspect its full payload, and can listen to either an individual audio chunk
or a full stereo mix of the conversation.

The architecture is a thin HTTP server + a single-page frontend. The server
owns parsing and audio assembly; the frontend owns visualization and
playback control.

## High-level architecture

```
┌──────────────────────────┐         ┌──────────────────────────────┐
│  Recording on disk       │  load   │  HTTP server                 │
│  (message_recorder log)  │ ──────► │  - parse log                 │
└──────────────────────────┘         │  - reconstruct timelines     │
                                     │  - assemble per-agent WAV    │
                                     │  - serve JSON + WAV + SPA    │
                                     └────────────┬─────────────────┘
                                                  │ HTTP
                                                  ▼
                                     ┌──────────────────────────────┐
                                     │  Single-page frontend (HTML  │
                                     │  + JS, no build step)        │
                                     │  - per-agent timeline rows   │
                                     │  - hover / pin tooltip       │
                                     │  - chunk + full playback     │
                                     │  - per-agent zoom + fisheye  │
                                     └──────────────────────────────┘
```

The server starts with **no recording loaded**. The user picks a file path or
uploads a file from the page; the server then parses and reconstructs it and
serves the page.

---

## Part 1 — Server

### 1.1 HTTP endpoints

| Method & Path                  | Purpose |
| ------------------------------ | --- |
| `GET /`                        | Serves the single-page HTML (and any other static assets). |
| `GET /api/agents`              | Returns a JSON payload describing every agent's timeline (see schema below). |
| `GET /api/audio/<idx>.wav`     | Returns a 24 kHz **stereo** WAV for agent `<idx>`. Left channel = client (sent) audio, right channel = server (received) audio. Silence gaps are preserved so playback time aligns with the timeline. Interrupted audio is **not** included. |
| `POST /api/load`               | Body `{"path": "..."}`. Loads a new recording from a server-visible path. Returns the new `/api/agents` payload. |
| `POST /api/upload`             | Multipart upload (`file=...`) of a recording from the browser. Returns the new `/api/agents` payload. |

Implementation notes:

- Both load endpoints replace the in-memory recording atomically under a lock
  so a concurrent `/api/agents` or `/api/audio` call sees a consistent
  snapshot.
- Reading + reconstruction + WAV mixing are CPU-bound; offload them to a
  worker thread so the HTTP loop stays responsive.
- The upload limit (e.g. 512 MiB) should be configurable via a flag.

### 1.2 Loading a recording (reconstruction pipeline)

The server transforms a flat log of recorded messages into one
**reconstructed track per agent**, where each entry has been annotated with
`(start_time, end_time, interrupted)` ready for the frontend.

#### 1.2.1 Read

Read every record from the log into a list. Each record carries:

- `payload` — a oneof with the original client message (sent) or server
  message (received) frame.
- `monotonic_nanos` — observation timestamp (nanoseconds, monotonic).
- `agent_name` — which agent / session this frame belongs to.

#### 1.2.2 Pick a shared origin `T0`

To keep all agents on a single shared timeline, pick `T0` = the
`monotonic_nanos` of the **earliest setup message** observed across **any**
agent (the very first client setup message in the log). All replay times are
expressed as offsets `t = monotonic_nanos - T0`.

Fallback: if no setup message exists, use the smallest `monotonic_nanos` in
the log.

#### 1.2.3 Group + sort

Group records by `agent_name`, preserving first-seen agent order. Within
each group, sort by `monotonic_nanos`.

#### 1.2.4 Compute per-message playback windows

Each agent maintains **two independent running cursors**:

- `last_end_client` — end of the last *client* message scheduled.
- `last_end_server` — end of the last *server* message scheduled.

The two directions of the bidirectional stream play back on independent
timelines and must not block one another.

For each message at relative time `t = monotonic_nanos - T0`:

- Determine direction from the payload arm (client vs. server).
- Compute:
  - `start = max(t, last_end_<dir>)`
    — honors the original wall-clock spacing while preventing a new chunk
    from overlapping the still-playing tail of the previous same-direction
    chunk (e.g. consecutive audio chunks).
  - `end = start + duration_for(modality)` where:
    - **Audio (16-bit PCM mono)**: `duration_nanos = num_bytes / (sample_rate_hz * 2) * 1e9` (the `2` accounts for 2 bytes/sample × 1 channel). The sample rate is parsed from the MIME type's `;rate=N` parameter, with a sensible default (e.g. 24000 Hz) when absent.
    - **Text**: instantaneous (`end = start`).
    - **Image / video frame**: instantaneous.
  - When a message contains multiple inline parts, **sum** the per-part
    durations.
- Update `last_end_<dir>` to the new `end`.

#### 1.2.5 Apply interrupt semantics

When a server interrupt frame is observed at relative time `T`:

1. Schedule it as a **zero-duration** event at `start = end = T`. Do **not**
   push it through the cursor — that would schedule the interrupt *after*
   the audio it is supposed to cancel.
2. Walk the agent's already-scheduled messages **backwards** and mark every
   prior **server** message whose `start > T` (i.e. it has not yet started
   playing at the moment the interrupt arrives) as `interrupted = true`.
   Stop as soon as you find a server message with `start <= T` (server
   starts are monotonically non-decreasing in append order, so anything
   earlier also satisfies the condition).
3. **Client** messages and server messages that have already started are
   left untouched.
4. **Rewind only the server cursor** to `T` so post-interrupt server
   messages start at their natural recorded times instead of being pushed
   past the cancelled audio tail. The client cursor is independent and
   unchanged.

The frontend (and the WAV mixer) skip messages flagged `interrupted`.

#### 1.2.6 Emit

Produce one reconstructed track per agent containing the ordered list of
playback messages. Each message carries **both** sets of timestamps so the
frontend can switch between view modes without a server roundtrip:

- `playback_start_ms`, `playback_end_ms` — the reconstructed window from
  § 1.2.4–1.2.5 (cursor-pushed, modality-aware durations, interrupt-aware).
- `wire_ms` — the raw send/receive time relative to `T0`, i.e.
  `(monotonic_nanos - T0) / 1e6`. Not pushed by any cursor; reflects the
  exact moment the frame crossed the wire.

(See § 2.9 for the JSON shape.)

### 1.3 Per-message classification (for the timeline view)

For each playback message, the server computes a JSON-friendly summary used
by the frontend. The classifier inspects the payload arm and the inner
oneof (`message_type`) to determine:

- **`direction`** — `"client"` | `"server"` | `"unknown"`.
- **`kind`** — the inner oneof arm (e.g. `"setup"`, `"client_content"`,
  `"realtime_input"`, `"tool_response"`, `"server_content"`, `"tool_call"`,
  `"tool_call_cancellation"`, `"setup_complete"`, plus a synthetic
  `"server_content (interrupted)"` when the server-side `interrupted` flag
  is set on a `server_content` frame).
- **`modality`** — drives the bar's color in the UI. One of:
  `"text"`, `"audio"`, `"image"`, `"video"`, `"tool"`, `"setup"`,
  `"interrupt"`, `"control"`, `"turn_complete"`, `"generation_complete"`.
  Selection rules:
  - `client_content`: walk every part; if any part is text → `text`, an
    audio inline blob → `audio`, an image inline blob → `image`, a video
    inline blob → `video`.
  - `realtime_input`: classify by which content field is set
    (`audio` / `video` / `text` / `media_chunks[]`). For `media_chunks`
    and `realtime_input.video`, classify by MIME type prefix
    (`audio/*` → audio, `image/*` → image, `video/*` → video) — webcam
    "video" is usually JPEG frames and should render as `image`.
  - `tool_response` / `tool_call` / `tool_call_cancellation` → `tool`.
  - `setup` / `setup_complete` → `setup`.
  - `server_content` with `interrupted: true` → `interrupt`.
  - `server_content` with `model_turn` → classify by parts as above.
  - `server_content` with only `turn_complete` → `turn_complete`.
  - `server_content` with only `generation_complete` → `generation_complete`.
- **`text_preview`** — concatenated text from any text parts, truncated to
  ~200 chars with an ellipsis.
- **`audio_bytes`**, **`audio_rate_hz`** — totals across all audio parts.
- **`mime_types`** — list of MIME types seen on the message.
- **`interrupted`** — copied from the playback flag.
- **`playback_start_ms`**, **`playback_end_ms`**, **`playback_duration_ms`**
  — derived from the reconstructed nanosecond windows.
- **`wire_ms`** — raw send/receive time relative to `T0` (no cursor
  pushing). Used by Message-mode rendering.
- **`detail`** — a JSON dump of the full proto for the hover panel, after
  two transformations (see below).

#### 1.3.1 Stripping large blobs

A naive proto-to-JSON serialization base64-encodes all `bytes` fields,
which would balloon the JSON for any audio frame. Walk the JSON dict and
replace any `data` field whose value is a long base64 string with the
marker `<base64 omitted, ~N bytes>` (estimate `N` from the base64 length).

#### 1.3.2 Stripping unresolvable type-erased payloads

Some recordings contain "any-of-typed" fields (e.g. server-side debug
metadata) whose packed type isn't linked into the viewer binary. A naive
serialization will throw when it tries to expand them. Recursively walk a
copy of the message; for each sub-message that is a generic "any-typed"
container, attempt to resolve its `type_url` against the descriptor pool,
and if the lookup fails, clear the field. Surface the list of stripped
type URLs as `_unresolved_any_types` in the JSON so the UI can still
indicate the field was present.

### 1.4 Audio assembly (`/api/audio/<idx>.wav`)

For one agent, build a single 24 kHz **stereo** WAV:

1. Allocate two empty `int16` PCM buffers (left, right).
2. Walk the agent's playback messages in order. For each message:
   - **Skip** if `interrupted`.
   - Pick the destination channel by direction: client → left, server →
     right.
   - Compute the sample offset:
     `offset_samples = start_time_nanos * 24000 // 1_000_000_000`.
   - For every audio inline blob on the message:
     - Decode as little-endian `int16` PCM.
     - **Resample** linearly from the blob's source rate (parsed from MIME
       `;rate=N`, default 24000) to 24000 Hz.
     - **Mix-add** (saturating to `[-32768, 32767]`) into the destination
       buffer at `offset_samples`. Extend the buffer with zeros if needed.
3. Pad both channels to the same length, interleave LRLR, and emit as a
   standard 16-bit PCM stereo WAV at 24000 Hz.

Note: the mix is bytes-only, no decoding library required.

---

## Part 2 — Frontend (single-page app)

A single self-contained HTML file with inline CSS and JS — no build step,
no framework. Layout:

```
┌──────────────────────────────────────────────────────────────┐
│ Header: title, current path, [path input] [Load] [Upload…]   │
├──────────────────────────────────────────────────────────────┤
│ Agent block (one per agent):                                 │
│   ┌────────────────────────────────────────────────────────┐ │
│   │ <agent_name>   N msgs · total time   [▶ Play] <audio>  │ │
│   │                                       [− 100% +] Reset │ │
│   ├────────────────────────────────────────────────────────┤ │
│   │ Color legend                                           │ │
│   ├────────────────────────────────────────────────────────┤ │
│   │ Timeline (horizontally scrollable):                    │ │
│   │   Ruler with time ticks                                │ │
│   │   CLIENT → │  ▒▒▒    │  ▓▓  │  ▒▒    │ ▌  │           │ │
│   │   SERVER ← │       ▓▓▓▓▓▓▓ │ ▒▓ │ ▓▓▓▓▓▓▓ │            │ │
│   └────────────────────────────────────────────────────────┘ │
│ ... more agent blocks ...                                    │
└──────────────────────────────────────────────────────────────┘
```

### 2.1 Toolbar (top header)

- **Path input + Load button** — POSTs the path to `/api/load`, then
  re-renders. Pressing Enter in the input triggers Load.
- **Upload button** — opens a file picker, then POSTs the file as
  multipart `file=` to `/api/upload`, then re-renders.
- **Status text** — neutral / `ok` (green) / `error` (red), reflecting the
  last load attempt.

### 2.2 Per-agent block

Rendered for each entry in the `agents` array of the JSON payload:

- **Header**: agent name, message count, total time.
- **Player controls**:
  - "▶ Play full timeline" button — plays the agent's full WAV (the native
    `<audio controls>` element exposed below it) from the start.
  - Native `<audio controls>` with `src="/api/audio/<idx>.wav"` and
    `preload="none"` so the WAV is only fetched on first play.
  - **Per-agent zoom widget**: `[−] [<level%>] [+] [Reset]`. Each agent's
    timeline has independent zoom.
  - **Per-agent view-mode toggle**: see § 2.3.
- **Legend**: small swatches for every modality color used on the
  timeline.
- **Timeline**: the heart of the view (§ 2.4).

### 2.3 View modes (Playback vs. Message)

The viewer supports **two view modes** for the timeline. The same set of
messages is rendered in both modes, but they answer fundamentally different
questions about the session:

- **Playback mode** answers: *"When will each piece of audio actually start
  playing, and when will it finish?"* It uses the reconstructed playback
  schedule, which pushes back-to-back chunks so they don't overlap and
  drops audio that was interrupted.
- **Message mode** answers: *"When did each frame actually cross the wire
  — when was it sent (client) or received (server)?"* It plots each
  message at its raw observation time. **It does NOT show when the message
  will be played**; it shows when the message was *transmitted*. A burst of
  20 audio chunks received in 50 ms appears as 20 bars packed into 50 ms,
  even though playing all of them back takes several seconds.

| Mode         | What is plotted                                          | Bar shape |
| ------------ | -------------------------------------------------------- | --- |
| **Playback** | Reconstructed `[playback_start_ms, playback_end_ms]` windows from § 1.2 — cursor-pushed, modality-aware durations, with interrupted server messages flagged. Shows **when audio will start playing and when it will end**, accounting for back-to-back chunks that must queue and for interrupts that drop pending audio. | Audio = wide duration bars (width ∝ playback duration). All other messages = thin instant markers. (This is the original timeline rendering described in § 2.4.) |
| **Message**  | Raw `wire_ms` — the exact moment each frame was **sent** (client direction) or **received** (server direction). **Not** when the message will be played. No cursor pushing, no playback-duration accounting, no interrupt flag affecting bar shape. Shows the precise wire-time spacing of frames so dense bursts and quiet gaps in the actual stream are visible. | **Every** message — including audio — is a small fixed-width bar (e.g. 4 px wide, full row height) anchored at `wire_ms`. Color still encodes modality; interrupted audio is still marked with the dropped pattern. |

#### 2.3.1 Mode toggle UI

- A **global toggle** lives in the top header (e.g. `[Playback | Message]`
  segmented control). Switching it sets the **default** mode for every
  agent and re-renders all timelines.
- Each agent header **also** carries a small mode toggle (next to the
  zoom widget). Changing it **overrides** the global default for that
  agent only. This lets the user, for example, view agent A in Playback
  mode while comparing the wire-time bursts of agent B in Message mode.
- When the global toggle is changed, agent-level overrides are **cleared**
  so all agents follow the new global setting. (Otherwise the global
  toggle would feel "stuck" on agents the user previously overrode.)
- Persist the global mode and per-agent overrides only in memory (no
  cookies / localStorage required); reset on every fresh `/api/agents`
  payload.

#### 2.3.2 Effect on layout & total duration

- `total_ms` per agent is computed differently per mode:
  - Playback mode: `max(playback_end_ms)` over the agent's messages.
  - Message mode: `max(wire_ms)` over the agent's messages.
- The two values are usually close but **not identical** — playback mode
  tends to be longer because cursor-pushing extends the schedule past
  closely-spaced audio chunks.
- When switching modes, recompute `total_ms`, the `pxPerMs`, the ruler
  ticks, and re-render. Apply the same viewport-center preservation
  trick from § 2.8 if you want zoom-stable mode switching (optional —
  starting at the left edge after a mode switch is also fine).

#### 2.3.3 Effect on bars and clusters

- **Playback mode** renders bars exactly as described in § 2.4 (duration
  bars + instant markers + cluster band split).
- **Message mode** renders **every** message as a small fixed-width bar
  (e.g. 4 px) anchored at `wire_ms`. Because every bar has the same width,
  bars at the same wire time would otherwise stack on top of each other and
  be indistinguishable. To keep them all visible and individually clickable:
  - Sort by `wire_ms` and append in order; later bars paint on top.
  - The fisheye magnification (§ 2.5) still applies and visibly fans the
    bars apart near the cursor, restoring clickability in dense regions.
  - **Same-time stacking → vertical band split (REQUIRED).** Apply the same
    cluster-band layout used by Playback-mode markers (§ 2.4.2): when N
    messages on the same row share (or land within `CLUSTER_PX` of) the
    same wire time, **split the row height into N equal horizontal
    bands**. The first message occupies the top `1/N` slice (`top: 0`,
    `height: ROW_H/N`), the second occupies the next `1/N` slice
    (`top: ROW_H/N`), and so on. All N members are simultaneously visible
    and individually clickable — no two bars at the same wire time may
    fully cover one another.
  - **Cursor-driven band height (REQUIRED).** When the cursor hovers near
    a same-time cluster, **redistribute the row height between the cluster
    members based on the cursor's vertical position**: the band whose
    resting center is closest to the cursor's `Y` grows taller, while
    siblings shrink — but the bands always sum to the full row height so
    they never overlap. This is the same mechanism described for
    Playback-mode markers in § 2.5.1; reuse the same algorithm here:
    - Gate the redistribution by horizontal proximity to the cluster's
      column so distant clusters keep their resting equal-band layout.
    - Weight each band by a Gaussian of `(cursorY - bandCenterY)` (smaller
      sigma = sharper near/far contrast), and rescale so the weights sum
      to `ROW_H`.
    - Reset to the equal `ROW_H/N` band heights on `mouseleave`.
  - Tag each cluster member with `data-cluster-id`, `data-band-index`,
    `data-band-count`, `data-band-locked='1'` so the magnification code
    in § 2.5 / § 2.5.1 picks them up automatically.
- The interrupted overlay (striped/strike-through) still applies in both
  modes — interrupted audio existed on the wire even though the player
  skips it.

#### 2.3.4 Audio playback is unaffected

Audio playback **always uses the reconstructed (playback-mode) timeline**,
regardless of the active view mode. Concretely:

- The "▶ Play full timeline" button and the native `<audio>` element
  always play the agent's WAV (built from the playback windows per § 1.4).
- The "▶ Play audio" button in the pinned tooltip always plays the
  message's `[playback_start_ms, playback_end_ms]` slice of the WAV (per
  § 2.7.2), even when the user is currently looking at message mode.

The mode toggle is purely a **visualization** choice. This decoupling lets
the user compare wire-time bursts visually while still hearing the
correctly-scheduled audio.

### 2.4 Timeline rendering

The timeline is a fixed-height area split into two horizontal **rows**:

- Top row: `CLIENT →` (frames sent by the application).
- Bottom row: `SERVER ←` (frames received from the model).

A **ruler** above the rows shows time ticks. The pixel-per-millisecond scale
is computed as:

```
basePxPerMs = max(viewport_width / total_ms, 0.001)   # fit-to-width
pxPerMs     = basePxPerMs * agentZoom                 # apply zoom
totalWidth  = total_ms * pxPerMs + LABEL_W + RIGHT_PAD
```

Ruler tick interval is chosen from a "nice" list (50, 100, 200, 500,
1000, 2000, ...) so there are roughly 6–12 ticks across the timeline.

A sticky left **gutter** (e.g. 90 px wide) carries the row label
(`CLIENT →` / `SERVER ←`) and stays fixed when the user scrolls
horizontally.

#### 2.4.1 Bars (duration messages)

For every message with `playback_duration_ms > 0` (today: only audio):

- Position: `left = playback_start_ms * pxPerMs`,
  `width = max(2, playback_duration_ms * pxPerMs)`.
- Height: full row height (~18 px).
- Color: from `modality` (audio = blue for client, amber for server, etc.).
- If `interrupted`: render with a diagonal-stripe pattern and strike-through
  text — the bar represents audio that the player will **skip**.
- If width ≥ 30 px, paint the modality name inside the bar; otherwise add a
  `tiny` class that hides the label.

**Paint order**: append duration bars in **decreasing-width order** so a
shorter chunk that visually nests inside a longer one stays on top and
remains clickable.

#### 2.4.2 Markers (instant messages)

For every message with `playback_duration_ms == 0` (text, setup, tool, interrupt,
turn_complete, ...):

- Render as a **thin (2 px) vertical line** that overhangs the row by
  ~2 px above and below.
- Always painted on top of duration bars (high `z-index`) so they remain
  visible even when overlapping a long audio chunk.

**Cluster handling.** When several instant markers land on the same /
nearly-same pixel column (e.g. `setup` + the first text frame, or rapid
tool exchanges):

1. Group them into clusters: walk the sorted instants and merge any
   marker whose `left` is within `CLUSTER_PX` (e.g. 6 px) of the running
   cluster's max `left`.
2. For a solo cluster (n=1): keep the default full-row marker styling.
3. For a multi-member cluster (n>1): split the row height into **N equal
   horizontal bands**. Each marker occupies its own band (top 1/N, middle
   1/N, ...) so all members stay simultaneously visible and individually
   clickable. Tag each member with `data-cluster-id`, `data-band-index`,
   `data-band-count` so the magnification code (§ 2.5) can recognize and
   re-balance them.

#### 2.4.3 Color palette (suggested)

| Modality                             | CSS variable     | Purpose |
| ------------------------------------ | ---------------- | --- |
| `audio` (client direction)           | blue             | Client realtime / inline audio. |
| `audio` (server direction)           | amber            | Model audio output. |
| `text`                               | green            | Any text part. |
| `image`                              | pink             | Inline image / image-typed video frame. |
| `video`                              | teal             | True video MIME. |
| `tool`                               | purple           | Tool call / response / cancellation. |
| `setup`                              | slate            | `setup` / `setup_complete`. |
| `interrupt`                          | red              | Server interrupt. |
| `control` / `turn_complete` / `generation_complete` | dark slate | Misc control. |
| Interrupted (overlay)                | striped gray     | Overlaid on the bar's base color. |

A legend in each agent block exposes the palette.

### 2.5 Cursor magnification (fisheye / "macOS dock" effect)

**User-facing behavior (REQUIRED).** When the user moves the mouse over
a row track, the bars **closest to the cursor visibly zoom in** — they
grow wider (and instant markers grow thicker) so they're easier to read
and click — while bars far from the cursor stay at their resting size.
The effect is exactly like the macOS Dock magnifying icons under the
cursor: chunks fan outward smoothly, with the maximum magnification at
the cursor itself and a soft falloff over a fixed pixel window. **No
clicking is required to trigger it; just hovering**. This makes dense
audio streams (many tiny chunks crammed into a few pixels) inspectable
without permanently zooming the timeline.

Concrete user-visible properties:

- An audio chunk that's, say, 6 px wide at rest can grow to ~12 px when
  the cursor sits directly over it, then smoothly shrink back to 6 px
  as the cursor moves away.
- A 2 px instant marker grows to ~4 px under the cursor.
- Bars *outside* the lens window are completely unaffected — their
  on-screen position and width never change. This is critical: it
  guarantees timeline alignment far from the cursor never shifts.
- Adjacent magnified chunks are separated by a small visual gap so the
  user can clearly see and click each one individually.
- A thin vertical cursor line is painted under the mouse so the user
  can read off the timeline coordinate.

Implementation outline:

1. On `mouseenter`, **snapshot** every bar's resting `[left, width]`
   (and, for cluster band members, their resting `[top, height]`) so
   `mouseleave` can restore them exactly.
2. On `mousemove`, for each bar:
   - Apply a **non-uniform x-mapping** that pins both endpoints of a
     lens window of half-width `R` (e.g. 140 px) around the cursor.
     Inside the window, space near the cursor is locally stretched by a
     factor `k` (e.g. 2.4); outside the window, `x` is unchanged.
     A simple choice: for `t = (x - cursor) / R` with `|t| < 1`,
     `t' = sign(t) * (1 - (1 - |t|)^k)`; `x' = cursor + t' * R`.
   - For a **duration bar (audio chunk)**: re-place at `[x'(left),
     x'(left+width)]`, subtract a small visual gap (e.g. 1 px) so
     adjacent chunks have a visible seam. **The bar's width is
     `x'(left+width) - x'(left) - gap`, which is naturally larger than
     the resting width whenever the bar overlaps the lens window —
     this is the "chunk zoom-in on hover" effect.** The closer the bar
     is to the cursor, the more its `[left, right]` endpoints are
     pulled apart, so the wider it becomes on screen.
   - For an **instant marker**: translate `left → x'(left)` AND apply a
     horizontal `scaleX` based on a Gaussian magnification (peak ~2.0
     at the cursor, sigma ~70 px) so the 2 px line visibly fattens
     near the cursor.
3. Paint a thin vertical **cursor line** under the cursor.
4. On `mouseleave`, **restore** every bar from the snapshot. The
   restoration MUST be exact: any drift on repeated enter/leave cycles
   is a bug.

Tuning hints (the testing implementation uses these):

- Lens half-width `R`: ~140 px works well for typical timeline
  densities; the lens is wide enough to magnify ~5–10 chunks
  simultaneously but narrow enough that the unaffected outside region
  is the majority of the screen.
- Stretch factor `k`: ~2.4. Higher values give a more dramatic peak
  but exaggerate the boundary discontinuity at `|t| = 1`.
- Peak magnification for instant markers: ~2.0×.
- Gaussian sigma for instant magnification: ~70 px. Smaller sigma
  yields a sharper, more localized "spotlight" effect.
- Visual gap between adjacent magnified chunks: ~1 px.
- Throttle paint via `requestAnimationFrame` (see end of § 2.5.1) so
  the magnification stays smooth even on dense timelines.

#### 2.5.1 Cluster band redistribution

For each multi-member cluster (§ 2.4.2), additionally redistribute the
row height between members based on the cursor's vertical position:

- Gate by horizontal proximity (only kick in when the cursor's column is
  also near the cluster's column) so distant clusters keep their resting
  equal-band layout.
- Weight each band by a Gaussian of `(cursorY - bandCenterY)`; bands closer
  to the cursor grow taller, others shrink, but the **sum equals the row
  height** so siblings never overlap in Y.
- Set each band's `top` and `height` from the normalized weights.

This is purely visual; the underlying `(left, width)` of each marker is
unchanged.

Throttle paint via `requestAnimationFrame`.

### 2.6 Tooltip / inspector panel

A floating panel anchored near the cursor.

- **Hover** any bar/marker → the panel appears with a summary view (read-only,
  follows the cursor).
- **Click** any bar/marker → the panel becomes **pinned**: the border
  highlights, it stops following the cursor, and its contents become
  interactive (scrollable, with action buttons).

Panel contents:

- Header: `<DIRECTION> · <kind>` (e.g. `SERVER · server_content`).
- Badges: the modality, plus `INTERRUPTED — not played` (red) if applicable.
- A **"▶ Play audio"** button — only when the message has `audio_bytes > 0`
  and is not interrupted. Clicking plays only that chunk (see § 2.7.2).
- If the message has audio bytes but is interrupted, show a note instead:
  "Audio was interrupted; not present in the agent WAV."
- A small key/value table: `start`, `end`, `duration`, `audio` (bytes + Hz),
  `mime`, `text` preview.
- A `<pre>` block with the full `detail` JSON dump (the proto → JSON
  conversion from the server, with large blobs already replaced by
  `<base64 omitted, ~N bytes>` markers).

Pinning UX:

- Pinned panel has a "×" close button in a top bar plus a hint
  ("scroll to read · click outside to close").
- Clicking another bar/marker re-pins to the new message.
- Clicking outside the panel (and not on a bar) unpins.
- `Escape` unpins.

Positioning: place near the cursor; flip horizontally / vertically to stay
on-screen.

### 2.7 Audio playback

Two distinct playback modes share a single per-agent decoded buffer cache.

#### 2.7.1 Play full timeline

The "▶ Play full timeline" button (and the native `<audio>` element below
it) play `/api/audio/<idx>.wav` from the start using the standard browser
`<audio>` element.

#### 2.7.2 Play a single message chunk

Clicking "Play audio" in the pinned panel plays exactly the
`[playback_start_ms, playback_end_ms]` slice of the agent's WAV. Implementation:

1. Use the **Web Audio API** (not the `<audio>` element — its seeking
   semantics with `preload='none'` and media-fragment caching are
   unreliable).
2. Lazily fetch + decode the agent's WAV the first time it's needed. Cache
   the decoded `AudioBuffer` per agent (keyed by agent index). Cache the
   in-flight decode promise too so rapid repeated clicks don't issue
   duplicate fetches.
3. To play a chunk: stop any previously-playing chunk source, create a
   new `BufferSource`, connect to `destination`, and call
   `source.start(0, startSec, durationSec)` — the third parameter bounds
   playback to exactly the slice.
4. Track the active source so the next click can stop it; clear it on
   `onended`.
5. If the `AudioContext` is suspended (browsers require a user gesture),
   resume it on the click.

The native `<audio>` element used for full playback also has a "stop-at"
helper attached so that if the user later starts manual playback via the
native controls, any previously-set stop boundary is cleared and unbounded
playback works as expected.

### 2.8 Zoom

Per-agent zoom controls ( `[−] [<level%>] [+] [Reset]` ) live in each agent
header.

- `ZOOM_STEP = √2` (so two clicks doubles the zoom).
- Clamp between e.g. `1/8×` and `64×`.
- "Reset" returns to fit-to-width (`1.0×`).
- Applying a new zoom: re-render only that agent's timeline at the new
  `pxPerMs`, then **restore the same timestamp at the viewport center**:
  - Capture `centerMs = (oldScrollLeft + viewportW/2 - LABEL_W) / oldPxPerMs`.
  - After replacing the timeline element, set
    `newScrollLeft = centerMs * newPxPerMs + LABEL_W - viewportW/2`,
    clamped to `[0, scrollWidth - clientWidth]`.
- Zoom state lives in a per-agent map; on each fresh `/api/agents` payload,
  clear the map so newly-loaded agents start at fit-to-width.

### 2.9 JSON payload shape (between `/api/agents` and the frontend)

```jsonc
{
  "input_path": "/path/to/recording  OR  <uploaded: name>",
  "agents": [
    {
      "index": 0,
      "agent_name": "alice",
      "total_ms": 128340.5,
      "messages": [
        {
          "direction": "client" | "server" | "unknown",
          "kind": "setup" | "client_content" | "realtime_input"
                | "tool_response" | "server_content" | "tool_call"
                | "tool_call_cancellation" | "setup_complete"
                | "server_content (interrupted)" | "unknown",
          "modality": "text" | "audio" | "image" | "video" | "tool"
                    | "setup" | "interrupt" | "control"
                    | "turn_complete" | "generation_complete",
          "interrupted": false,
          // Reconstructed playback window (cursor-pushed, modality-aware
          // duration). Used by Playback mode and by audio playback.
          "playback_start_ms": 1234.5,
          "playback_end_ms":   1456.7,
          "playback_duration_ms": 222.2,
          // Raw wire time relative to T0. Used by Message mode.
          "wire_ms": 1230.1,
          "text_preview": "...",
          "audio_bytes": 9600,
          "audio_rate_hz": 24000,
          "mime_types": ["audio/pcm;rate=24000"],
          "detail": { /* full proto-to-JSON, large blobs stripped */ }
        }
      ]
    }
  ]
}
```

---

## Reproducing this viewer in another stack

The viewer is intentionally split so each half is independently
reimplementable:

- **Server**: any HTTP framework (Tornado, FastAPI, Express, Go's
  `net/http`, etc.) is fine. The only real work is:
  1. Parsing the recorder's on-disk format into a list of records.
  2. Reconstructing per-agent tracks per § 1.2 (this is the hard part).
  3. Mixing per-agent stereo WAVs per § 1.4.
  4. Serving the four endpoints in § 1.1.
- **Frontend**: vanilla HTML/CSS/JS is sufficient — no framework needed.
  The whole UI (~1200 lines) is one HTML file with one inline `<style>` and
  one inline `<script>`. A framework is fine if you prefer; the only tricky
  pieces are the cluster-band layout (§ 2.4.2), the fisheye magnification
  (§ 2.5), and the chunk-precise Web Audio playback (§ 2.7.2).

The contract between the two halves is the JSON shape in § 2.9 plus the
WAV endpoint. As long as a server produces those, any frontend can render
the view; as long as a frontend consumes those, any server can back it.
