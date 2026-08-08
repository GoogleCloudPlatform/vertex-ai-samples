---
name: live-api-message-recorder
description: >-
  Defines the contract for an asynchronous, non-blocking message recorder that
  captures the bidirectional traffic of a Live API session (sent
  `BidiGenerateContentClientMessage` and received
  `BidiGenerateContentServerMessage` frames) to durable storage. Use when you
  need to persist a complete, replayable transcript of a Live API conversation
  without stalling the conversation's I/O loops.
---

# Live API Message Recorder

A `MessageRecorder` is an optional companion to a Live API session manager. Its
sole job is to **persist every frame that crosses the WebSocket** — both the
client messages the application sends and the server messages it receives — to
an append-only log on disk (or any other byte-stream sink), in the order they
were observed, **without ever blocking the conversation's send/receive loops**.

The recorder is intentionally a narrow, single-purpose component: it does not
parse, transform, or interpret messages. It treats each message as opaque
bytes (or an opaque structured record) and is responsible only for buffering,
serializing, and writing.

## Core Responsibilities

1. **Capture both directions.** Accept and persist both sent
   `BidiGenerateContentClientMessage` frames and received
   `BidiGenerateContentServerMessage` frames (see
   `client_server_messages.md`). The two directions must be distinguishable in
   the persisted log.
2. **Be non-blocking.** A call to record a message must return effectively
   immediately. The actual write to disk happens on a background worker and
   must never stall the session's send or receive loop.
3. **Lossless ordering.** Records are written in the order they were enqueued.
   Any reasonable reader of the log can reconstruct the exact sequence of
   client-↔-server frames.
4. **Safe lifecycle.** Provide explicit `start` / `close` (or open / flush /
   close) semantics, plus a context-manager-style helper in languages that
   support it. `close` must drain any pending in-memory records before
   returning.
5. **Fail-soft.** A failure to write a single record (disk full, transient
   I/O error, serialization bug on a single record) must NOT crash the
   session. Log the error and keep the recorder alive for subsequent records.
6. **Optional integration.** A session manager that supports a recorder must
   treat it as **optional**: if no recorder is supplied, the manager behaves
   exactly as before with zero overhead and zero recording-related code paths
   active.

## Record Schema

Each persisted entry should be a self-describing record containing at minimum:

| Field           | Description |
| --------------- | --- |
| `payload`       | A oneof / tagged union holding either the client message or the server message. The two arms must be distinguishable so a reader knows which direction the frame travelled. |
| `timestamp`     | A monotonic or wall-clock timestamp (e.g. nanoseconds) marking when the frame was observed. Used for replay timing and latency analysis. |
| `agent_name`    | (Optional but recommended) An identifier of the participant — useful when multiple sessions or multiple roles share a log. |

### Wire format — REQUIRED

All recorders **MUST** persist records as a **length-prefixed serialized
protobuf** stream written to a file with the `.pb` extension. No other
container format is permitted.

Concretely, every record on disk is:

```
[varint length][serialized protobuf bytes]
[varint length][serialized protobuf bytes]
...
```

This format is:

- **Append-only** and **streamable** — a crash mid-write loses at most
  the last partial record.
- **Order-preserving** — records are read back in the order they were
  written.
- **Portable across languages** — any language with a protobuf library
  can read/write it.
- **Compact and fast** — no base64 expansion of binary audio/image data.

### Formats that MUST NOT be used

- **JSON / JSON Lines (`.jsonl`)** — human-readable but inflates audio
  payloads ~33% via base64 and loses fidelity on binary fields.
- **TFRecord** — extra TensorFlow-specific framing; not used here.
- **`recordio`** — Google-internal format with no widely-available
  cross-language readers/writers.
- Anything that requires the entire log to be parsed as a single root
  message (see size-limit warning below).

### Per-record size — IMPORTANT

**Write exactly one message per record. Never batch multiple messages into a
single record before writing.**

Protocol Buffers impose a hard **2 GB limit** on the size of any single
serialized message (and most implementations enforce a much lower default
parse limit, e.g. 64 MB). A long Live API session can easily produce
gigabytes of audio frames; if you accumulate many records into a wrapper
proto and serialize the wrapper as one record, you risk:

- **Hitting the 2 GB hard cap** and being unable to write at all.
- **Hitting the default parse limit** at read time, making the log
  unreadable without raising parser limits.
- **Losing the entire batch** on any single write/parse failure, instead of
  losing one frame.

Always serialize and write **one record = one message**. The on-disk
length-prefixed stream is what concatenates records, not protobuf itself.

The recorder treats the record as opaque: callers populate the
direction-specific arm of `payload`, the `timestamp`, and the `agent_name`.
The recorder neither inspects nor mutates the record before writing it.

## Concurrency / Async Model

The recorder must be safe to call from the conversation's I/O loop without
back-pressuring it. The recommended structure:

1. **In-memory queue.** `record(...)` enqueues the record onto an unbounded
   (or bounded with documented overflow policy) in-memory queue and returns
   immediately. No disk I/O happens on the caller's thread/task.
2. **Background writer.** A single background worker (asyncio task,
   goroutine, dedicated thread, etc.) drains the queue and performs the
   actual serialization + write.
3. **Offload blocking I/O.** If the host runtime is single-threaded async
   (e.g. Python asyncio, JS event loop), the actual file write must be
   dispatched to a worker thread / thread pool so the event loop is never
   blocked.
4. **Loop affinity (async runtimes).** In async languages, the recorder is
   bound to the event loop that called `start()`. `record(...)` must be
   invoked from that same loop. If the application needs to record from a
   different thread (e.g. a gRPC callback thread), it must marshal the call
   onto the recorder's loop (e.g. `asyncio.run_coroutine_threadsafe` in
   Python, equivalent primitives elsewhere).

## Lifecycle

The recorder must support:

- **Construction** — pure, side-effect free. No file is opened, no queue is
  created. Only configuration (output path, etc.) is captured.
- **`start()`** — opens the output sink, creates the in-memory queue, and
  spawns the background writer. **Idempotent**: a second call is a no-op.
  Opening the sink is itself blocking I/O and must be offloaded to a worker
  thread in async runtimes.
- **`record(message)`** — enqueues a single record. Non-blocking. After
  `close()` has been called, this becomes a no-op that logs a warning
  (rather than raising) so a late-arriving frame from a still-running
  receive loop does not crash the process.
- **`close()`** — signals the writer to drain remaining records, then closes
  the sink. **Idempotent.** Implementations should:
  - Use a queue-shutdown / sentinel mechanism so the writer drains existing
    items before exiting.
  - Bound the wait with a timeout (e.g. 10 s). If the writer does not exit
    in time, force-cancel it so `close()` never hangs the caller.
  - Always close the underlying sink on exit, even if the writer raised.
- **Context-manager helper** — in languages with such a construct (`async
  with` in Python, `defer` / `using` / RAII elsewhere), provide one that
  calls `start()` on entry and `close()` on exit, guaranteeing flush + close
  even on exceptions.

## Background Writer Loop (reference algorithm)

```text
loop:
  try:
    item = queue.get()           # awaits / blocks until an item or shutdown
  on queue-shutdown:
    break                        # graceful exit; close the sink in finally
  try:
    serialized = serialize(item)
    blocking_write(sink, serialized)   # offload to worker thread in async runtimes
  on any error:
    log_error("dropping one record"); continue
  finally:
    queue.task_done()

finally:
  try: blocking_close(sink)
  on any error: log_error("error closing sink")
```

Key invariants:

- A single bad record never kills the writer; it is logged and dropped.
- The sink is closed exactly once, on writer exit.
- The writer exits cleanly when the queue is shut down by `close()`.

## Public API (language-agnostic)

| Operation                       | Semantics |
| ------------------------------- | --- |
| `MessageRecorder(output_path)`  | Construct; no I/O. |
| `start()`                       | Open sink, start writer task. Idempotent. |
| `record(message)`               | Enqueue a record. Non-blocking. No-op + warning after `close()`. |
| `close()`                       | Drain queue, stop writer, close sink. Idempotent. Bounded by a timeout. |
| Context-manager (if available)  | `start()` on enter, `close()` on exit. |

## Integration Contract with the Session Manager

A `SessionManager` that supports a recorder MUST:

1. Accept an **optional** recorder instance at construction (or via a
   setter). If `None` / not supplied, recording is disabled and there is no
   runtime overhead.
2. For every successfully sent client frame and every received server frame,
   **build a fully-populated record before** calling `recorder.record(...)`.
   The recorder treats the record as opaque and will not fill in any fields.
   Specifically, the session manager MUST:
   1. Set the appropriate arm of the `payload` oneof (client-message arm for
      sent frames, server-message arm for received frames).
   2. Set `timestamp` to the moment the frame was observed (capture it as
      close to the send/receive point as possible for accurate replay
      timing).
   3. Set `agent_name` (and any other contextual fields) if relevant.
   4. Only then pass the completed record to `recorder.record(...)`.
3. Treat recording as **best-effort**: the session manager's correctness and
   liveness must not depend on the recorder. Recorder errors are logged but
   do not interrupt the session.
4. **Not** call `recorder.start()` or `recorder.close()`. The recorder's
   lifecycle is owned by the caller that constructed it, not by the session
   manager. A single recorder instance MAY be shared across multiple
   concurrent session managers (e.g. one log capturing the traffic of
   several agents in a multi-agent system); in that case the `agent_name`
   field on each record is what disambiguates which session a frame belongs
   to. The caller is responsible for `start()`-ing the recorder before
   handing it to any session manager and `close()`-ing it after all session
   managers using it have finished.

## Example Usage (Python, illustrative)

The caller owns the recorder's lifecycle. A single recorder can be shared
across multiple session managers (e.g. one transcript for several agents):

```python
async with MessageRecorder("/tmp/multi_agent.pb") as recorder:
    # Multiple session managers share the same recorder; agent_name on each
    # record disambiguates which session a frame belongs to.
    alice = SessionManager(..., recorder=recorder, agent_name="alice")
    bob   = SessionManager(..., recorder=recorder, agent_name="bob")
    await asyncio.gather(alice.run(), bob.run())
# recorder is flushed and closed here, after BOTH sessions have ended.
```

Note that neither `alice` nor `bob` calls `recorder.start()` or
`recorder.close()` — that is the caller's responsibility.

## Common Pitfalls

- **Calling `record()` from the wrong thread/loop** in async runtimes —
  always marshal onto the recorder's owning loop.
- **Blocking the event loop on the actual write** — the file write must
  happen on a worker thread in async runtimes.
- **Letting one bad record kill the writer** — wrap each write in a
  try/except and continue.
- **Hanging on `close()`** — always bound the drain with a timeout and
  cancel the writer if it overruns.
- **Recording before `start()`** or **after `close()`** — both should be
  no-ops with a warning, not crashes.
- **Batching multiple messages into one record** — never combine several
  frames into a single wrapper proto before serializing. Protobuf has a
  2 GB hard size limit per message (and a much lower default parse limit);
  a long audio session will blow past it. One record = one message.
- **Choosing any format other than length-prefixed `.pb`** — JSONL,
  TFRecord, `recordio`, and similar formats are not permitted. The
  recorder MUST always write a length-prefixed serialized protobuf
  stream to a `.pb` file.
