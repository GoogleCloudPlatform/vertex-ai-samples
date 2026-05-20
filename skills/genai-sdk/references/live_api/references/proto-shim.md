---
name: liveapi-proto-shim
description: >-
  Contract and reference implementation for the browser-side proto shim
  that bridges the playground / viewer reference scripts (written for
  the Closure-style google-protobuf JS API: `getX`/`setX`/`hasX`/
  `serializeBinary`/`deserializeBinary`) onto the reflective
  `protobufjs` runtime that loads `client_server_messages.proto` at
  runtime. Use when the chat / viewer frontend is generated in a
  language whose canonical proto runtime is not the Closure compiler's
  google-protobuf-js bundle.
---

# LiveAPI Browser Proto Shim

The reference frontends (`playground_frontend/script.js`,
`recording_viewer_frontend/script.js`) are written against the
google-protobuf JS API:

```js
const msg = new ClientMessage()
    .setRealtimeInput(new BidiGenerateContentRealtimeInput()
        .setAudio(new Blob().setMimeType('audio/pcm;rate=16000').setData(bytes)));
const wire = msg.serializeBinary();   // Uint8Array

const parsed = ServerMessage.deserializeBinary(event.data);
if (parsed.hasServerContent()) { ... }
```

A non-trivial number of generated frontends will not have those
Closure-style classes available; they will instead load
`client_server_messages.proto` via the lighter `protobufjs` runtime.
The shim defined here exposes the Closure-style classes as window
globals on top of protobufjs so the reference scripts work
unmodified.

This contract has tripped up enough adapters that the skill ships a
tested reference implementation
(`proto-shim/proto-shim.js` + `proto-shim/proto-shim.test.js`); use
that as the starting point rather than rewriting from scratch.

---

## Hard requirements

A conformant shim MUST satisfy ALL of the following. Each requirement
maps to a known failure mode; violating any one of them silently
breaks the chat UI (and frequently the recording viewer as well).

### R1. Resolve message references before iterating fields

`protobufjs` lazily resolves `field.resolvedType` for message-typed
fields. If the shim iterates `Type.fields` immediately after
`protobuf.load(...)`, every `field.resolvedType` is `null` and the
shim's setters cannot distinguish message-typed fields from
primitives. The resulting setter writes the wrapper *itself* onto the
parent object, which encodes as zero bytes on the wire.

**Required:** call `root.resolveAll()` (or the equivalent in your
runtime) **before** building any accessors.

```js
const root = await protobuf.load('/proto/client_server_messages.proto');
root.resolveAll();                 // <-- REQUIRED
```

### R2. Setters for message-typed fields MUST unwrap

When the caller invokes `cm.setRealtimeInput(riWrapper)`, the inner
`protobufjs` message instance is `riWrapper.msg`, NOT `riWrapper`.
Storing the wrapper directly produces empty payloads on the wire (see
R1 for the typical cause).

```js
// Correct:
this.msg[fieldName] = val && val.msg ? val.msg : val;
```

### R3. `bytes` field accessors MUST return raw `Uint8Array`

The reference `playAudioChunk` does:

```js
const pcm = new Int16Array(audioData.buffer);
```

`audioData` here is the result of `part.getInlineData().getData()`.
That expression MUST return a `Uint8Array`; returning a wrapper
object whose `.buffer` is not an `ArrayBuffer` silently produces
garbage. Accept either `Uint8Array` or base64 `string` on the input
side (`setData(...)`); always return `Uint8Array` on the output side.

### R4. Oneof helpers MUST clear sibling arms

When `cm.setRealtimeInput(...)` is called after a previous
`cm.setClientContent(...)`, the previous arm MUST be cleared so the
encoded frame contains a single arm. `protobufjs` handles this
automatically when you assign to the active oneof field; the shim
needs only to forward to that mechanism (no extra cleanup needed).

### R5. `deserializeBinary` MUST return a wrapper, not a raw message

Reference scripts call `ServerMessage.deserializeBinary(event.data)`
and then `message.hasServerContent()`. Returning the raw `protobufjs`
message breaks the Closure-style API surface.

### R6. The shim MUST be vendored, not loaded from a CDN

If the runtime is fetched from a public CDN at chat-page load time,
a blocked CDN (corporate proxy, NoScript, slow network) leaves
`window.protoReady` pending forever and the chat page presents as a
spinning blank. Vendor `protobuf.min.js` (or the equivalent) under
the static-asset tree. Surface a visible error banner on
`protoReady` rejection so the failure isn't invisible to users with
dev-tools closed.

### R7. The shim MUST be tested with a round-trip assertion

A passing JS unit test MUST exist that:

1. Builds a non-trivial `ClientMessage` (e.g. `realtimeInput.text =
   "ping"`) using the shim's `setX` setters.
2. Calls `serializeBinary()`.
3. Asserts the resulting `Uint8Array.length` is **strictly greater
   than 2** (a 2-byte payload of `1a 00` is the single-arm-tag +
   zero-length-bytes stub that the broken-setter bug produces).
4. Calls `ClientMessage.deserializeBinary(wire)` and asserts the
   resulting message round-trips the original field.

Without (3) the broken-setter bug silently passes naive tests that
only check "did encode + decode succeed?" — both succeed on the
broken path, they just lose the payload.

---

## Reference implementation

`proto-shim/proto-shim.js` (this directory) is the canonical
implementation that satisfies R1 – R5 + the structural part of R6.
Vendor it verbatim into the generated project under the same
`web/proto/proto-shim.js` path the chat HTML imports.

Vendor `protobuf.min.js` alongside it
(`web/proto/protobuf.min.js`) from your preferred npm registry
(version pin recommended).

The well-known `.proto` files imported by `client_server_messages.proto`
(`google/protobuf/duration.proto` and `google/protobuf/struct.proto`)
MUST also be present under `web/proto/google/protobuf/` so the
runtime's import resolver can find them. They ship with every
`protoc` distribution.

### Minimum file layout

```
web/proto/
├── client_server_messages.proto      # the canonical proto, served verbatim
├── protobuf.min.js                   # vendored protobufjs runtime
├── proto-shim.js                     # this reference shim
└── google/protobuf/
    ├── duration.proto                # imported by client_server_messages.proto
    └── struct.proto                  # imported by client_server_messages.proto
```

### Minimum HTML wiring

In both `web/chat/index.html` and `web/viewer/index.html` (the
viewer reference doesn't currently need the shim, but the shared
shell can preload it once for both surfaces):

```html
<script src="/proto/protobuf.min.js"></script>
<script src="/proto/proto-shim.js"></script>
<script src="script.js"></script>
```

The chat script's bootstrap MUST `await window.protoReady` before
calling any shim global, and MUST render a visible banner if
`protoReady` rejects.

---

## Conformance test (R7)

A Node-runnable test harness is included alongside the reference
shim: `proto-shim/proto-shim.test.js`. It loads the shim in a fake
window context, builds the round-trip assertion described in R7,
and exits non-zero on failure. Run it as part of the Step-8
verification gate when the generated frontend uses this shim.

```
node references/proto-shim/proto-shim.test.js
# expected: PASS
```

If the test fails, do not ship the generated frontend; the failure
indicates one of R1 – R5 is violated and the chat UI will produce
empty wire frames.
