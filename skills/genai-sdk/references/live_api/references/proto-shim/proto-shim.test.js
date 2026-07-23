/**
 * Conformance test for proto-shim.js (R7 in proto-shim.md).
 *
 * Builds a `ClientMessage` with a non-trivial inner field, serializes
 * it through the shim, asserts the encoded bytes are NOT the
 * known-broken zero-length stub, then deserializes and checks the
 * round-tripped value equals the original.
 *
 * Usage:
 *   node references/proto-shim/proto-shim.test.js
 *
 * Exits 0 on PASS, 1 on FAIL.
 *
 * This test exists because the shim's most likely failure mode (R1:
 * `field.resolvedType` not populated -> message-typed setter stores
 * the wrapper -> wire payload is 2 bytes of `1a 00`) produces a clean
 * encode + clean decode but loses the field's contents. A naive
 * "did serializeBinary succeed?" test passes on the broken shim; only
 * the wire-length + value-equality assertion below catches it.
 */

'use strict';

const path = require('path');
const fs = require('fs');

// Locate sibling files. This test is intentionally zero-dependency:
// it expects `protobuf.min.js`, `proto-shim.js`, and the canonical
// `client_server_messages.proto` (plus its well-known imports under
// `google/protobuf/`) to be reachable from the script's own dir or
// from a sibling `web/proto/` tree.
const HERE = __dirname;
const PROTO_ROOTS = [
  HERE,                                                  // alongside the shim
  path.join(HERE, '..'),                                 // skill references/ dir
  path.join(HERE, '..', '..', 'web', 'proto'),           // typical embedded layout
];

function findFile(rel) {
  for (const root of PROTO_ROOTS) {
    const p = path.join(root, rel);
    if (fs.existsSync(p)) return p;
  }
  return null;
}

const protobufJsPath = findFile('protobuf.min.js');
if (!protobufJsPath) {
  console.error('FAIL: protobuf.min.js not found. ' +
      'Vendor it next to this test (see proto-shim.md § R6).');
  process.exit(1);
}

const shimPath = path.join(HERE, 'proto-shim.js');
if (!fs.existsSync(shimPath)) {
  console.error('FAIL: proto-shim.js not found next to this test.');
  process.exit(1);
}

const protoPath = findFile('client_server_messages.proto');
if (!protoPath) {
  console.error('FAIL: client_server_messages.proto not found.');
  process.exit(1);
}
const PROTO_DIR = path.dirname(protoPath);

// ---- Set up a minimal browser-like environment ----------------------------
global.window = global;
global.atob = (b) => Buffer.from(b, 'base64').toString('binary');
global.btoa = (s) => Buffer.from(s, 'binary').toString('base64');

// Load protobufjs and re-route the shim's `/proto/...` URL into the
// local filesystem.
global.protobuf = require(protobufJsPath);
const origLoad = global.protobuf.load;
global.protobuf.load = (url, ...rest) => {
  let abs = url;
  if (url.startsWith('/proto/')) {
    abs = path.join(PROTO_DIR, url.slice('/proto/'.length));
  } else if (!path.isAbsolute(url)) {
    abs = path.join(PROTO_DIR, url);
  }
  return origLoad(abs, ...rest);
};

// Now evaluate the shim. It is an IIFE that registers globals.
require(shimPath);

(async () => {
  await window.protoReady;

  // ---- R1 sanity: the wrapper classes were created ----
  for (const name of ['ClientMessage', 'ServerMessage',
                      'BidiGenerateContentRealtimeInput',
                      'BidiGenerateContentClientContent', 'Blob']) {
    if (typeof window[name] !== 'function') {
      console.error('FAIL: shim did not register global', name);
      process.exit(1);
    }
  }

  // ---- R2 / R7: realtimeInput.text round-trip ----
  {
    const ri = new window.BidiGenerateContentRealtimeInput().setText('ping');
    const cm = new window.ClientMessage().setRealtimeInput(ri);
    const wire = cm.serializeBinary();
    if (!(wire instanceof Uint8Array)) {
      console.error('FAIL: serializeBinary did not return Uint8Array');
      process.exit(1);
    }
    if (wire.length <= 2) {
      // 1a 00 is the canonical broken-shim signature (realtime_input
      // tag with zero-length payload). Any test that passes here must
      // ALSO assert on length > 2 to catch it.
      console.error('FAIL: encoded ClientMessage is suspiciously short ' +
          '(' + wire.length + ' bytes); this is the R1 broken-setter signature.');
      console.error('       Wire bytes (hex): ' +
          Buffer.from(wire).toString('hex'));
      process.exit(1);
    }
    const back = window.ClientMessage.deserializeBinary(wire);
    const json = back.toObject();
    if (!json.realtimeInput || json.realtimeInput.text !== 'ping') {
      console.error('FAIL: round-trip lost realtimeInput.text. Got:',
          JSON.stringify(json));
      process.exit(1);
    }
  }

  // ---- R3: Blob bytes round-trip ----
  {
    const bytes = new Uint8Array([0x01, 0x02, 0x03, 0x04]);
    const blob = new window.Blob().setMimeType('audio/pcm;rate=16000').setData(bytes);
    const ri = new window.BidiGenerateContentRealtimeInput().setAudio(blob);
    const cm = new window.ClientMessage().setRealtimeInput(ri);
    const wire = cm.serializeBinary();
    const back = window.ClientMessage.deserializeBinary(wire);
    // hasInlineData-style accessor lives on Part; for the realtimeInput
    // arm the audio Blob is at .audio. Pull via the shim's accessor.
    const ri2 = back.getRealtimeInput();
    const audio = ri2.getAudio();
    if (!audio) {
      console.error('FAIL: round-trip lost realtimeInput.audio');
      process.exit(1);
    }
    if (audio.getMimeType() !== 'audio/pcm;rate=16000') {
      console.error('FAIL: mime_type lost. Got:', audio.getMimeType());
      process.exit(1);
    }
    const data = audio.getData();
    if (!(data instanceof Uint8Array)) {
      console.error('FAIL: Blob.getData() must return Uint8Array; got',
          data && data.constructor && data.constructor.name);
      process.exit(1);
    }
    if (data.length !== 4 || data[0] !== 1 || data[3] !== 4) {
      console.error('FAIL: Blob.data round-trip lost bytes. Got:',
          Array.from(data));
      process.exit(1);
    }
  }

  // ---- R4: setting a second oneof arm clears the first ----
  {
    const cm = new window.ClientMessage()
        .setRealtimeInput(new window.BidiGenerateContentRealtimeInput().setText('a'))
        .setClientContent(new window.BidiGenerateContentClientContent());
    const wire = cm.serializeBinary();
    const back = window.ClientMessage.deserializeBinary(wire);
    if (back.hasRealtimeInput && back.hasRealtimeInput()) {
      console.error('FAIL: setting second oneof arm did not clear the first');
      process.exit(1);
    }
    if (!back.hasClientContent || !back.hasClientContent()) {
      console.error('FAIL: second oneof arm is not present after switch');
      process.exit(1);
    }
  }

  console.log('PASS');
})().catch((err) => {
  console.error('FAIL:', err && err.stack || err);
  process.exit(1);
});
