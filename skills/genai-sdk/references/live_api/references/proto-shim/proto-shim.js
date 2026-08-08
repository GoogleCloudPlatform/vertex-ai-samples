/**
 * Proto bootstrap shim.
 *
 * The reference frontends in `references/playground_frontend/script.js`
 * and `references/recording_viewer_frontend/script.js` were written
 * against the *google-protobuf* JS runtime (Closure-style API with
 * `getX()`, `setX(v)`, `hasX()`, `serializeBinary()`,
 * `Foo.deserializeBinary(bytes)` and oneof helpers like
 * `setRealtimeInput(child)` mutating the parent's oneof arm).
 *
 * We don't want to ship a CommonJS-bundled google-protobuf build here,
 * so this shim builds the same surface dynamically on top of
 * `protobufjs` (the small reflective runtime loaded from CDN). It
 * exposes the global classes the reference scripts expect:
 *
 *   ClientMessage, ServerMessage,
 *   BidiGenerateContentSetup,
 *   BidiGenerateContentClientContent,
 *   BidiGenerateContentRealtimeInput,
 *   Blob, FileData, Content, Part,
 *   FunctionCall, FunctionResponse, ...
 *
 * Each global is a thin wrapper that:
 *   - Mirrors every proto field as `getFieldName()` / `setFieldName(v)` /
 *     `hasFieldName()` accessors (camelCase).
 *   - For *message* fields, returns / accepts wrapped instances of the
 *     corresponding wrapper class.
 *   - For *bytes* fields, `setX` accepts Uint8Array | base64 string;
 *     `getX` returns a Uint8Array.
 *   - Supports `serializeBinary()` → Uint8Array, and a static
 *     `Klass.deserializeBinary(Uint8Array)` → wrapper instance.
 *   - For *oneof* fields, calling `setArmName(child)` writes the
 *     correct oneof arm; the other arms are cleared automatically by
 *     protobuf.js.
 *
 * One global promise, `window.protoReady`, resolves once the runtime
 * has finished loading the `.proto` file and exposing every class. The
 * shared shell waits on this before activating either surface.
 */

(function () {
  'use strict';

  // protobuf.js global is `protobuf` (added by the CDN script).
  if (typeof protobuf === 'undefined') {
    console.error('proto-shim: protobufjs runtime not loaded');
    return;
  }

  const PROTO_URL = '/proto/client_server_messages.proto';
  const PROTO_PACKAGE = 'google.ai.live.v1';

  // Names from the .proto we want to expose as window globals. (Other
  // message types are still reachable via the protobuf.js root, but the
  // reference frontends only construct these by name.)
  const EXPOSED = [
    'ClientMessage',
    'ServerMessage',
    'BidiGenerateContentSetup',
    'BidiGenerateContentClientContent',
    'BidiGenerateContentRealtimeInput',
    'BidiGenerateContentToolResponse',
    'BidiGenerateContentServerContent',
    'BidiGenerateContentSetupComplete',
    'BidiGenerateContentToolCall',
    'BidiGenerateContentToolCallCancellation',
    'GenerationConfig',
    'SpeechConfig',
    'VoiceConfig',
    'PrebuiltVoiceConfig',
    'SessionResumptionConfig',
    'ContextWindowCompressionConfig',
    'RealtimeInputConfig',
    'AutomaticActivityDetection',
    'AudioTranscriptionConfig',
    'ProactivityConfig',
    'ThinkingConfig',
    'SessionResumptionUpdate',
    'GoAway',
    'UsageMetadata',
    'Content',
    'Part',
    'Blob',
    'FileData',
    'FunctionCall',
    'FunctionResponse',
    'Transcription',
  ];

  // Convert snake_case -> camelCase (protobuf JSON convention).
  function camel(name) {
    return name.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
  }

  // Convert snake_case -> PascalCase (for accessor method names:
  // getFooBar / setFooBar).
  function pascal(name) {
    const c = camel(name);
    return c.charAt(0).toUpperCase() + c.slice(1);
  }

  // Make a fresh Wrapper class for a given protobuf.js Type. The
  // wrapper holds a single `.msg` (the protobuf.js plain object) and
  // exposes the Closure-style methods on top of it.
  function makeWrapper(Type, wrappers) {
    class Wrapper {
      constructor(msg) {
        // `msg` is a protobuf.js message instance (created via
        // Type.create). When omitted, allocate a fresh empty one.
        this._type = Type;
        this.msg = msg || Type.create();
      }

      // -------- binary serialization --------
      serializeBinary() {
        // protobuf.js's `encode` returns a Writer; finish() yields a
        // Uint8Array.
        return Type.encode(this.msg).finish();
      }

      static deserializeBinary(bytes) {
        // Tolerate Uint8Array OR ArrayBuffer OR base64 string.
        if (typeof bytes === 'string') {
          bytes = base64ToBytes(bytes);
        } else if (bytes instanceof ArrayBuffer) {
          bytes = new Uint8Array(bytes);
        }
        const inst = Type.decode(bytes);
        return new Wrapper(inst);
      }

      // toObject() is used by the chat UI to pretty-print server
      // frames; mirror google-protobuf's default JSON output.
      toObject() {
        return Type.toObject(this.msg, {
          enums: String,
          bytes: String, // base64
          longs: String,
          defaults: false,
          arrays: true,
        });
      }
    }

    // Build a per-oneof sibling map so each setter can clear the
    // other arms when it sets one. Required for R4 in `proto-shim.md`:
    // without this, calling setRealtimeInput then setClientContent
    // emits both fields on the wire (decoded by strict consumers as
    // "oneof is already set").
    const oneofSiblings = {}; // camelCase field name -> [other camelCase names]
    for (const oneofName of Object.keys(Type.oneofs || {})) {
      const fnames = Type.oneofs[oneofName].oneof.map(camel);
      for (const f of fnames) {
        oneofSiblings[f] = fnames.filter((x) => x !== f);
      }
    }

    // Stamp accessors for every declared field. We do this once per
    // class at load time; per-field overhead at call time is just a
    // property lookup.
    for (const fieldName of Object.keys(Type.fields)) {
      const field = Type.fields[fieldName];
      const cap = pascal(fieldName);
      const camelKey = camel(fieldName); // protobuf.js stores keys as camelCase
      const siblings = oneofSiblings[camelKey] || [];

      // Determine the wrapper to use when the field is a message.
      const isMessage = field.resolvedType && field.resolvedType.fields;
      const isMap = field.map;
      const isRepeated = field.repeated;
      const isBytes = field.type === 'bytes';

      // getter
      Wrapper.prototype['get' + cap + (isRepeated ? 'List' : '')] = function () {
        let v = this.msg[camelKey];
        if (v === undefined || v === null) {
          if (isRepeated) return [];
          if (isBytes) return new Uint8Array();
          return null;
        }
        if (isMessage) {
          if (isRepeated) {
            return v.map((child) => {
              const W = wrappers.get(field.resolvedType.fullName) || Wrapper;
              return new W(child);
            });
          }
          const W = wrappers.get(field.resolvedType.fullName) || Wrapper;
          return new W(v);
        }
        if (isBytes && typeof v === 'string') {
          return base64ToBytes(v);
        }
        if (isBytes) return new Uint8Array(v);
        return v;
      };

      // setter
      Wrapper.prototype['set' + cap + (isRepeated ? 'List' : '')] = function (val) {
        if (val === null || val === undefined) {
          delete this.msg[camelKey];
          return this;
        }
        // If this field is part of a oneof, clear the sibling arms
        // first. Otherwise the encoded frame can contain TWO arms,
        // which violates the oneof contract and produces "oneof is
        // already set" errors on strict decoders downstream.
        for (const sib of siblings) delete this.msg[sib];
        if (isMessage) {
          if (isRepeated) {
            this.msg[camelKey] = val.map((w) => (w && w.msg ? w.msg : w));
          } else {
            this.msg[camelKey] = val && val.msg ? val.msg : val;
          }
          return this;
        }
        if (isBytes) {
          if (typeof val === 'string') {
            this.msg[camelKey] = base64ToBytes(val);
          } else {
            this.msg[camelKey] = val instanceof Uint8Array ? val : new Uint8Array(val);
          }
          return this;
        }
        this.msg[camelKey] = val;
        return this;
      };

      // hasX (matters mostly for singular message fields)
      Wrapper.prototype['has' + cap] = function () {
        const v = this.msg[camelKey];
        return v !== undefined && v !== null;
      };
    }

    // Convenience for oneof: expose `getXCase()` returning the active
    // arm's PascalCase field name, mirroring google-protobuf.
    for (const [oneofName, oneofFields] of Object.entries(Type.oneofs || {})) {
      const cap = pascal(oneofName);
      Wrapper.prototype['get' + cap + 'Case'] = function () {
        for (const fname of oneofFields.oneof) {
          if (this.msg[camel(fname)] !== undefined && this.msg[camel(fname)] !== null) {
            return pascal(fname);
          }
        }
        return null;
      };
    }

    Wrapper._typeName = Type.fullName;
    return Wrapper;
  }

  function base64ToBytes(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  // Kick off the load and stash the resulting promise globally so the
  // application bootstrappers (chat + viewer) can `await` it once.
  window.protoReady = (async function () {
    const root = await protobuf.load(PROTO_URL);
    // CRITICAL: resolveAll() forces protobuf.js to populate
    // `field.resolvedType` on every message field. Without this,
    // wrapper setters for message-typed fields (e.g. `setRealtimeInput`)
    // can't tell the field is a message and fall through to the
    // primitive branch, storing the *Wrapper instance* instead of the
    // inner protobuf message. The resulting frames encode as empty
    // bytes for that field on the wire.
    root.resolveAll();
    const ns = root.lookup(PROTO_PACKAGE);
    if (!ns) {
      throw new Error('proto-shim: package ' + PROTO_PACKAGE + ' not found');
    }
    const wrappers = new Map();
    // First pass: create wrapper classes for every requested type.
    for (const name of EXPOSED) {
      const Type = ns.lookupTypeOrEnum(name);
      if (!Type || !Type.fields) continue; // enums skipped
      const W = makeWrapper(Type, wrappers);
      wrappers.set(Type.fullName, W);
      window[name] = W;
    }
    // Second pass: any wrapper class that references another by message
    // type needs the map filled, which we now have. (makeWrapper looks
    // up wrappers lazily inside getters/setters, so no re-stamp needed.)
    return wrappers;
  })().catch((err) => {
    console.error('proto-shim: failed to load proto', err);
    throw err;
  });
})();
