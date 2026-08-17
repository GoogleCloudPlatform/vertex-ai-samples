# Live API `ClientMessage` and `ServerMessage` Reference

This document describes the **wire-level WebSocket protocol** for the Gemini Live
API as exposed by two public endpoints:

| Backend | Docs |
| --- | --- |
| **Gemini Enterprise Agent Platform** (`BidiGenerateContent*`) | https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/multimodal-live |
| **Google AI / Gemini Developer API** (`LiveClient*` / `LiveServer*`) | https://ai.google.dev/api/live |

The two services share the same underlying RPC (`BidiGenerateContent`) and the
same JSON/proto shapes, but differ in endpoint URL, auth, model name format,
and a handful of fields. Search corresponding websites for more information.

---

## 1. Connection

### Endpoints

| Backend | WebSocket URI |
| --- | --- |
| Gemini Enterprise Agent Platform | `wss://{LOCATION}-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent` |
| Gemini Enterprise Agent Platform (global) | `wss://aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent` |
| Google AI | `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent` |
| Google AI (ephemeral token) | `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContentConstrained` |

### Auth

- **Gemini Enterprise Agent Platform** — `Authorization: Bearer <ADC token>` (or API key in Express
  mode).
- **Google AI** — `?key=<API_KEY>` query parameter, **or**
  `Authorization: Token <ephemeral_token>` against the `Constrained` method.

### Frames

All frames are JSON-serialized protobuf messages of either `ClientMessage`
(client → server) or `ServerMessage` (server → client). Each frame is exactly
one WebSocket text message.

---

## 2. Top-level structure (oneof)

`ClientMessage` and `ServerMessage` are both **oneof envelopes** — each frame
sets exactly one of the listed fields.

### `ClientMessage` (you → server)

| Field | Type | When to use |
| --- | --- | --- |
| `setup` | `BidiGenerateContentSetup` | First frame only. Configures the session. |
| `clientContent` | `BidiGenerateContentClientContent` | Append to conversation history; turn-based input. Interrupts the model. |
| `realtimeInput` | `BidiGenerateContentRealtimeInput` | Continuous, low-latency audio/video/text input. Not added to history. |
| `toolResponse` | `BidiGenerateContentToolResponse` | Reply to a server-issued `toolCall`. |

### `ServerMessage` (server → you)

| Field | Type | Meaning |
| --- | --- | --- |
| `setupComplete` | `BidiGenerateContentSetupComplete` | Sent once after `setup` is accepted. Gate further sends on this. |
| `serverContent` | `BidiGenerateContentServerContent` | Streamed model output (audio/text), turn lifecycle. |
| `toolCall` | `BidiGenerateContentToolCall` | Model is requesting tool execution. |
| `toolCallCancellation` | `BidiGenerateContentToolCallCancellation` | Cancels previously-issued tool calls (e.g. on user interruption). |
| `usageMetadata` | `UsageMetadata` | Token / duration accounting. |
| `goAway` | `GoAway` | Connection will be terminated soon. |
| `sessionResumptionUpdate` | `SessionResumptionUpdate` | Resume handle for reconnects. |

> Audio transcriptions are delivered **inside `serverContent`** (see § 9), not
> as separate top-level frames.

---

## 3. Lifecycle

```
Client                                  Server
  │                                       │
  ├── ClientMessage{ setup }  ──────────► │
  │                                       │
  │ ◄──────  ServerMessage{ setupComplete }
  │                                       │
  ├── realtimeInput / clientContent ────► │
  │   (audio frames, text, etc.)          │
  │                                       │
  │ ◄──────  serverContent (audio chunks, modelTurn parts ...)
  │ ◄──────  serverContent { generationComplete: true }
  │ ◄──────  serverContent { turnComplete: true }
  │                                       │
  │ ◄──────  toolCall { functionCalls[] }
  ├── toolResponse { functionResponses[] } ► │
  │                                       │
  │ ◄──────  serverContent ...
  │ ◄──────  goAway { timeLeft }   (eventually)
  │                                       │
  │       (close & reconnect using sessionResumptionUpdate.newHandle)
```

**Rules:**

1. The first frame **must** be `setup`. Do not send anything else until you
   receive `setupComplete`.
2. Use **`clientContent`** for turn-based, history-affecting messages
   (e.g. a typed user message). Sending it interrupts any current model
   generation.
3. Use **`realtimeInput`** for continuous audio/video. It does **not** go
   into history. Turn boundaries come from VAD (or from explicit
   `activityStart`/`activityEnd` if VAD is disabled).
4. Reply to `toolCall` with `toolResponse` (never with `clientContent`).
5. On `goAway`, reconnect using the most recent
   `sessionResumptionUpdate.newHandle`.

---

## 4. `BidiGenerateContentSetup` (`setup`)

Initial-and-only-once configuration for the session.

| Field | Type | Notes |
| --- | --- | --- |
| `model` | `string` (required) | Gemini Enterprise: `projects/{p}/locations/{l}/publishers/google/models/{m}`. Google AI: `models/{m}`. |
| `generationConfig` | `GenerationConfig` | Unsupported sub-fields here: `responseLogprobs`, `responseMimeType`, `logprobs`, `responseSchema`, `stopSequence`, `routingConfig`, `audioTimestamp`. |
| `systemInstruction` | `Content` | Text-only parts. |
| `tools[]` | repeated `Tool` | Function declarations and built-ins (Search, code execution). |
| `sessionResumption` | `SessionResumptionConfig` | `{ handle?: string, transparent?: bool }`. Provide `handle` to resume; omit to start a new resumable session. |
| `contextWindowCompression` | `ContextWindowCompressionConfig` | `{ triggerTokens?: int64, slidingWindow?: { targetTokens?: int64 } }`. |
| `realtimeInputConfig` | `RealtimeInputConfig` | See below. |
| `inputAudioTranscription` | `AudioTranscriptionConfig` | Gemini Enterprise: empty type. Google AI: `{ languageCodes?: string[] }`. |
| `outputAudioTranscription` | `AudioTranscriptionConfig` | Same. |
| `proactivity` | `ProactivityConfig` | **Google AI only.** `{ proactiveAudio?: bool }`. |

### `RealtimeInputConfig`

| Field | Type | Notes |
| --- | --- | --- |
| `automaticActivityDetection` | `AutomaticActivityDetection` | Unset → server-side VAD enabled by default. |
| `activityHandling` | enum | `START_OF_ACTIVITY_INTERRUPTS` (default) \| `NO_INTERRUPTION`. |
| `turnCoverage` | enum | Gemini Enterprise default `TURN_INCLUDES_ALL_INPUT`; Google AI default `TURN_INCLUDES_ONLY_ACTIVITY`. Also `TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO`. |

### `AutomaticActivityDetection`

| Field | Type | Notes |
| --- | --- | --- |
| `disabled` | `bool` | If true, you must send `activityStart` / `activityEnd` yourself. |
| `startOfSpeechSensitivity` | enum | `START_SENSITIVITY_HIGH` \| `LOW`. |
| `endOfSpeechSensitivity` | enum | `END_SENSITIVITY_HIGH` \| `LOW`. |
| `prefixPaddingMs` | `int32` | Min speech duration to commit start-of-speech. |
| `silenceDurationMs` | `int32` | Min silence to commit end-of-speech. |

### `GenerationConfig` — Live-API-relevant subset

Fields most often used in `setup.generationConfig`:

| Field | Type | Notes |
| --- | --- | --- |
| `responseModalities[]` | repeated enum | `TEXT` \| `AUDIO`. Pick **one** per session (mixing not supported). Default `AUDIO`. |
| `temperature` | `float` | 0.0–2.0. |
| `topP`, `topK`, `maxOutputTokens` | various | Standard sampling / length controls. |
| `speechConfig` | `SpeechConfig` | Voice / language — only effective when `responseModalities=[AUDIO]`. See below. |
| `mediaResolution` | enum | `MEDIA_RESOLUTION_LOW` \| `MEDIUM` \| `HIGH`. Controls token-cost vs. quality of input images / video frames. |
| `thinkingConfig` | `ThinkingConfig` | `{ thinkingBudget?: int }`. Only on models that support thinking (e.g. Gemini 2.5 Flash). Set `thinkingBudget: 0` to disable. |

> **Unsupported in Live `setup.generationConfig`:** `responseLogprobs`,
> `responseMimeType`, `logprobs`, `responseSchema`, `stopSequence`,
> `routingConfig`, `audioTimestamp`.

### `SpeechConfig`

Controls the **voice** the model speaks with. Only meaningful when
`responseModalities` includes `AUDIO`.

| Field | Type | Notes |
| --- | --- | --- |
| `voiceConfig` | `VoiceConfig` | Single-speaker voice. The Live API supports **only single-speaker** output. |
| `languageCode` | `string` | BCP-47 (e.g. `en-US`, `de-DE`, `ja-JP`). Output language. Native-audio models auto-detect / switch on their own; for non-native-audio models, set this explicitly. See [Languages supported](#languages-supported). |

> **Note on multi-speaker output:** `MultiSpeakerVoiceConfig` /
> `multiSpeakerVoiceConfig` exist in the **standalone TTS API**
> (`generateContent` against models like `gemini-2.5-flash-preview-tts`,
> see <https://ai.google.dev/gemini-api/docs/speech-generation#multi-speaker>),
> **not** in the Live API. A Live API session has exactly one
> `prebuiltVoiceConfig.voiceName`. To approximate multiple speakers in a
> Live session, use prompt engineering inside `systemInstruction` or
> `clientContent` to have the single voice play different roles.

#### `VoiceConfig`

| Field | Type | Notes |
| --- | --- | --- |
| `prebuiltVoiceConfig` | `PrebuiltVoiceConfig` | Selects a named built-in voice. |

#### `PrebuiltVoiceConfig`

| Field | Type | Notes |
| --- | --- | --- |
| `voiceName` | `string` | Name of the prebuilt voice. **30 voices supported.** See [Voices supported](#voices-supported) below for the full list and references. |

##### Voices supported

The Live API supports **30 prebuilt voices**. Names are case-sensitive.

| Voice | Style | Voice | Style | Voice | Style |
| --- | --- | --- | --- | --- | --- |
| Zephyr | Bright | Puck | Upbeat | Charon | Informative |
| Kore | Firm | Fenrir | Excitable | Leda | Youthful |
| Orus | Firm | Aoede | Breezy | Callirrhoe | Easy-going |
| Autonoe | Bright | Enceladus | Breathy | Iapetus | Clear |
| Umbriel | Easy-going | Algieba | Smooth | Despina | Smooth |
| Erinome | Clear | Algenib | Gravelly | Rasalgethi | Informative |
| Laomedeia | Upbeat | Achernar | Soft | Alnilam | Firm |
| Schedar | Even | Gacrux | Mature | Pulcherrima | Forward |
| Achird | Friendly | Zubenelgenubi | Casual | Vindemiatrix | Gentle |
| Sadachbia | Lively | Sadaltager | Knowledgeable | Sulafat | Warm |

**References (authoritative voice list):**

- Gemini Enterprise Agent Platform: <https://cloud.google.com/gemini-enterprise-agent-platform/models/live-api/configure-language-voice#voices-supported>
- Google AI (Gemini Developer API) — Speech generation voices: <https://ai.google.dev/gemini-api/docs/speech-generation#voices>

> The available set may vary per model (e.g. native-audio vs. half-cascade
> models). If a voice is rejected during `setup`, the WebSocket closes
> instead of returning `setupComplete`.

#### Languages supported

The Live API supports **24 BCP-47 languages** for `speechConfig.languageCode`:

`ar-EG`, `bn-BD`, `de-DE`, `en-IN` (bundled with `hi-IN`), `en-US`, `es-US`,
`fr-FR`, `hi-IN`, `id-ID`, `it-IT`, `ja-JP`, `ko-KR`, `mr-IN`, `nl-NL`,
`pl-PL`, `pt-BR`, `ro-RO`, `ru-RU`, `ta-IN`, `te-IN`, `th-TH`, `tr-TR`,
`uk-UA`, `vi-VN`.

Reference: <https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/live-api/configure-language-voice#languages-supported>

> Native-audio models (e.g. `gemini-live-2.5-flash-native-audio`) can switch
> languages mid-conversation; for those, `languageCode` is optional and the
> model auto-detects. For non-native-audio models, set it explicitly.

#### Examples

Basic prebuilt voice:

```json
{
  "setup": {
    "model": "...",
    "generationConfig": {
      "responseModalities": ["AUDIO"],
      "speechConfig": {
        "voiceConfig": {
          "prebuiltVoiceConfig": { "voiceName": "Aoede" }
        }
      }
    }
  }
}
```

Voice + output language pinned to German:

```json
{
  "setup": {
    "model": "...",
    "generationConfig": {
      "responseModalities": ["AUDIO"],
      "speechConfig": {
        "voiceConfig": { "prebuiltVoiceConfig": { "voiceName": "Charon" } },
        "languageCode": "de-DE"
      }
    }
  }
}
```

> **Notes**
> - `speechConfig` is silently ignored if `responseModalities` is `["TEXT"]`.
> - `voiceName` is case-sensitive and model-specific. Sending an unknown name
>   typically fails the `setup` (you'll see the WebSocket close instead of
>   `setupComplete`).
> - `speechConfig.languageCode` controls **output** TTS language; the
>   **recognition** language for input audio is configured separately via
>   `inputAudioTranscription.languageCodes` (Google AI only).
> - The Live API is **single-speaker only**. Multi-speaker TTS
>   (`MultiSpeakerVoiceConfig`) is a feature of the standalone speech-generation
>   `generateContent` API, not Live.

---

## 5. `BidiGenerateContentClientContent` (`clientContent`)

Turn-based input. Append to history and (optionally) trigger generation.

| Field | Type | Notes |
| --- | --- | --- |
| `turns[]` | repeated `Content` | Conversation history + the latest user request. |
| `turnComplete` | `bool` | If true, server starts generating immediately. |

> Sending `clientContent` while the model is speaking **interrupts** it. Do
> not use `clientContent` to deliver `FunctionResponse`s — use
> `toolResponse`.

---

## 6. `BidiGenerateContentRealtimeInput` (`realtimeInput`)

Continuous, low-latency input. Does not populate history. End-of-turn is
derived from VAD (or activity events).

| Field | Type | Notes |
| --- | --- | --- |
| `mediaChunks[]` | repeated `Blob` | Combined audio/video chunks. Primary field on Gemini Enterprise raw WebSocket. |
| `audio` | `Blob` | Typed realtime audio. PCM 16-bit, 16 kHz mono input. (Google AI raw WebSocket; SDKs on both backends accept this.) |
| `video` | `Blob` | Typed realtime video frame (`image/jpeg`/`png`/`webp`). (Google AI raw WebSocket; SDKs on both backends accept this.) |
| `text` | `string` | Realtime text input. **Google AI only.** |
| `audioStreamEnd` | `bool` | Mic turned off. Only valid with auto-VAD enabled. |
| `activityStart` | `ActivityStart` (empty) | Only when auto-VAD is **disabled**. |
| `activityEnd` | `ActivityEnd` (empty) | Only when auto-VAD is **disabled**. |

**Audio formats**

- Input: `audio/pcm;rate=16000` (16 kHz, 16-bit signed PCM, mono, little-endian).
- Output: 24 kHz, 16-bit signed PCM, mono.

---

## 7. `BidiGenerateContentToolResponse` (`toolResponse`)

| Field | Type | Notes |
| --- | --- | --- |
| `functionResponses[]` | repeated `FunctionResponse` | Match `FunctionCall.id` from the server. **Google AI requires `id`**; Gemini Enterprise does not. |

---

## 8. `BidiGenerateContentSetupComplete` (`setupComplete`)

| Field | Type | Notes |
| --- | --- | --- |
| `sessionId` | `string` | Server-assigned session identifier. |

Receipt of this message is the gate for sending any other client frame.

---

## 9. `BidiGenerateContentServerContent` (`serverContent`)

Primary streaming-output channel.

| Field | Type | Notes |
| --- | --- | --- |
| `modelTurn` | `Content` | Streamed model output parts (text and/or `inlineData` audio). |
| `generationComplete` | `bool` | Model has finished generating; playback may still be flushing. |
| `turnComplete` | `bool` | Logical end of turn. |
| `interrupted` | `bool` | Generation was interrupted by client input — drop any queued audio playback. |
| `groundingMetadata` | `GroundingMetadata` | When grounding (e.g. Google Search) is used. |
| `inputTranscription` | `Transcription` | `{ text?: string }` — transcription of the user's spoken input. Requires `inputAudioTranscription` set in `setup`. |
| `outputTranscription` | `Transcription` | `{ text?: string }` — transcription of the model's spoken output. Requires `outputAudioTranscription` set in `setup`. |

**Playback note:** audio chunks arrive inside `modelTurn.parts[].inlineData`
with mimeType `audio/pcm;rate=24000`. Concatenate as they stream. On
`interrupted: true`, **flush** the playback queue.

---

## 10. `BidiGenerateContentToolCall` (`toolCall`)

| Field | Type | Notes |
| --- | --- | --- |
| `functionCalls[]` | repeated `FunctionCall` | Each has `id`, `name`, `args`. Reply with a matching `FunctionResponse.id`. |

---

## 11. `BidiGenerateContentToolCallCancellation` (`toolCallCancellation`)

| Field | Type | Notes |
| --- | --- | --- |
| `ids[]` | repeated `string` | IDs of previously-issued tool calls to cancel. Typically caused by user interruption. Stop the work; do not send `toolResponse` for these. |

---

## 12. `GoAway`, `SessionResumptionUpdate`, `UsageMetadata`

### `GoAway`

| Field | Type | Notes |
| --- | --- | --- |
| `timeLeft` | `Duration` (string) | Time until the connection terminates with `ABORTED`. Reconnect using the latest `SessionResumptionUpdate.newHandle`. |

### `SessionResumptionUpdate`

| Field | Type | Notes |
| --- | --- | --- |
| `newHandle` | `string` | Handle to resume; empty if `resumable=false`. |
| `resumable` | `bool` | Whether the session is currently resumable. |
| `lastConsumedClientMessageIndex` | `int64` | Set only when `SessionResumptionConfig.transparent=true` — enables transparent reconnect. |

### `UsageMetadata`

**Gemini Enterprise (cached-content style):**

| Field | Type |
| --- | --- |
| `totalTokenCount` | `int32` |
| `textCount` | `int32` |
| `imageCount` | `int32` |
| `videoDurationSeconds` | `int32` |
| `audioDurationSeconds` | `int32` |

**Google AI (per-response style):**

| Field | Type |
| --- | --- |
| `promptTokenCount` | `int32` |
| `cachedContentTokenCount` | `int32` |
| `responseTokenCount` | `int32` |
| `toolUsePromptTokenCount` | `int32` |
| `thoughtsTokenCount` | `int32` |
| `totalTokenCount` | `int32` |
| `promptTokensDetails[]` | repeated `ModalityTokenCount` |
| `cacheTokensDetails[]` | repeated `ModalityTokenCount` |
| `responseTokensDetails[]` | repeated `ModalityTokenCount` |
| `toolUsePromptTokensDetails[]` | repeated `ModalityTokenCount` |
| `trafficType` | enum (`PAYG` \| `PROVISIONED_THROUGHPUT`) |

---

## 13. End-to-end JSON examples

### Setup (Gemini Enterprise)

```json
{
  "setup": {
    "model": "projects/my-proj/locations/us-central1/publishers/google/models/gemini-2.0-flash-live-preview-04-09",
    "generationConfig": {
      "responseModalities": ["AUDIO"],
      "speechConfig": {
        "voiceConfig": { "prebuiltVoiceConfig": { "voiceName": "Aoede" } }
      }
    },
    "systemInstruction": {
      "parts": [{ "text": "You are a concise voice assistant." }]
    },
    "realtimeInputConfig": {
      "automaticActivityDetection": { "disabled": false }
    },
    "sessionResumption": {},
    "outputAudioTranscription": {}
  }
}
```

### Setup (Google AI)

```json
{
  "setup": {
    "model": "models/gemini-2.0-flash-live-001",
    "generationConfig": { "responseModalities": ["AUDIO"] },
    "outputAudioTranscription": {}
  }
}
```

### Setup complete (server)

```json
{ "setupComplete": { "sessionId": "abc-123" } }
```

### Streaming user audio (realtime)

```json
{
  "realtimeInput": {
    "audio": {
      "mimeType": "audio/pcm;rate=16000",
      "data": "<base64-pcm-bytes>"
    }
  }
}
```

(Gemini Enterprise equivalent uses `mediaChunks: [...]` instead of `audio:`.)

### Typed user message (turn-based)

```json
{
  "clientContent": {
    "turns": [{ "role": "user", "parts": [{ "text": "Hello!" }] }],
    "turnComplete": true
  }
}
```

### Streaming model output (server)

```json
{
  "serverContent": {
    "modelTurn": {
      "role": "model",
      "parts": [{
        "inlineData": {
          "mimeType": "audio/pcm;rate=24000",
          "data": "<base64-pcm-bytes>"
        }
      }]
    }
  }
}
```

### Tool call / response

```json
// Server → client
{
  "toolCall": {
    "functionCalls": [
      { "id": "call_42", "name": "get_weather", "args": { "city": "Paris" } }
    ]
  }
}

// Client → server
{
  "toolResponse": {
    "functionResponses": [
      {
        "id": "call_42",
        "name": "get_weather",
        "response": { "tempC": 18, "summary": "Partly cloudy" }
      }
    ]
  }
}
```

### GoAway + resumption

```json
{ "goAway": { "timeLeft": "10s" } }
{ "sessionResumptionUpdate": { "newHandle": "ses_xyz", "resumable": true } }
```

Reconnect with:

```json
{ "setup": { "model": "...", "sessionResumption": { "handle": "ses_xyz" } } }
```

---

## 14. Full input examples — audio, video, text

There are **two distinct ways** to send user input. Pick one based on intent:

| | Part 1 — Realtime input (`realtimeInput`) | Part 2 — Add-context input (`clientContent`) |
| --- | --- | --- |
| **Goal** | Stream live mic / camera / text continuously | Append a discrete turn to the conversation |
| **Latency** | Lowest possible; sub-second | Normal request/response |
| **Added to history?** | **No** (transient signals) | **Yes** (persistent conversation) |
| **Turn boundary** | VAD (auto), or explicit `activityStart`/`End` | Explicit `turnComplete: true` |
| **Effect on model** | Streamed; auto-triggers a turn when VAD fires | Setting `turnComplete` triggers generation; **interrupts** any ongoing model output |
| **Field carriers** | `audio` / `video` / `text` (Google AI) or `mediaChunks[]` (Gemini Enterprise) | `turns[].parts[].text` / `inlineData` / `fileData` |
| **Typical use** | Live voice + screen sharing, push-to-talk | Typed chat, uploading an image/clip, replaying history on resume |

> You may use both in the same session — e.g. send a `clientContent` system
> nudge once, then continue streaming `realtimeInput`. But within a single
> logical user turn, pick one.

The two parts below give every supported variant for **audio**, **video**,
and **text** in each mode, for both Gemini Enterprise and Google AI.

---

# Part 1 — Realtime input (`realtimeInput`)

Continuous, low-latency input that does **not** populate conversation
history. End-of-turn comes from server-side VAD by default, or from explicit
`activityStart` / `activityEnd` events when auto-VAD is disabled in `setup`.

## 1.A Realtime audio

Required format: **PCM, 16-bit signed, 16 kHz, mono, little-endian**, base64
encoded. MIME type: `audio/pcm;rate=16000`. Send one frame per chunk
(~20–100 ms is typical).

### 1.A.1 Continuous mic — Google AI

```json
{
  "realtimeInput": {
    "audio": {
      "mimeType": "audio/pcm;rate=16000",
      "data": "<base64 PCM chunk>"
    }
  }
}
```

When the mic turns off (and server-side VAD is enabled), commit end-of-stream:

```json
{ "realtimeInput": { "audioStreamEnd": true } }
```

### 1.A.2 Continuous mic — Gemini Enterprise

Gemini Enterprise uses the combined `mediaChunks[]` field:

```json
{
  "realtimeInput": {
    "mediaChunks": [
      { "mimeType": "audio/pcm;rate=16000", "data": "<base64 PCM chunk>" }
    ]
  }
}
```

### 1.A.3 Manual VAD (auto-VAD disabled) — both backends

Required `setup`:

```json
{
  "setup": {
    "model": "...",
    "realtimeInputConfig": { "automaticActivityDetection": { "disabled": true } }
  }
}
```

Then explicitly frame each utterance:

```json
{ "realtimeInput": { "activityStart": {} } }
{ "realtimeInput": { "audio": { "mimeType": "audio/pcm;rate=16000", "data": "..." } } }
{ "realtimeInput": { "audio": { "mimeType": "audio/pcm;rate=16000", "data": "..." } } }
{ "realtimeInput": { "activityEnd":   {} } }
```

## 1.B Realtime video

Video = a **stream of sampled image frames** (typically 1–2 fps for
screenshare, up to ~10 fps for camera). Each frame is JPEG / PNG / WebP. The
model does **not** consume an encoded container (mp4/webm) — sample frames
client-side and send each as an inline image.

### 1.B.1 Continuous camera — Google AI

```json
{
  "realtimeInput": {
    "video": {
      "mimeType": "image/jpeg",
      "data": "<base64 JPEG frame>"
    }
  }
}
```

### 1.B.2 Continuous camera — Gemini Enterprise

```json
{
  "realtimeInput": {
    "mediaChunks": [
      { "mimeType": "image/jpeg", "data": "<base64 JPEG frame>" }
    ]
  }
}
```

## 1.C Realtime text

### 1.C.1 — Google AI

```json
{ "realtimeInput": { "text": "Switch to a calmer tone." } }
```

### 1.C.2 — Gemini Enterprise

`realtimeInput.text` is **not supported** on Gemini Enterprise. To inject ad-hoc text
during a live session, fall back to **Part 2** (`clientContent`) — note that
sending `clientContent` will interrupt any ongoing model generation.

## 1.D Combined realtime audio + video (+ text)

Typical "talk-to-the-screen" flow.

### 1.D.1 — Google AI

```json
{ "realtimeInput": { "video": { "mimeType": "image/jpeg",          "data": "<frame_t0>" } } }
{ "realtimeInput": { "audio": { "mimeType": "audio/pcm;rate=16000", "data": "<pcm_t0>"   } } }
{ "realtimeInput": { "video": { "mimeType": "image/jpeg",          "data": "<frame_t1>" } } }
{ "realtimeInput": { "audio": { "mimeType": "audio/pcm;rate=16000", "data": "<pcm_t1>"   } } }
{ "realtimeInput": { "text":  "Focus on the chart in the upper-right." } }
{ "realtimeInput": { "audio": { "mimeType": "audio/pcm;rate=16000", "data": "<pcm_t2>"   } } }
{ "realtimeInput": { "audioStreamEnd": true } }
```

### 1.D.2 — Gemini Enterprise

Audio + video can be combined in a single frame via `mediaChunks[]`:

```json
{
  "realtimeInput": {
    "mediaChunks": [
      { "mimeType": "image/jpeg",          "data": "<frame_t0>" },
      { "mimeType": "audio/pcm;rate=16000", "data": "<pcm_t0>"   }
    ]
  }
}
{
  "realtimeInput": {
    "mediaChunks": [
      { "mimeType": "image/jpeg",          "data": "<frame_t1>" },
      { "mimeType": "audio/pcm;rate=16000", "data": "<pcm_t1>"   }
    ]
  }
}
```

## 1.E Realtime quick reference

| Modality | Google AI | Gemini Enterprise |
| --- | --- | --- |
| Audio chunk | `realtimeInput.audio` | `realtimeInput.mediaChunks[]` (`audio/pcm;rate=16000`) |
| Video frame | `realtimeInput.video` | `realtimeInput.mediaChunks[]` (`image/jpeg`/`png`/`webp`) |
| Text | `realtimeInput.text` | *(not supported — use Part 2)* |
| Mic-off signal | `realtimeInput.audioStreamEnd: true` | *(implicit on silence)* |
| Manual turn boundary | `realtimeInput.activityStart` / `activityEnd` (auto-VAD disabled) | same |

---

# Part 2 — Add-context input (`clientContent`)

Discrete, **history-bearing** turns. Use this when you want the message to be
permanently part of the conversation context the model sees on subsequent
turns. Setting `turnComplete: true` triggers generation immediately and
**interrupts** any ongoing model output.

> Do **not** use `clientContent` to reply to a `toolCall` — use `toolResponse`.

## 2.A Add-context text

### 2.A.1 Single user turn — both backends

```json
{
  "clientContent": {
    "turns": [
      { "role": "user", "parts": [{ "text": "What's the capital of France?" }] }
    ],
    "turnComplete": true
  }
}
```

### 2.A.2 Multi-turn history (e.g. on resume)

```json
{
  "clientContent": {
    "turns": [
      { "role": "user",  "parts": [{ "text": "Hi, my name is Sam." }] },
      { "role": "model", "parts": [{ "text": "Nice to meet you, Sam!" }] },
      { "role": "user",  "parts": [{ "text": "What's my name?" }] }
    ],
    "turnComplete": true
  }
}
```

### 2.A.3 Streamed turn (don't generate yet — more parts coming)

```json
{ "clientContent": { "turns": [{ "role": "user", "parts": [{ "text": "Once upon" }] }] } }
{ "clientContent": { "turns": [{ "role": "user", "parts": [{ "text": " a time..." }] }], "turnComplete": true } }
```

## 2.B Add-context audio

A pre-recorded clip delivered as a discrete history-bearing turn.

### 2.B.1 Audio clip alone — both backends

```json
{
  "clientContent": {
    "turns": [{
      "role": "user",
      "parts": [{
        "inlineData": {
          "mimeType": "audio/pcm;rate=16000",
          "data": "<base64 of full clip>"
        }
      }]
    }],
    "turnComplete": true
  }
}
```

### 2.B.2 Audio clip with accompanying text prompt

```json
{
  "clientContent": {
    "turns": [{
      "role": "user",
      "parts": [
        { "text": "Transcribe this and summarize:" },
        { "inlineData": { "mimeType": "audio/pcm;rate=16000", "data": "..." } }
      ]
    }],
    "turnComplete": true
  }
}
```

## 2.C Add-context video / image

### 2.C.1 Single image — both backends

```json
{
  "clientContent": {
    "turns": [{
      "role": "user",
      "parts": [
        { "text": "What's in this picture?" },
        { "inlineData": { "mimeType": "image/jpeg", "data": "<base64 image>" } }
      ]
    }],
    "turnComplete": true
  }
}
```

### 2.C.2 Multiple frames as a single turn (sampled clip)

```json
{
  "clientContent": {
    "turns": [{
      "role": "user",
      "parts": [
        { "text": "Here are 3 frames from a video. What's happening?" },
        { "inlineData": { "mimeType": "image/jpeg", "data": "<frame_0>" } },
        { "inlineData": { "mimeType": "image/jpeg", "data": "<frame_1>" } },
        { "inlineData": { "mimeType": "image/jpeg", "data": "<frame_2>" } }
      ]
    }],
    "turnComplete": true
  }
}
```

### 2.C.3 Image by URI (`fileData`) — Gemini Enterprise

```json
{
  "clientContent": {
    "turns": [{
      "role": "user",
      "parts": [
        { "text": "Describe this image." },
        { "fileData": { "mimeType": "image/jpeg", "fileUri": "gs://my-bucket/cat.jpg" } }
      ]
    }],
    "turnComplete": true
  }
}
```

### 2.C.4 Image by URI (`fileData`) — Google AI

Upload via the Files API first, then reference the returned URI:

```json
{
  "clientContent": {
    "turns": [{
      "role": "user",
      "parts": [
        { "text": "Describe this image." },
        {
          "fileData": {
            "mimeType": "image/jpeg",
            "fileUri": "https://generativelanguage.googleapis.com/v1beta/files/abc-123"
          }
        }
      ]
    }],
    "turnComplete": true
  }
}
```

## 2.D Combined add-context multimodal turn

Text + image + audio in a single user turn:

```json
{
  "clientContent": {
    "turns": [{
      "role": "user",
      "parts": [
        { "text": "Compare what I'm saying with what I'm showing:" },
        { "inlineData": { "mimeType": "image/jpeg",          "data": "<image>" } },
        { "inlineData": { "mimeType": "audio/pcm;rate=16000", "data": "<clip>"  } }
      ]
    }],
    "turnComplete": true
  }
}
```

## 2.E Add-context quick reference

| Modality | Field | Notes |
| --- | --- | --- |
| Text | `turns[].parts[].text` | One or more `text` parts per turn. |
| Inline audio | `turns[].parts[].inlineData` (`audio/pcm;rate=16000`) | Full clip, base64. |
| Inline image / video frame | `turns[].parts[].inlineData` (`image/jpeg`/`png`/`webp`) | Multiple parts allowed for sampled clips. |
| Remote file | `turns[].parts[].fileData` (`fileUri`) | Gemini Enterprise: GCS URI. Google AI: Files-API URI. |
| Trigger generation | `turnComplete: true` | Omit to keep streaming more parts. |
| History order | `turns[]` ordered oldest → newest | `role` is `"user"` or `"model"`. |

---

## 15. Gemini Enterprise vs Google AI differences

| Area | Gemini Enterprise | Google AI |
| --- | --- | --- |
| Type prefix | `BidiGenerateContent*` | `LiveClient*` / `LiveServer*` (wire types still `BidiGenerateContent*`) |
| Endpoint host | `{location}-aiplatform.googleapis.com` (or global) | `generativelanguage.googleapis.com` |
| Service path | `google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent` | `google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent` |
| Auth | OAuth Bearer (ADC) or Express-mode API key | API key query param; ephemeral `Authorization: Token` against `BidiGenerateContentConstrained` (v1alpha) |
| `model` format | `projects/{p}/locations/{l}/publishers/google/models/{m}` | `models/{m}` |
| `FunctionResponse.id` | Optional | **Required** |
| `AudioTranscriptionConfig` | Empty | `{ languageCodes?: string[] }` |
| `realtimeInput` raw-WS shape | Primary field is `mediaChunks[]` | Typed fields `audio` / `video` / `text` / `audioStreamEnd` |
| `realtimeInput.text` | Not supported (use `clientContent`) | Supported |
| Default `turnCoverage` | `TURN_INCLUDES_ALL_INPUT` | `TURN_INCLUDES_ONLY_ACTIVITY` |
| `UsageMetadata` shape | Cached-content counters (`totalTokenCount`, `textCount`, `imageCount`, `videoDurationSeconds`, `audioDurationSeconds`) | Per-response counters with modality details and `trafficType` |
| `ProactivityConfig` (`proactiveAudio`) | Not present | Supported |
| Default API version | `v1beta1` (also `v1`) | `v1beta` (also `v1alpha` for ephemeral tokens) |

---

## 16. Common pitfalls

- **Sending data before `setupComplete`** — server will close the connection.
- **Mixing `clientContent` and `realtimeInput`** for the same logical turn —
  pick one mode. `clientContent` interrupts ongoing generation.
- **Ignoring `interrupted: true`** — leftover queued audio will play over the
  user's next utterance.
- **Replying to `toolCall` with `clientContent`** — must be `toolResponse`.
- **Forgetting `FunctionResponse.id` on Google AI** — request rejected.
- **Wrong audio format** — input must be 16 kHz PCM, output is 24 kHz PCM.
- **No reconnect handling** — sessions have a max duration; always honor
  `goAway` and persist the latest `SessionResumptionUpdate.newHandle`.
- **Wrong model-name format** between Gemini Enterprise (`projects/.../models/...`) and
  Google AI (`models/...`).
