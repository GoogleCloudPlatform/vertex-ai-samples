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

What you should do:

Step 1:

Copy existing reference files to user provided destination folder

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

**Attention**

The service should reuse the `ServerMessage` and `ClientMessage` defined in the
proto for sending and receiving messages.

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
