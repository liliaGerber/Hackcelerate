import uuid
import json
import os

from flask import Flask, request, jsonify
import azure.cognitiveservices.speech as speechsdk
from flask_sock import Sock
from flask_cors import CORS
from flasgger import Swagger

from openai import OpenAI
import io

AZURE_SPEECH_KEY = "BJ953og5dx7zhJDZayAhkShitCd8qHjlGTb0oimuxHsRJ5ldiL2IJQQJ99BCACI8hq2XJ3w3AAAYACOGYWu0"
AZURE_SPEECH_REGION = "switzerlandnorth"
OPENAI_KEY = "sk-svcacct-kNbLuIwHgQkPaf2BofPRVwKzMtlQpKiDwGrpkq_poibG1tmEQC7qHEJA6Xcxaiuc0HpVxZYjfJT3BlbkFJqLp67XuziOR9DYyQgfpOGIC9PQ0Ldk5xORmOcnrQ8uYIhQvBrprWA2R4YpOI9MZSr4GvdzHCUAs"
client = OpenAI(api_key=OPENAI_KEY)

app = Flask(__name__)
sock = Sock(app)
cors = CORS(app)
swagger = Swagger(app)

sessions = {}


def transcribe_whisper(audio_recording):
    audio_file = io.BytesIO(audio_recording)
    audio_file.name = 'audio.wav'  # Whisper requires a filename with a valid extension
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        # language = ""  # specify Language explicitly
    )
    print(f"openai transcription: {transcription.text}")
    return transcription.text


# def transcribe_preview(session):
#     if session["audio_buffer"] is not None:
#         text = transcribe_whisper(session["audio_buffer"])
#         # send transcription
#         ws = session.get("websocket")
#         if ws:
#             message = {
#                 "event": "recognizing",
#                 "text": text,
#                 "language": session["language"]
#             }
#             ws.send(json.dumps(message))

@app.route("/chats/<chat_session_id>/sessions", methods=["POST"])
def open_session(chat_session_id):
    """
    Open a new voice input session and start continuous recognition.
    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: ID of the chat session
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - language
          properties:
            language:
              type: string
              description: Language code for speech recognition (e.g., en-US)
    responses:
      200:
        description: Session created successfully
        schema:
          type: object
          properties:
            session_id:
              type: string
              description: Unique identifier for the voice recognition session
      400:
        description: Language parameter missing
        schema:
          type: object
          properties:
            error:
              type: string
              description: Description of the error
    """
    session_id = str(uuid.uuid4())

    body = request.get_json()
    if "language" not in body:
        return jsonify({"error": "Language not specified"}), 400
    language = body["language"]

    sessions[session_id] = {
        "audio_buffer": None,
        "chatSessionId": chat_session_id,
        "language": language,
        "websocket": None  # will be set when the client connects via WS
    }

    return jsonify({"session_id": session_id})


@app.route("/chats/<chat_session_id>/sessions/<session_id>/wav", methods=["POST"])
def upload_audio_chunk(chat_session_id, session_id):
    """
    Upload an audio chunk (expected 16kb, ~0.5s of WAV data).
    The chunk is appended to the push stream for the session.
    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: ID of the chat session
      - name: session_id
        in: path
        type: string
        required: true
        description: ID of the voice input session
      - name: audio_chunk
        in: body
        required: true
        schema:
          type: string
          format: binary
          description: Raw WAV audio data
    responses:
      200:
        description: Audio chunk received successfully
        schema:
          type: object
          properties:
            status:
              type: string
              description: Status message
      404:
        description: Session not found
        schema:
          type: object
          properties:
            error:
              type: string
              description: Description of the error
    """
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    audio_data = request.get_data()  # raw binary data from the POST body

    if sessions[session_id]["audio_buffer"] is not None:
        sessions[session_id]["audio_buffer"] = sessions[session_id]["audio_buffer"] + audio_data
    else:
        sessions[session_id]["audio_buffer"] = audio_data

    # TODO optionally transcribe real time audio chunks, see transcribe_preview()

    return jsonify({"status": "audio_chunk_received"})


@app.route("/chats/<chat_session_id>/sessions/<session_id>", methods=["DELETE"])
def close_session(chat_session_id, session_id):
    """
    Close the session (stop recognition, close push stream, cleanup).
    
    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The ID of the chat session
      - name: session_id
        in: path
        type: string
        required: true
        description: The ID of the session to close
    responses:
      200:
        description: Session successfully closed
        schema:
          type: object
          properties:
            status:
              type: string
              example: session_closed
      404:
        description: Session not found
        schema:
          type: object
          properties:
            error:
              type: string
              example: Session not found
    """
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    if sessions[session_id]["audio_buffer"] is not None:
        # TODO preprocess audio/text, extract and save speaker identification

        text = transcribe_whisper(sessions[session_id]["audio_buffer"])
        # send transcription
        ws = sessions[session_id].get("websocket")
        if ws:
            message = {
                "event": "recognized",
                "text": text,
                "language": sessions[session_id]["language"]
            }
            ws.send(json.dumps(message))

    # # Remove from session store
    sessions.pop(session_id, None)

    return jsonify({"status": "session_closed"})


@sock.route("/ws/chats/<chat_session_id>/sessions/<session_id>")
def speech_socket(ws, chat_session_id, session_id):
    """
    WebSocket endpoint for clients to receive STT results.

    This WebSocket allows clients to connect and receive speech-to-text (STT) results
    in real time. The connection is maintained until the client disconnects. If the 
    session ID is invalid, an error message is sent, and the connection is closed.

    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The unique identifier for the chat session.
      - name: session_id
        in: path
        type: string
        required: true
        description: The unique identifier for the speech session.
    responses:
      400:
        description: Session not found.
      101:
        description: WebSocket connection established.
    """
    if session_id not in sessions:
        ws.send(json.dumps({"error": "Session not found"}))
        return

    # Store the websocket reference in the session
    sessions[session_id]["websocket"] = ws

    # Keep the socket open to send events
    # Typically we'd read messages from the client in a loop if needed
    while True:
        # If the client closes the socket, an exception is thrown or `ws.receive()` returns None
        msg = ws.receive()
        if msg is None:
            break


@app.route('/chats/<chat_session_id>/set-memories', methods=['POST'])
def set_memories(chat_session_id):
    """
    Set memories for a specific chat session.

    ---
    tags:
      - Memories
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The unique identifier of the chat session.
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            chat_history:
              type: array
              items:
                type: object
                properties:
                  text:
                    type: string
                    description: The chat message text.
              description: List of chat messages in the session.
    responses:
      200:
        description: Memory set successfully.
        schema:
          type: object
          properties:
            success:
              type: string
              example: "1"
      400:
        description: Invalid request data.
    """
    chat_history = request.get_json()

    # TODO preprocess data (chat history & system message)

    print(f"{chat_session_id} extracting memories for conversation a:{chat_history[-1]['text']}")

    return jsonify({"success": "1"})


@app.route('/chats/<chat_session_id>/get-memories', methods=['GET'])
def get_memories(chat_session_id):
    """
    Retrieve stored memories for a specific chat session.
    ---
    tags:
      - Memories
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The unique identifier of the chat session.
    responses:
      200:
        description: Successfully retrieved memories for the chat session.
        schema:
          type: object
          properties:
            memories:
              type: string
              description: The stored memories for the chat session.
      400:
        description: Invalid chat session ID.
      404:
        description: Chat session not found.
    """
    print(f"{chat_session_id}: replacing memories...")

    # TODO load relevant memories from your database. Example return value:
    return jsonify({"memories": "The guest typically orders menu 1 and a glass of sparkling water."})


if __name__ == "__main__":
    # In production, you would use a real WSGI server like gunicorn/uwsgi
    app.run(debug=True, host="0.0.0.0", port=5000)
