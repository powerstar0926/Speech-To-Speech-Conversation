from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import threading
import queue
import json
from speech_to_speech import process_input, main
from src.utils.config import settings
from src.utils import VoiceGenerator
import requests
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from src.utils.speech import init_vad_pipeline, record_continuous_audio, detect_speech_segments, transcribe_audio
import socket
import sys
import atexit
import time
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# Configure SocketIO with specific settings
socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   async_mode='threading',
                   logger=True,
                   engineio_logger=True)

# Global variables for the conversation
messages = []
audio_queue = queue.Queue()
is_recording = False
recording_thread = None
stop_event = threading.Event()

def cleanup():
    """Cleanup function to be called on exit"""
    global is_recording
    is_recording = False
    stop_event.set()
    if recording_thread:
        recording_thread.join()

# Register cleanup function
atexit.register(cleanup)

def initialize_models():
    """Initialize all required models and components."""
    session = requests.Session()
    generator = VoiceGenerator(settings.MODELS_DIR, settings.VOICES_DIR)
    whisper_processor = WhisperProcessor.from_pretrained(settings.WHISPER_MODEL)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(settings.WHISPER_MODEL)
    vad_pipeline = init_vad_pipeline(settings.HUGGINGFACE_TOKEN)
    generator.initialize(settings.TTS_MODEL, settings.VOICE_NAME)
    
    return session, generator, whisper_processor, whisper_model, vad_pipeline

# Initialize models
session, generator, whisper_processor, whisper_model, vad_pipeline = initialize_models()

def recording_worker():
    """Worker thread for handling audio recording and processing."""
    global is_recording
    while is_recording and not stop_event.is_set():
        try:
            audio_data = record_continuous_audio()
            if audio_data is not None:
                speech_segments = detect_speech_segments(vad_pipeline, audio_data)
                if speech_segments is not None:
                    user_input = transcribe_audio(whisper_processor, whisper_model, speech_segments)
                    if user_input.strip():
                        socketio.emit('user_message', {'text': user_input})
                        
                        # Create a message div for the AI response
                        message_id = f"ai_message_{int(time.time() * 1000)}"
                        socketio.emit('ai_message_start', {'id': message_id})
                        
                        def text_callback(text):
                            socketio.emit('ai_message_chunk', {'id': message_id, 'text': text})
                        
                        # Process the input and get the response
                        was_interrupted, response = process_input(
                            session, 
                            user_input, 
                            messages, 
                            generator, 
                            settings.SPEED,
                            text_callback=text_callback
                        )
                        
                        # Emit the complete message
                        socketio.emit('ai_message_complete', {'id': message_id})
                        
                        if was_interrupted:
                            socketio.emit('interrupted', {'message': 'User interrupted the response'})
        except Exception as e:
            socketio.emit('error', {'message': str(e)})
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global is_recording, recording_thread, stop_event
    if not is_recording:
        is_recording = True
        stop_event.clear()
        recording_thread = threading.Thread(target=recording_worker)
        recording_thread.daemon = True
        recording_thread.start()
        return jsonify({'status': 'recording started'})
    return jsonify({'status': 'already recording'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global is_recording
    is_recording = False
    stop_event.set()
    if recording_thread:
        recording_thread.join(timeout=1.0)
    return jsonify({'status': 'recording stopped'})

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    try:
        # Get port from environment variable or use default
        port = int(os.environ.get('PORT', 8000))
        host = '0.0.0.0'  # Allow external connections
        
        print(f"\nStarting server on {host}:{port}")
        print("The server is now accessible from your local machine")
        print("Press CTRL+C to stop the server")
        
        # Run the server
        socketio.run(app, 
                    host=host,
                    port=port,
                    debug=False,
                    allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        cleanup()
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        cleanup()
        sys.exit(1) 