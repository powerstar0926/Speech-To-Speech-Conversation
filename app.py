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

# Find an available port
def find_available_port(start_port=5000, max_port=6000):
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError("No available ports found")

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
                        socketio.emit('ai_thinking')
                        
                        # Process the input and get the response
                        was_interrupted, response = process_input(session, user_input, messages, generator, settings.SPEED)
                        
                        # Emit the complete response
                        if response:
                            socketio.emit('ai_message', {'text': response})
                        
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
        port = find_available_port()
        print(f"\nStarting server on port {port}")
        print("Open your browser and navigate to: http://localhost:" + str(port))
        print("Press CTRL+C to stop the server")
        
        # Run the server without debug mode
        socketio.run(app, 
                    host='127.0.0.1',
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