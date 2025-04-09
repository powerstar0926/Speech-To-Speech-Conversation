import requests
import io
import numpy as np
from typing import Optional, Tuple
import soundfile as sf
import tempfile
import os

class GoogleTTS:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        """
        Initialize Google TTS with API key and base URL.
        
        Args:
            api_key (str): Google API key
            base_url (str): Base URL for the API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def generate_speech(self, text: str, voice: str = "alloy", speed: float = 1.0) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Generate speech from text using Google TTS.
        
        Args:
            text (str): Text to convert to speech
            voice (str): Voice to use (default: "alloy")
            speed (float): Speed multiplier (default: 1.0)
            
        Returns:
            Tuple[np.ndarray, str]: Audio data as numpy array and phonemes (empty string for Google TTS)
        """
        try:
            url = f"{self.base_url}/audio/speech"
            data = {
                "model": "tts-1",
                "input": text,
                "voice": voice,
                "response_format": "mp3"
            }
            
            response = requests.post(url, json=data, headers=self.headers)
            response.raise_for_status()
            
            # Convert MP3 to numpy array
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file.write(response.content)
                temp_file.flush()
                
                # Read the audio file using soundfile
                audio_data, sample_rate = sf.read(temp_file.name)
                
                # Clean up temporary file
                os.unlink(temp_file.name)
                
                return audio_data, ""
                
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            return None, "" 