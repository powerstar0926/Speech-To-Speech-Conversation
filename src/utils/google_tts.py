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
        Initialize TTS with API key and base URL.
        
        Args:
            api_key (str): OpenAI API key
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
        Generate speech from text using OpenAI TTS.
        
        Args:
            text (str): Text to convert to speech
            voice (str): Voice to use (default: "alloy")
            speed (float): Speed multiplier (default: 1.0)
            
        Returns:
            Tuple[np.ndarray, str]: Audio data as numpy array and phonemes (empty string for OpenAI TTS)
        """
        temp_file_path = None
        try:
            url = f"{self.base_url}/audio/speech"
            data = {
                "model": "tts-1",
                "input": text,
                "voice": voice,
                "response_format": "mp3"
            }
            
            response = requests.post(url, json=data, headers=self.headers)
            print("generated audio:", response)
            response.raise_for_status()
            
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            temp_file_path = temp_file.name
            
            # Write the content and close the file
            temp_file.write(response.content)
            temp_file.close()
            
            # Read the audio file using soundfile
            audio_data, sample_rate = sf.read(temp_file_path)
            
            return audio_data, ""
                
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            return None, ""
        finally:
            # Clean up temporary file if it exists
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file: {str(e)}") 