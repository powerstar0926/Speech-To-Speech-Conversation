import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from .google_tts import GoogleTTS
from .voice import split_into_sentences

class VoiceGenerator:
    """
    A class to manage voice generation using OpenAI TTS.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        """
        Initializes the VoiceGenerator with OpenAI TTS.

        Args:
            api_key (str): OpenAI API key
            base_url (str): Base URL for the API
        """
        self.tts = GoogleTTS(api_key, base_url)
        self.voice_name = "alloy"  # Default voice
        self._initialized = True

    def initialize(self, model_path: str, voice_name: str) -> str:
        """
        Sets the voice name for OpenAI TTS.

        Args:
            model_path (str): Not used for OpenAI TTS
            voice_name (str): The name of the voice to use

        Returns:
            str: A message indicating the voice has been set.
        """
        self.voice_name = voice_name
        return f"Using OpenAI TTS voice: {voice_name}"

    def list_available_voices(self) -> List[str]:
        """
        Lists available OpenAI TTS voices.

        Returns:
            list: A list of available voice names.
        """
        return ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    def is_initialized(self) -> bool:
        """
        Checks if the generator is properly initialized.

        Returns:
            bool: Always True for OpenAI TTS
        """
        return self._initialized

    def generate(
        self,
        text: str,
        lang: Optional[str] = None,
        speed: float = 1.0,
        pause_duration: int = 4000,
        short_text_limit: int = 200,
        return_chunks: bool = False,
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Generates speech from the given text using OpenAI TTS.

        Args:
            text (str): The text to generate speech from.
            lang (str, optional): Not used for OpenAI TTS.
            speed (float, optional): The speed of speech generation. Defaults to 1.0.
            pause_duration (int, optional): The duration of pause between sentences in milliseconds. Defaults to 4000.
            short_text_limit (int, optional): The character limit for considering text as short. Defaults to 200.
            return_chunks (bool, optional): If True, returns a list of audio chunks instead of concatenated audio. Defaults to False.

        Returns:
            tuple: A tuple containing the generated audio (numpy array or list of numpy arrays) and an empty list of phonemes.
        """
        if not self.is_initialized():
            raise RuntimeError("OpenAI TTS not initialized")

        text = text.strip()
        if not text:
            return (None, []) if not return_chunks else ([], [])

        try:
            if len(text) < short_text_limit:
                audio, _ = self.tts.generate_speech(text, self.voice_name, speed)
                if audio is None or len(audio) == 0:
                    raise ValueError(f"Failed to generate audio for text: {text}")
                return (audio, []) if not return_chunks else ([audio], [])

            sentences = split_into_sentences(text)
            if not sentences:
                return (None, []) if not return_chunks else ([], [])

            audio_segments = []
            failed_sentences = []

            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue

                try:
                    if audio_segments and not return_chunks:
                        audio_segments.append(np.zeros(pause_duration))

                    audio, _ = self.tts.generate_speech(sentence, self.voice_name, speed)
                    if audio is not None and len(audio) > 0:
                        audio_segments.append(audio)
                    else:
                        failed_sentences.append((i, sentence, "Generated audio is empty"))
                except Exception as e:
                    failed_sentences.append((i, sentence, str(e)))
                    continue

            if failed_sentences:
                error_msg = "\n".join([f"Sentence {i+1}: '{s}' - {e}" for i, s, e in failed_sentences])
                raise ValueError(f"Failed to generate audio for some sentences:\n{error_msg}")

            if not audio_segments:
                return (None, []) if not return_chunks else ([], [])

            if return_chunks:
                return audio_segments, []
            return np.concatenate(audio_segments), []

        except Exception as e:
            raise ValueError(f"Error in audio generation: {str(e)}")
