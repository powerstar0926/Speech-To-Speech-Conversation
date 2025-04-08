import torch
import numpy as np
from pathlib import Path
from src.models.models import build_model
from src.core.kokoro import generate
from .voice import split_into_sentences


class VoiceGenerator:
    """
    A class to manage voice generation using a pre-trained model.
    """

    def __init__(self, models_dir, voices_dir):
        """
        Initializes the VoiceGenerator with model and voice directories.

        Args:
            models_dir (Path): Path to the directory containing model files.
            voices_dir (Path): Path to the directory containing voice pack files.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.voicepack = None
        self.voice_name = None
        self.models_dir = models_dir
        self.voices_dir = voices_dir
        self._initialized = False

    def initialize(self, model_path, voice_name):
        """
        Initializes the model and voice pack for audio generation.

        Args:
            model_path (str): The filename of the model.
            voice_name (str): The name of the voice pack.

        Returns:
            str: A message indicating the voice has been loaded.

        Raises:
            FileNotFoundError: If the model or voice pack file is not found.
        """
        model_file = self.models_dir / model_path
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_file}. Please place the model file in the 'models' directory."
            )

        self.model = build_model(str(model_file), self.device)
        self.voice_name = voice_name

        voice_path = self.voices_dir / f"{voice_name}.pt"
        if not voice_path.exists():
            raise FileNotFoundError(
                f"Voice pack not found at {voice_path}. Please place voice files in the 'data/voices' directory."
            )

        self.voicepack = torch.load(voice_path, weights_only=True).to(self.device)
        self._initialized = True
        return f"Loaded voice: {voice_name}"

    def list_available_voices(self):
        """
        Lists all available voice packs in the voices directory.

        Returns:
            list: A list of voice pack names (without the .pt extension).
        """
        if not self.voices_dir.exists():
            return []
        return [f.stem for f in self.voices_dir.glob("*.pt")]

    def is_initialized(self):
        """
        Checks if the generator is properly initialized.

        Returns:
            bool: True if the model and voice pack are loaded, False otherwise.
        """
        return (
            self._initialized and self.model is not None and self.voicepack is not None
        )

    def generate(
        self,
        text,
        lang=None,
        speed=1.0,
        pause_duration=4000,
        short_text_limit=200,
        return_chunks=False,
    ):
        """
        Generates speech from the given text.

        Handles both short and long-form text by splitting long text into sentences.

        Args:
            text (str): The text to generate speech from.
            lang (str, optional): The language of the text. Defaults to None.
            speed (float, optional): The speed of speech generation. Defaults to 1.0.
            pause_duration (int, optional): The duration of pause between sentences in milliseconds. Defaults to 4000.
            short_text_limit (int, optional): The character limit for considering text as short. Defaults to 200.
            return_chunks (bool, optional): If True, returns a list of audio chunks instead of concatenated audio. Defaults to False.

        Returns:
            tuple: A tuple containing the generated audio (numpy array or list of numpy arrays) and a list of phonemes.

        Raises:
            RuntimeError: If the model is not initialized.
            ValueError: If there is an error during audio generation.
        """
        if not self.is_initialized():
            raise RuntimeError("Model not initialized. Call initialize() first.")

        if lang is None:
            lang = self.voice_name[0]

        text = text.strip()
        if not text:
            return (None, []) if not return_chunks else ([], [])

        try:
            if len(text) < short_text_limit:
                try:
                    audio, phonemes = generate(
                        self.model, text, self.voicepack, lang=lang, speed=speed
                    )
                    if audio is None or len(audio) == 0:
                        raise ValueError(f"Failed to generate audio for text: {text}")
                    return (
                        (audio, phonemes) if not return_chunks else ([audio], phonemes)
                    )
                except Exception as e:
                    raise ValueError(
                        f"Error generating audio for text: {text}. Error: {str(e)}"
                    )

            sentences = split_into_sentences(text)
            if not sentences:
                return (None, []) if not return_chunks else ([], [])

            audio_segments = []
            phonemes_list = []
            failed_sentences = []

            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue

                try:
                    if audio_segments and not return_chunks:
                        audio_segments.append(np.zeros(pause_duration))

                    audio, phonemes = generate(
                        self.model, sentence, self.voicepack, lang=lang, speed=speed
                    )
                    if audio is not None and len(audio) > 0:
                        audio_segments.append(audio)
                        phonemes_list.extend(phonemes)
                    else:
                        failed_sentences.append(
                            (i, sentence, "Generated audio is empty")
                        )
                except Exception as e:
                    failed_sentences.append((i, sentence, str(e)))
                    continue

            if failed_sentences:
                error_msg = "\n".join(
                    [f"Sentence {i+1}: '{s}' - {e}" for i, s, e in failed_sentences]
                )
                raise ValueError(
                    f"Failed to generate audio for some sentences:\n{error_msg}"
                )

            if not audio_segments:
                return (None, []) if not return_chunks else ([], [])

            if return_chunks:
                return audio_segments, phonemes_list
            return np.concatenate(audio_segments), phonemes_list

        except Exception as e:
            raise ValueError(f"Error in audio generation: {str(e)}")
