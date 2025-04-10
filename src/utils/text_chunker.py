from .config import settings


class TextChunker:
    """A class to handle intelligent text chunking for voice generation."""

    def __init__(self):
        """Initialize the TextChunker with break points and priorities."""
        self.current_text = []
        self.found_first_sentence = False
        self.semantic_breaks = {
            "however": 4,
            "therefore": 4,
            "furthermore": 4,
            "moreover": 4,
            "nevertheless": 4,
            "while": 3,
            "although": 3,
            "unless": 3,
            "since": 3,
            "and": 2,
            "but": 2,
            "because": 2,
            "then": 2,
        }
        self.punctuation_priorities = {
            ".": 5,
            "!": 5,
            "?": 5,
            ";": 4,
            ":": 4,
            ",": 3,
            "-": 2,
        }
        self.abbreviations = {
            "Mr.": "Mr",
            "Mrs.": "Mrs",
            "Dr.": "Dr",
            "Ms.": "Ms",
            "Prof.": "Prof",
            "Sr.": "Sr",
            "Jr.": "Jr",
            "vs.": "vs",
            "etc.": "etc",
            "i.e.": "ie",
            "e.g.": "eg",
            "a.m.": "am",
            "p.m.": "pm",
        }

    def is_complete_sentence(self, text: str) -> bool:
        """Check if the text contains a complete sentence.
        
        Args:
            text (str): The text to check.
            
        Returns:
            bool: True if the text contains a complete sentence, False otherwise.
        """
        # Handle abbreviations
        for abbr, repl in self.abbreviations.items():
            text = text.replace(abbr, repl)

        # Check for sentence-ending punctuation
        if any(text.endswith(p) for p in [".", "!", "?"]):
            # Make sure it's not an abbreviation
            last_word = text.split()[-1].lower()
            if not any(last_word.startswith(abbr.lower()) for abbr in self.abbreviations):
                return True

        return False

    def should_process(self, text: str) -> bool:
        """Determines if text should be processed based on complete sentences.

        Args:
            text (str): The text to check.

        Returns:
            bool: True if the text should be processed, False otherwise.
        """
        # Only process if we have a complete sentence
        return self.is_complete_sentence(text)

    def find_break_point(self, words: list, target_size: int) -> int:
        """Finds optimal break point in text, prioritizing complete sentences.

        Args:
            words (list): The list of words to find a break point in.
            target_size (int): The target size of the chunk.

        Returns:
            int: The index of the break point.
        """
        if len(words) <= target_size:
            return len(words)

        # First, look for sentence-ending punctuation
        for i, word in enumerate(words[: target_size + 3]):
            if any(word.endswith(p) for p in [".", "!", "?"]):
                # Check if it's not an abbreviation
                word_lower = word.lower()
                if not any(word_lower.startswith(abbr.lower()) for abbr in self.abbreviations):
                    return i + 1

        # If no complete sentence found, return the full text
        return len(words)

    def process(self, text: str, audio_queue) -> str:
        """Process text chunk and return remaining text.

        Args:
            text (str): The text to process.
            audio_queue: The audio queue to add sentences to.

        Returns:
            str: The remaining text after processing.
        """
        if not text:
            return ""

        words = text.split()
        if not words:
            return ""

        # Find the last complete sentence
        last_sentence_end = 0
        for i, word in enumerate(words):
            if any(word.endswith(p) for p in [".", "!", "?"]):
                # Check if it's not an abbreviation
                word_lower = word.lower()
                if not any(word_lower.startswith(abbr.lower()) for abbr in self.abbreviations):
                    last_sentence_end = i + 1

        if last_sentence_end > 0:
            # Process the complete sentence
            chunk = " ".join(words[:last_sentence_end]).strip()
            if chunk and any(c.isalnum() for c in chunk):
                chunk = chunk.rstrip(",")
                audio_queue.add_sentences([chunk])
                self.found_first_sentence = True
                return " ".join(words[last_sentence_end:]) if last_sentence_end < len(words) else ""

        return text  # Return the full text if no complete sentence found
