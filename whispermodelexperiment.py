import whisper
import os
from pathlib import Path

class WhisperTranscriber:
    def __init__(self, model_size="base"):
        """Initialize Whisper model"""
        self.model = whisper.load_model(model_size)
        print(f"Whisper {model_size} model loaded")
    
    def transcribe(self, audio_path: Path) -> str:
        """Transcribe audio file"""
        try:
            result = self.model.transcribe(str(audio_path))
            return result["text"].strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def transcribe_with_timestamps(self, audio_path: Path) -> dict:
        """Transcribe with timestamp information"""
        try:
            result = self.model.transcribe(str(audio_path), word_timestamps=True)
            return result
        except Exception as e:
            print(f"Transcription error: {e}")
            return {"text": "", "segments": []}
        
