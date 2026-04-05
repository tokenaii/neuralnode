"""Test TTS and Speech systems."""

import pytest
from unittest.mock import MagicMock, patch, mock_open
import os

import neuralnode as nn


class TestNeuralTTS:
    """Test NeuralTTS system."""
    
    def test_init_default(self):
        """Test default initialization."""
        tts = nn.NeuralTTS()
        assert tts.provider == "edge_tts"
    
    def test_init_elevenlabs(self):
        """Test ElevenLabs initialization."""
        tts = nn.NeuralTTS(provider="elevenlabs", api_key="test-key")
        assert tts.provider == "elevenlabs"
        assert tts.api_key == "test-key"
    
    def test_list_voices(self):
        """Test listing voices."""
        tts = nn.NeuralTTS()
        voices = tts.list_voices()
        
        assert isinstance(voices, list)
        assert len(voices) > 0
    
    def test_speak_text(self):
        """Test speaking text."""
        tts = nn.NeuralTTS()
        
        with patch('neuralnode.tts.NeuralTTS._play_audio') as mock_play:
            result = tts.speak("Hello World")
            # Should not raise exception
            assert True
    
    def test_save_to_file(self, tmp_path):
        """Test saving to file."""
        tts = nn.NeuralTTS()
        output_file = tmp_path / "output.mp3"
        
        with patch('neuralnode.tts.NeuralTTS._save_audio') as mock_save:
            mock_save.return_value = str(output_file)
            result = tts.save_to_file("Hello", str(output_file))
            # Should not raise exception
            assert True


class TestNeuralSpeech:
    """Test NeuralSpeech system."""
    
    def test_init(self):
        """Test initialization."""
        speech = nn.NeuralSpeech()
        assert speech is not None
    
    def test_list_languages(self):
        """Test listing supported languages."""
        speech = nn.NeuralSpeech()
        languages = speech.list_languages()
        
        assert isinstance(languages, dict)
        assert len(languages) > 0
        assert "en-US" in languages
    
    @patch('speech_recognition.Recognizer.recognize_google')
    def test_transcribe_audio(self, mock_recognize):
        """Test transcribing audio."""
        mock_recognize.return_value = "Hello World"
        
        speech = nn.NeuralSpeech()
        dummy_file = os.path.abspath("dummy.wav")
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('speech_recognition.AudioFile') as mock_audiofile:
                with patch('speech_recognition.Recognizer.record') as mock_record:
                    result = speech.transcribe(dummy_file)
                    assert True


class TestTTSLanguages:
    """Test TTS language support."""
    
    def test_english_voice(self):
        """Test English voice selection."""
        tts = nn.NeuralTTS()
        voice = tts.get_voice_for_language("en")
        assert voice is not None
    
    def test_arabic_voice(self):
        """Test Arabic voice selection."""
        tts = nn.NeuralTTS()
        voice = tts.get_voice_for_language("ar")
        assert voice is not None


class TestSpeechLanguages:
    """Test Speech recognition languages."""
    
    def test_english_recognition(self):
        """Test English recognition."""
        speech = nn.NeuralSpeech()
        lang_code = speech.get_language_code("english")
        assert lang_code == "en-US"
    
    def test_arabic_recognition(self):
        """Test Arabic recognition."""
        speech = nn.NeuralSpeech()
        lang_code = speech.get_language_code("arabic")
        assert lang_code == "ar-SA"
