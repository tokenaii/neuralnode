import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, "src")

from neuralnode.speech import HTTPSTT
from neuralnode.tts import HTTPTTS
from neuralnode.voice import VoiceAgent


class FakeAgent:
    def run(self, text):
        return {"output": f"Reply to: {text}"}


class FakeSpeech:
    def transcribe(self, path, language=None):
        return f"transcribed {path} ({language})"

    def listen(self, language=None):
        return "microphone text"


class FakeTTS:
    def speak(self, text, output="output.mp3"):
        return output


class TestVoiceAgent(unittest.TestCase):
    def test_audio_file_turn_runs_stt_agent_and_tts(self):
        voice = VoiceAgent(FakeAgent(), FakeSpeech(), FakeTTS())

        turn = voice.process_audio_file("input.wav", language="en-US", output_audio="reply.mp3")

        self.assertEqual(turn.transcript, "transcribed input.wav (en-US)")
        self.assertEqual(turn.response, "Reply to: transcribed input.wav (en-US)")
        self.assertEqual(turn.audio_path, "reply.mp3")

    def test_microphone_turn(self):
        voice = VoiceAgent(FakeAgent(), FakeSpeech(), FakeTTS())

        turn = voice.listen_once()

        self.assertEqual(turn.transcript, "microphone text")
        self.assertIsNone(turn.audio_path)

    @patch("neuralnode.speech.requests.post")
    def test_http_stt_posts_raw_audio(self, post):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "speech.wav"
            audio.write_bytes(b"wav-data")
            response = Mock()
            response.json.return_value = {"text": "recognized"}
            post.return_value = response

            text = HTTPSTT("http://localhost:5010", token="secret").transcribe(str(audio))

        self.assertEqual(text, "recognized")
        self.assertEqual(post.call_args.kwargs["data"], b"wav-data")
        self.assertEqual(post.call_args.kwargs["headers"]["X-Opet-Tts-Token"], "secret")

    @patch("requests.post")
    def test_http_tts_saves_response_audio(self, post):
        response = Mock()
        response.content = b"wav-data"
        post.return_value = response

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reply.wav"
            result = HTTPTTS("http://localhost:5012", voice="sara").speak("hello", str(output))
            self.assertEqual(output.read_bytes(), b"wav-data")

        self.assertEqual(result, str(output.resolve()))
        self.assertEqual(post.call_args.kwargs["json"]["voice"], "sara")


if __name__ == "__main__":
    unittest.main()
