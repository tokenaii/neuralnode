import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, "src")

from neuralnode.speech import HTTPSTT, HuggingFaceSTT
from neuralnode.tts import HTTPTTS, HuggingFaceTTS
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

            text = HTTPSTT("http://localhost:5010", token="secret").transcribe(
                str(audio), language="en"
            )

        self.assertEqual(text, "recognized")
        self.assertEqual(post.call_args.kwargs["data"], b"wav-data")
        self.assertEqual(post.call_args.kwargs["headers"]["X-Opet-Tts-Token"], "secret")
        self.assertEqual(post.call_args.kwargs["params"], {"language": "en"})

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

    def test_huggingface_stt_transcribes_a_local_file(self):
        pipeline = Mock()
        asr = Mock(return_value={"text": "recognized words"})
        pipeline.return_value = asr
        pipeline.return_value.feature_extractor.sampling_rate = 16000
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.pipeline = pipeline
        import numpy as np

        fake_soundfile = types.ModuleType("soundfile")
        fake_soundfile.read = Mock(return_value=(np.array([0.0, 0.2], dtype=np.float32), 16000))
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "speech.wav"
            audio.write_bytes(b"audio")
            with patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "transformers": fake_transformers,
                    "soundfile": fake_soundfile,
                },
            ):
                result = HuggingFaceSTT(model_id="openai/whisper-tiny").transcribe(
                    str(audio), language="en-US"
                )

        self.assertEqual(result, "recognized words")
        pipeline.assert_called_once()
        self.assertEqual(pipeline.call_args.args[0], "automatic-speech-recognition")
        self.assertEqual(asr.call_args.args[0]["sampling_rate"], 16000)
        self.assertEqual(asr.call_args.kwargs["generate_kwargs"], {"language": "english"})

    def test_huggingface_tts_writes_pcm_wave(self):
        import numpy as np

        pipeline = Mock()
        synthesizer = Mock(
            return_value={"audio": np.array([0.0, 0.5, -0.5]), "sampling_rate": 16000}
        )
        pipeline.return_value = synthesizer
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.pipeline = pipeline
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "reply.wav"
            with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
                result = HuggingFaceTTS(model_id="facebook/mms-tts-eng").speak("hello", str(output))
            self.assertEqual(result, str(output.resolve()))
            self.assertTrue(output.exists())
            import wave

            with wave.open(str(output), "rb") as wav_file:
                self.assertEqual(wav_file.getframerate(), 16000)
                self.assertEqual(wav_file.getnframes(), 3)

        self.assertEqual(pipeline.call_args.args[0], "text-to-audio")
        self.assertEqual(synthesizer.call_args.args[0], "hello")


if __name__ == "__main__":
    unittest.main()
