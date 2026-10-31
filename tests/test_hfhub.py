import sys
import tempfile
import unittest
import base64
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")
from neuralnode.providers import infer_model_type
from neuralnode.providers.hfhub import _download_file, download_repo
from neuralnode.providers.universal_local import (
    UniversalLocalProvider,
    _generation_options,
)
from neuralnode.core import Message, MessageRole


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape
        self.device = "cpu"

    def to(self, _device):
        return self

    def __getitem__(self, _index):
        return self

    def __len__(self):
        return self.shape[1]


class TestHFHubHelpers(unittest.TestCase):
    def test_infer_gguf(self):
        self.assertEqual(infer_model_type(["a.gguf"]), "gguf")

    def test_infer_safetensors(self):
        self.assertEqual(infer_model_type(["a.safetensors"]), "safetensors")

    def test_infer_onnx(self):
        self.assertEqual(infer_model_type(["m.onnx"]), "onnx")

    def test_infer_unknown(self):
        self.assertEqual(infer_model_type(["README.md"]), "unknown")

    @patch("neuralnode.providers.hfhub.repo_info", return_value={"private": False})
    @patch(
        "neuralnode.providers.hfhub.list_repo_files",
        return_value=[
            "config.json",
            "tokenizer.json",
            "shards/model-00001.safetensors",
            "shards/model-00002.safetensors",
        ],
    )
    @patch("neuralnode.providers.hfhub._download_file")
    def test_download_sharded_safetensors_returns_model_directory(
        self, download_file, _list_files, _repo_info
    ):
        def write_file(_repo_id, path, dest, _progress_callback=None):
            output = dest.joinpath(*Path(path).parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("fixture")
            return output

        download_file.side_effect = write_file
        with tempfile.TemporaryDirectory() as cache:
            model_dir = download_repo("org/model", cache_dir=cache)

            self.assertEqual(model_dir, Path(cache) / "org__model")
            self.assertTrue((model_dir / "config.json").is_file())
            self.assertTrue((model_dir / "shards/model-00001.safetensors").is_file())
            self.assertTrue((model_dir / "shards/model-00002.safetensors").is_file())

    @patch("neuralnode.providers.hfhub.repo_info", return_value={"private": False})
    @patch(
        "neuralnode.providers.hfhub.list_repo_files",
        return_value=["model.safetensors", "config.json", "preprocessor_config.json"],
    )
    @patch("neuralnode.providers.hfhub._download_file")
    def test_download_repo_includes_vision_processor_config(
        self, download_file, _list_files, _repo_info
    ):
        def write_file(_repo_id, path, dest, _progress_callback=None):
            output = dest.joinpath(*Path(path).parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("fixture")
            return output

        download_file.side_effect = write_file
        with tempfile.TemporaryDirectory() as cache:
            model_dir = download_repo("org/vision", cache_dir=cache)

        self.assertIn(
            "preprocessor_config.json",
            [call.args[1] for call in download_file.call_args_list],
        )

    @patch("neuralnode.providers.hfhub.repo_info", return_value={"private": False})
    @patch("neuralnode.providers.hfhub.list_repo_files", return_value=["model.gguf"])
    @patch("neuralnode.providers.hfhub._download_file")
    def test_download_gguf_returns_model_file(self, download_file, _list_files, _repo_info):
        with tempfile.TemporaryDirectory() as cache:
            model_file = download_repo("org/model", cache_dir=cache)

        self.assertEqual(model_file.name, "model.gguf")
        download_file.assert_called_once()

    def test_download_file_rejects_paths_outside_cache(self):
        with tempfile.TemporaryDirectory() as cache:
            with self.assertRaises(ValueError):
                _download_file("org/model", "../outside.bin", Path(cache))
            with self.assertRaises(ValueError):
                _download_file("org/model", r"..\outside.bin", Path(cache))

    @patch("neuralnode.providers.hfhub.requests.get")
    def test_download_file_uses_huggingface_resolve_route(self, get):
        response = MagicMock()
        response.headers = {"Content-Length": "7"}
        response.iter_content.return_value = [b"fixture"]
        response.__enter__.return_value = response
        get.return_value = response

        with tempfile.TemporaryDirectory() as cache:
            downloaded = _download_file("org/model", "weights/model file.bin", Path(cache))
            self.assertEqual(downloaded.read_bytes(), b"fixture")

        get.assert_called_once_with(
            "https://huggingface.co/org/model/resolve/main/weights/model%20file.bin",
            stream=True,
            timeout=60,
        )

    def test_universal_local_provider_is_instantiable(self):
        with tempfile.TemporaryDirectory() as model_dir:
            provider = UniversalLocalProvider(model_path=model_dir, device="cpu")

        self.assertEqual(provider.provider_name, "universal-local")

    def test_greedy_generation_does_not_pass_zero_temperature(self):
        options = _generation_options(0, 8, 0, 2)

        self.assertFalse(options["do_sample"])
        self.assertNotIn("temperature", options)
        self.assertEqual(options["max_new_tokens"], 8)

    def test_sampling_generation_passes_temperature(self):
        options = _generation_options(0.4, 8, 0, 2)

        self.assertTrue(options["do_sample"])
        self.assertEqual(options["temperature"], 0.4)

    def test_local_vision_chat_passes_decoded_image_to_processor(self):
        from PIL import Image

        image_buffer = BytesIO()
        Image.new("RGB", (2, 2), color="red").save(image_buffer, format="PNG")
        image_b64 = base64.b64encode(image_buffer.getvalue()).decode("ascii")

        inputs = {"input_ids": _FakeTensor((1, 2))}
        processor = MagicMock()
        processor.image_token = "<image>"
        processor.apply_chat_template.return_value = "describe this image"
        processor.return_value = inputs
        processor.decode.return_value = "a red image"
        model = MagicMock()
        model.config = SimpleNamespace(is_encoder_decoder=False)
        model.parameters.return_value = iter([SimpleNamespace(device="cpu")])
        model.generate.return_value = _FakeTensor((1, 4))
        torch = SimpleNamespace(no_grad=lambda: __import__("contextlib").nullcontext())

        with tempfile.TemporaryDirectory() as model_dir:
            provider = UniversalLocalProvider(model_path=model_dir, device="cpu")
            provider._processor = processor
            provider._model = model
            # The model is mocked so this does not need model weights or a GPU.
            with patch.dict(sys.modules, {"torch": torch}):
                result = provider._chat_vision_transformers(
                    [
                        Message(
                            role=MessageRole.USER,
                            content="What is shown?",
                            metadata={"images": [image_b64]},
                        )
                    ],
                    temperature=0,
                )

        self.assertEqual(result.content, "a red image")
        self.assertEqual(len(processor.call_args.kwargs["images"]), 1)
        self.assertIn("<image>", processor.call_args.kwargs["text"])
        self.assertFalse(model.generate.call_args.kwargs["do_sample"])

    def test_local_vision_provider_rejects_remote_image_urls(self):
        with self.assertRaisesRegex(ValueError, "local image path"):
            UniversalLocalProvider._open_message_images(
                Message(
                    role=MessageRole.USER,
                    content="describe",
                    metadata={"image_url": "https://example.com/a.png"},
                )
            )

    def test_captioning_processor_receives_image_without_chat_prompt(self):
        from PIL import Image

        image_buffer = BytesIO()
        Image.new("RGB", (2, 2), color="blue").save(image_buffer, format="PNG")
        image_b64 = base64.b64encode(image_buffer.getvalue()).decode("ascii")

        class CaptionProcessor:
            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return {"pixel_values": _FakeTensor((1, 3))}

            def decode(self, _generated, **_kwargs):
                return "a blue image"

        processor = CaptionProcessor()
        model = MagicMock()
        model.config = SimpleNamespace(is_encoder_decoder=True)
        model.parameters.return_value = iter([SimpleNamespace(device="cpu")])
        model.generate.return_value = _FakeTensor((1, 4))
        torch = SimpleNamespace(no_grad=lambda: __import__("contextlib").nullcontext())

        with tempfile.TemporaryDirectory() as model_dir:
            provider = UniversalLocalProvider(model_path=model_dir, device="cpu")
            provider._processor = processor
            provider._model = model
            with patch.dict(sys.modules, {"torch": torch}):
                result = provider._chat_vision_transformers(
                    [
                        Message(
                            role=MessageRole.USER,
                            content="Describe it",
                            metadata={"images": [image_b64]},
                        )
                    ],
                    temperature=0,
                )

        self.assertEqual(result.content, "a blue image")
        self.assertEqual(set(processor.kwargs), {"images", "return_tensors"})
        self.assertEqual(len(processor.kwargs["images"]), 1)


if __name__ == "__main__":
    unittest.main()
