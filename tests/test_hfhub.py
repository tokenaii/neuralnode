import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")
from neuralnode.providers import infer_model_type
from neuralnode.providers.hfhub import _download_file, download_repo


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


if __name__ == "__main__":
    unittest.main()
