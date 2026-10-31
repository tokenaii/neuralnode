import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, "src")

from neuralnode.providers.image_generation import HuggingFaceImageGenerator, load_hf_image_model


class TestHuggingFaceImageGenerator(unittest.TestCase):
    def test_factory_is_lazy(self):
        generator = load_hf_image_model("org/image-model", device="cpu")

        self.assertIsInstance(generator, HuggingFaceImageGenerator)
        self.assertIsNone(generator._pipeline)

    def test_load_uses_diffusers_pipeline_once(self):
        pipeline = Mock()
        pipeline.to.return_value = pipeline
        auto_pipeline = Mock()
        auto_pipeline.from_pretrained.return_value = pipeline
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(mps=None),
            float32="float32-dtype",
        )
        diffusers = types.SimpleNamespace(AutoPipelineForText2Image=auto_pipeline)
        generator = HuggingFaceImageGenerator("org/image-model", device="cpu")

        with patch.dict(sys.modules, {"torch": torch, "diffusers": diffusers}):
            self.assertIs(generator.load(), pipeline)
            self.assertIs(generator.load(), pipeline)

        auto_pipeline.from_pretrained.assert_called_once_with(
            "org/image-model", torch_dtype="float32-dtype", local_files_only=False
        )
        pipeline.to.assert_called_once_with("cpu")

    def test_generate_saves_returned_image(self):
        image = Mock()
        pipeline = Mock(return_value=types.SimpleNamespace(images=[image]))
        generator = HuggingFaceImageGenerator("org/image-model", device="cpu")
        generator._pipeline = pipeline
        torch_generator = Mock()
        torch_generator.manual_seed.return_value = torch_generator
        torch = types.SimpleNamespace(Generator=Mock(return_value=torch_generator))

        with patch.dict(sys.modules, {"torch": torch}), tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "generated.png"
            result = generator.generate_image(
                "a red cube", output_path=str(output), width=64, height=64, seed=4
            )

        self.assertIs(result, image)
        image.save.assert_called_once_with(output)
        args = pipeline.call_args.kwargs
        self.assertEqual(args["prompt"], "a red cube")
        self.assertEqual((args["width"], args["height"]), (64, 64))
        torch.Generator.assert_called_once_with(device="cpu")
        torch_generator.manual_seed.assert_called_once_with(4)
        self.assertIs(args["generator"], torch_generator)

    def test_rejects_invalid_prompt_dimensions_and_steps(self):
        generator = HuggingFaceImageGenerator("org/image-model")

        with self.assertRaises(ValueError):
            generator.generate_image(" ")
        with self.assertRaises(ValueError):
            generator.generate_image("valid", width=65)
        with self.assertRaises(ValueError):
            generator.generate_image("valid", num_inference_steps=0)


if __name__ == "__main__":
    unittest.main()
