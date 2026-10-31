import unittest
import sys
sys.path.insert(0, "src")
from neuralnode.providers import infer_model_type

class TestHFHubHelpers(unittest.TestCase):
    def test_infer_gguf(self):
        self.assertEqual(infer_model_type(["a.gguf"]), "gguf")
    def test_infer_safetensors(self):
        self.assertEqual(infer_model_type(["a.safetensors"]), "safetensors")
    def test_infer_onnx(self):
        self.assertEqual(infer_model_type(["m.onnx"]), "onnx")
    def test_infer_unknown(self):
        self.assertEqual(infer_model_type(["README.md"]), "unknown")

if __name__ == "__main__":
    unittest.main()
