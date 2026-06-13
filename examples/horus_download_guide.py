"""
Horus public model download and usage guide.

NeuralNode Horus downloads are public-only. If a Hugging Face repository is
private or gated, loading fails instead of using credentials.
"""

import neuralnode as nn


def example_simple_chat_model():
    model = nn.HorusModel("tokenaii/horus").load()
    response = model.chat([{"role": "user", "content": "Hello!"}])
    print(response.content)


def example_custom_cache():
    model = nn.HorusModel(
        "tokenaii/horus",
        cache_dir="D:/Models/Horus",
    ).load()
    print(model.get_model_info())


def example_lens_image():
    model = nn.HorusLensModel("tokenaii/Horus-Lens-1.0").load()
    model.generate_image(
        "A cinematic portrait of an ancient Egyptian AI oracle, detailed, warm light",
        output_path="outputs/horus_lens.png",
        seed=42,
    )
    print("Saved image to outputs/horus_lens.png")


if __name__ == "__main__":
    print(nn.HorusModel.list_available_models())
    example_simple_chat_model()
    example_custom_cache()
    example_lens_image()
