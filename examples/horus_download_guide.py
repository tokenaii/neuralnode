"""Horus model download and usage guide."""

from pathlib import Path

import neuralnode as nn


def example_simple_chat_model():
    model = nn.HorusModel("horus").load()
    response = model.chat([{"role": "user", "content": "Hello!"}])
    print(response.content)


def example_custom_cache():
    model = nn.HorusModel(
        "horus",
        cache_dir=str(Path.home() / ".cache" / "neuralnode" / "models"),
    ).load()
    print(model.get_model_info())


def example_lens_image():
    model = nn.HorusLensModel("Horus-Lens-1.0").load()
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
