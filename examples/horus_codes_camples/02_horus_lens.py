"""
Generate an image with Horus Lens.
"""

import neuralnode as nn


model = nn.HorusLensModel("tokenaii/Horus-Lens-1.0").load()

model.generate_image(
    "A detailed cinematic image of an ancient Egyptian AI lab, golden light",
    output_path="outputs/horus_lens.png",
    seed=42,
)

print("Saved image to outputs/horus_lens.png")
