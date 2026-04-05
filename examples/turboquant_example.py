"""TurboQuant example using the current NeuralNode integrations."""

import neuralnode as nn


def transformers_example():
    llm = nn.create_provider(
        "transformers",
        model="Qwen/Qwen2.5-3B-Instruct",
        turboquant=True,
        turboquant_bits=4,
    )

    response = llm.chat([{"role": "user", "content": "Explain TurboQuant briefly."}])
    print(response["content"] if isinstance(response, dict) else response.content)


def horus_example():
    model = nn.HorusModel(
        model_id="tokenaii/horus/Horus-1.0-4B",
        turboquant=True,
        turboquant_bits=4,
        enable_tts=True,
        tts_voice_id="replic-salma-language{ar-eg}",
    ).load()

    result = model.chat_and_speak(
        [{"role": "user", "content": "اشرح TurboQuant في سطرين فقط"}],
        output_file="horus_turboquant_reply.mp3",
    )

    print(result["response"].content)
    print(result["audio_path"])


if __name__ == "__main__":
    transformers_example()
    horus_example()
