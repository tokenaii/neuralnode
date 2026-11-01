# Changelog

All notable changes to NeuralNode are documented here.

## 2.2.9 — 2026-11-01

### Integrations and runtime cleanup

- Removed the deprecated WhatsApp/Twilio integration and legacy API gateway.
- Removed obsolete provider and deployment artifacts from the maintained core.
- Updated the OpenClaw manager for the current OpenClaw configuration schema,
  loopback-only gateway binding, token authentication, and OpenAI-compatible
  chat-completions enablement.
- Added a blocking-chat fallback for OpenClaw streaming requests so providers
  without native streaming can still complete gateway turns.
- Validated the NeuralNode → OpenClaw → local Qwen/llama.cpp path on AWS.
- Upgraded Ollama chat integration to the native `/api/chat` protocol with
  role-preserving messages, streaming, structured output, and tool calls.
- Added Ollama native reasoning control through `thinking=True` or
  `thinking=False`, preserving the model trace in response metadata.
- Updated Ollama embeddings to the batch `/api/embed` endpoint with a legacy
  compatibility fallback.
- Removed the remaining Twilio optional dependency with the retired WhatsApp
  integration.

## 2.2.8 — 2026-11-01

NeuralNode 2.2.8 is a feature and reliability release focused on local model
execution, multimodal workflows, integrations, and reproducible packaging.

### Multimodal and local model support

- Added local Hugging Face speech adapters:
  - `HuggingFaceSTT` for audio transcription.
  - `HuggingFaceTTS` for text-to-speech generation.
- Added configurable Hugging Face model IDs, lazy loading, language handling,
  CPU/CUDA device selection, and WAV output for the speech adapters.
- Added local vision-language input support through the existing image chat
  interface.
- Added Diffusers-compatible local text-to-image loading and generation.
- Improved local model generation determinism and provider identity.
- Improved model repository download handling for Hugging Face sources.
- Added explicit authentication support for private and gated Hugging Face
  repositories using `HF_TOKEN` or an explicit token parameter.
- Made optional Horus runtime imports tolerant of environments where the heavy
  local ML dependencies are not installed.

### Voice workflows

- Added a sequential `VoiceAgent` pipeline for file-based and push-to-talk
  interactions:

  `audio input → speech-to-text → agent/LLM turn → text-to-speech → audio output`

- Added HTTP speech adapters for connecting external STT and TTS services.
- Documented the distinction between sequential voice turns and low-latency,
  full-duplex live audio. Full-duplex streaming is outside this release.

### Agents and tools

- Improved provider-backed agent tool calling.
- Added support for textual actions from tool-capable models.
- Improved tool-result message handling and buffered streaming compatibility.
- Preserved the existing agents, memory, RAG, embeddings, vector-store, and
  integration APIs.

### Horus and OpenClaw

- Aligned Horus and OpenClaw integrations with the current public interfaces.
- Secured the OpenClaw local API behavior.
- Added test paths that do not require optional OpenClaw HTTP test packages.
### Release engineering

- Added source distribution and wheel build validation in CI.
- Isolated package build output from repository working files.
- Added trusted PyPI publishing through GitHub Actions and PyPI's OIDC
  trusted-publishing flow.
- Added public repository, ownership, license, and release metadata alignment.

### Compatibility and limitations

- Python 3.9, 3.10, 3.11, and 3.12 are supported by the package metadata.
- Heavy local-model, image, and speech features remain optional extras; the
  base installation does not install those large runtime dependencies.
- Public Hugging Face models can be loaded without a token. Private or gated
  repositories require a valid `HF_TOKEN`.
- The voice pipeline in this release processes one turn at a time. It is not a
  real-time, full-duplex audio session.

