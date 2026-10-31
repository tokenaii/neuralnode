"""Configuration system for NeuralNode with .env support."""

import os
from dataclasses import dataclass, field
import pydantic as pd
from typing import Optional, Dict, Any, List
from pathlib import Path
import json


def load_env_file(env_path: Optional[str] = None) -> Dict[str, str]:
    """Load environment variables from .env file."""
    env_vars = {}

    if env_path is None:
        # Look for .env in current directory and parent directories
        current = Path.cwd()
        for _ in range(5):  # Check up to 5 levels up
            env_file = current / ".env"
            if env_file.exists():
                env_path = str(env_file)
                break
            current = current.parent

    if env_path and Path(env_path).exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    env_vars[key.strip()] = value

    return env_vars


def _load_provider_api_key(data):
    """Load API key from environment if not provided."""
    if data.get("api_key") is None and data.get("name"):
        env_key = f"{data['name'].upper()}_API_KEY"
        env_key = os.getenv(env_key)
        if env_key:
            data["api_key"] = env_key
    return data


class ProviderConfig(pd.BaseModel):
    """Configuration for an LLM provider."""

    name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    timeout: int = pd.Field(60, ge=1)
    max_retries: int = pd.Field(3, ge=0)

    def __init__(self, **values):
        super().__init__(**_load_provider_api_key(dict(values)))


@dataclass
class LocalModelConfig:
    """Configuration for local models."""

    model_id: str
    cache_dir: str = field(default_factory=lambda: os.path.expanduser("~/.neuralnode/models"))
    device: str = "auto"  # auto, cpu, cuda, mps
    quantization: Optional[str] = None  # q4, q8, fp16
    max_memory: Optional[int] = None  # Max GPU memory in MB

    def __post_init__(self):
        """Ensure cache directory exists."""
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Main configuration for NeuralNode."""

    # Default provider
    default_provider: str = "auto"

    # Provider configurations
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)

    # Local model settings
    local_models: List[LocalModelConfig] = field(default_factory=list)

    # Agent settings
    max_iterations: int = 10
    auto_confirm: bool = False

    # Memory settings
    memory_enabled: bool = True
    max_memory_messages: int = 100

    # Tool settings
    safe_mode: bool = True
    allowed_paths: List[str] = field(default_factory=lambda: [str(Path.cwd())])
    blocked_commands: List[str] = field(
        default_factory=lambda: ["rm -rf /", "rm -rf /*", "format", "mkfs", "dd if=/dev/zero"]
    )

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    def __post_init__(self):
        """Initialize provider configs from environment."""
        # Load .env file
        env_vars = load_env_file()
        os.environ.update(env_vars)

        # Initialize known providers
        provider_names = ["anthropic", "google", "ollama"]

        for name in provider_names:
            if name not in self.providers:
                self.providers[name] = ProviderConfig(
                    name=name,
                    model=self._get_default_model(name),
                    base_url=os.getenv(f"{name.upper()}_BASE_URL"),
                )

    def _get_default_model(self, provider: str) -> Optional[str]:
        """Get default model for provider."""
        defaults = {
            "anthropic": "claude-3-haiku-20240307",
            "google": "gemini-pro",
            "ollama": "llama3.2",
        }
        return defaults.get(provider)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        # Handle nested provider configs
        providers = {}
        if "providers" in data:
            for name, cfg in data["providers"].items():
                providers[name] = ProviderConfig(name=name, **cfg)

        # Handle local models
        local_models = []
        if "local_models" in data:
            for cfg in data["local_models"]:
                local_models.append(LocalModelConfig(**cfg))

        # Filter out nested configs from main data
        main_data = {k: v for k, v in data.items() if k not in ["providers", "local_models"]}

        return cls(providers=providers, local_models=local_models, **main_data)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load configuration from JSON or YAML file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            if path.suffix in [".yaml", ".yml"]:
                try:
                    import yaml

                    data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML config files")
            else:
                data = json.load(f)

        return cls.from_dict(data)

    def get_provider_config(self, name: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider."""
        return self.providers.get(name)

    def save_to_file(self, path: str) -> None:
        """Save configuration to file."""
        data = {
            "default_provider": self.default_provider,
            "max_iterations": self.max_iterations,
            "auto_confirm": self.auto_confirm,
            "memory_enabled": self.memory_enabled,
            "max_memory_messages": self.max_memory_messages,
            "safe_mode": self.safe_mode,
            "log_level": self.log_level,
            "providers": {
                name: {
                    "api_key": "***" if cfg.api_key else None,  # Mask API keys
                    "base_url": cfg.base_url,
                    "model": cfg.model,
                    "timeout": cfg.timeout,
                    "max_retries": cfg.max_retries,
                }
                for name, cfg in self.providers.items()
            },
            "local_models": [
                {
                    "model_id": cfg.model_id,
                    "cache_dir": cfg.cache_dir,
                    "device": cfg.device,
                    "quantization": cfg.quantization,
                }
                for cfg in self.local_models
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


# Global config instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _global_config
    _global_config = config


def configure_from_file(path: str) -> Config:
    """Configure from a file and set as global."""
    config = Config.from_file(path)
    set_config(config)
    return config


from pydantic import BaseModel as _PYD_BaseModel
import pydantic as pd
