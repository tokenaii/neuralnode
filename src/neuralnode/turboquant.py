"""TurboQuant Integration for NeuralNode.

TurboQuant is an advanced KV Cache compression technology from Google Research
that achieves 6x+ memory reduction with zero accuracy loss.

This module provides seamless integration with NeuralNode's provider system,
allowing you to use TurboQuant compression with any supported LLM.

Requirements:
    pip install turboquant

Example::

    import neuralnode as nn

    # Create TurboQuant cache with 4-bit compression
    cache = nn.TurboQuant(bits=4)

    # Use with any provider that supports KV caching
    ai = nn.NeuralNode(provider="transformers", model="Qwen/Qwen2.5-3B-Instruct")

    # Generate with compressed cache
    response = ai.generate(
        "Explain quantum computing",
        past_key_values=cache,
        max_new_tokens=200
    )

Advanced Usage::

    # Asymmetric quantization (different bits for keys and values)
    cache = nn.TurboQuant(key_bits=4, value_bits=3)

    # Protect specific layers from quantization
    cache = nn.TurboQuant(
        bits=4,
        protected_layers=[0, -1]  # First and last layer stay in FP16
    )

    # Use with vLLM for production serving
    from neuralnode.turboquant import TurboQuantEngine
    engine = TurboQuantEngine(
        model="meta-llama/Llama-3.1-8B-Instruct",
        bits=4
    )

References:
    - Google Research Blog: https://research.google/blog/turboquant
    - Paper: https://arxiv.org/abs/2504.19874
"""

from typing import Optional, List, Union, Dict, Any, Tuple
from dataclasses import dataclass, field
import warnings

# Try to import turboquant, provide fallback if not available
try:
    from turboquant import TurboQuantCache as _TurboQuantCache
    from turboquant import TurboQuantConfig as _TurboQuantConfig
    TURBOQUANT_AVAILABLE = True
except ImportError:
    TURBOQUANT_AVAILABLE = False
    _TurboQuantCache = None
    _TurboQuantConfig = None


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV Cache compression.
    
    Args:
        bits: Number of bits for symmetric quantization (2-8).
            Common values: 4 (recommended), 3 (maximum compression).
        key_bits: Number of bits for key compression (asymmetric mode).
        value_bits: Number of bits for value compression (asymmetric mode).
        protected_layers: Layer indices to keep in FP16 (no compression).
            Use 0 for first layer, -1 for last layer.
        use_polar_quant: Whether to use PolarQuant algorithm (default: True).
        use_qjl: Whether to use QJL for residual correction (default: True).
        
    Examples::
        
        # Symmetric 4-bit quantization (recommended)
        config = TurboQuantConfig(bits=4)
        
        # Asymmetric: 4-bit keys, 3-bit values
        config = TurboQuantConfig(key_bits=4, value_bits=3)
        
        # Protect first and last layers
        config = TurboQuantConfig(bits=4, protected_layers=[0, -1])
    """
    bits: Optional[int] = None
    key_bits: Optional[int] = None
    value_bits: Optional[int] = None
    protected_layers: Optional[List[int]] = None
    use_polar_quant: bool = True
    use_qjl: bool = True
    
    def __post_init__(self):
        if self.bits is not None and (self.bits < 2 or self.bits > 8):
            raise ValueError(f"bits must be between 2 and 8, got {self.bits}")
        if self.key_bits is not None and (self.key_bits < 2 or self.key_bits > 8):
            raise ValueError(f"key_bits must be between 2 and 8, got {self.key_bits}")
        if self.value_bits is not None and (self.value_bits < 2 or self.value_bits > 8):
            raise ValueError(f"value_bits must be between 2 and 8, got {self.value_bits}")


class TurboQuant:
    """TurboQuant KV Cache compression for NeuralNode.
    
    Achieves 6x+ memory reduction on KV Cache with zero accuracy loss.
    Compatible with any model that uses standard transformer attention.
    
    Args:
        bits: Symmetric quantization bits (default: 4).
        key_bits: Asymmetric key bits (overrides bits if set).
        value_bits: Asymmetric value bits (overrides bits if set).
        protected_layers: Layers to exclude from quantization.
        **kwargs: Additional configuration options.
        
    Attributes:
        config: The TurboQuantConfig instance.
        cache: The underlying cache object (if created).
        compression_ratio: Estimated compression ratio achieved.
        
    Examples::
        
        import neuralnode as nn
        
        # Basic usage with 4-bit quantization
        cache = nn.TurboQuant(bits=4)
        
        # Use with Transformers provider
        ai = nn.NeuralNode(
            provider="transformers",
            model="Qwen/Qwen2.5-3B-Instruct"
        )
        
        # The cache is passed during generation
        # Note: Actual usage depends on provider implementation
        
    Note:
        TurboQuant requires the `turboquant` package:
        pip install turboquant
    """
    
    def __init__(
        self,
        bits: int = 4,
        key_bits: Optional[int] = None,
        value_bits: Optional[int] = None,
        protected_layers: Optional[List[int]] = None,
        **kwargs
    ):
        if not TURBOQUANT_AVAILABLE:
            raise ImportError(
                "TurboQuant is not installed. "
                "Install it with: pip install turboquant"
            )
        
        self.config = TurboQuantConfig(
            bits=bits,
            key_bits=key_bits,
            value_bits=value_bits,
            protected_layers=protected_layers,
            **{k: v for k, v in kwargs.items() if k in [
                'use_polar_quant', 'use_qjl'
            ]}
        )
        
        self._cache = None
        self._compression_ratio = 16 / bits  # FP16 to target bits
    
    @property
    def compression_ratio(self) -> float:
        """Estimated memory compression ratio."""
        if self.config.bits:
            return 16 / self.config.bits
        if self.config.key_bits and self.config.value_bits:
            # Average compression for asymmetric
            avg_bits = (self.config.key_bits + self.config.value_bits) / 2
            return 16 / avg_bits
        return 4.0  # Default estimate
    
    def create_cache(self, **kwargs) -> Any:
        """Create the underlying TurboQuant cache object.
        
        Args:
            **kwargs: Additional arguments passed to TurboQuantCache.
            
        Returns:
            The configured TurboQuantCache instance.
            
        Example::
            
            cache = nn.TurboQuant(bits=4)
            turbo_cache = cache.create_cache()
            
            # Use with Hugging Face Transformers
            outputs = model.generate(
                **inputs,
                past_key_values=turbo_cache,
                use_cache=True
            )
        """
        if not TURBOQUANT_AVAILABLE:
            raise ImportError(
                "TurboQuant is not installed. "
                "Install it with: pip install turboquant"
            )
        
        # Build configuration
        config_dict = {}
        
        if self.config.bits is not None:
            config_dict['bits'] = self.config.bits
        else:
            if self.config.key_bits is not None:
                config_dict['key_bits'] = self.config.key_bits
            if self.config.value_bits is not None:
                config_dict['value_bits'] = self.config.value_bits
        
        if self.config.protected_layers is not None:
            config_dict['protected_layers'] = self.config.protected_layers
        
        config_dict.update(kwargs)
        
        self._cache = _TurboQuantCache(**config_dict)
        return self._cache
    
    def get_cache(self) -> Optional[Any]:
        """Get the created cache instance (if any)."""
        return self._cache
    
    def __repr__(self) -> str:
        if self.config.bits:
            return f"TurboQuant(bits={self.config.bits}, ratio={self.compression_ratio:.1f}x)"
        return f"TurboQuant(key_bits={self.config.key_bits}, value_bits={self.config.value_bits})"


class TurboQuantEngine:
    """Production-ready TurboQuant engine for high-performance serving.
    
    This class provides a vLLM-compatible interface for TurboQuant,
    optimized for production deployments.
    
    Args:
        model: Model name or path (Hugging Face format).
        bits: Quantization bits (default: 4).
        key_bits: Asymmetric key bits.
        value_bits: Asymmetric value bits.
        device: Device to run on ("cuda", "cpu", or "auto").
        **kwargs: Additional engine configuration.
        
    Examples::
        
        from neuralnode.turboquant import TurboQuantEngine
        
        # Create engine with 4-bit quantization
        engine = TurboQuantEngine(
            model="meta-llama/Llama-3.1-8B-Instruct",
            bits=4
        )
        
        # Generate text
        response = engine.generate(
            "Explain machine learning",
            max_tokens=200
        )
        
    Note:
        This requires vLLM with TurboQuant support:
        pip install turboquant-vllm
    """
    
    def __init__(
        self,
        model: str,
        bits: int = 4,
        key_bits: Optional[int] = None,
        value_bits: Optional[int] = None,
        device: str = "auto",
        **kwargs
    ):
        self.model_name = model
        self.bits = bits
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.device = device
        self.config = kwargs
        
        self._engine = None
        self._llm = None
        
        # Try to initialize vLLM engine if available
        try:
            self._init_vllm_engine()
        except ImportError:
            warnings.warn(
                "vLLM with TurboQuant not available. "
                "Falling back to transformers backend."
            )
            self._init_transformers_engine()
    
    def _init_vllm_engine(self):
        """Initialize vLLM engine with TurboQuant quantization."""
        try:
            from vllm import LLM
            from vllm.engine.arg_utils import EngineArgs
            
            # Set environment variables for TurboQuant
            import os
            os.environ['TQ4_K_BITS'] = str(self.key_bits or self.bits)
            os.environ['TQ4_V_BITS'] = str(self.value_bits or self.bits)
            
            # Create vLLM instance with TurboQuant
            self._llm = LLM(
                model=self.model_name,
                quantization="turboquant",
                **self.config
            )
            self._backend = "vllm"
        except ImportError as e:
            raise ImportError(
                "vLLM with TurboQuant support not available. "
                "Install with: pip install turboquant-vllm[vllm]"
            ) from e
    
    def _init_transformers_engine(self):
        """Initialize transformers engine with TurboQuant."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device if self.device != "auto" else None
            )
            
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
            
            self._backend = "transformers"
        except ImportError as e:
            raise ImportError(
                "Transformers not available. "
                "Install with: pip install transformers torch"
            ) from e
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate text with TurboQuant compression.
        
        Args:
            prompt: Input text prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated text response.
        """
        if self._backend == "vllm":
            return self._generate_vllm(prompt, max_tokens, temperature, top_p, **kwargs)
        else:
            return self._generate_transformers(prompt, max_tokens, temperature, top_p, **kwargs)
    
    def _generate_vllm(self, prompt, max_tokens, temperature, top_p, **kwargs):
        """Generate using vLLM backend."""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs
        )
        
        outputs = self._llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
    
    def _generate_transformers(self, prompt, max_tokens, temperature, top_p, **kwargs):
        """Generate using Transformers backend."""
        import torch
        
        # Create TurboQuant cache
        cache = TurboQuant(bits=self.bits).create_cache()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                past_key_values=cache,
                use_cache=True,
                **kwargs
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        prompt_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        if response.startswith(prompt_text):
            response = response[len(prompt_text):].strip()
        
        return response
    
    def __repr__(self) -> str:
        return f"TurboQuantEngine(model={self.model_name!r}, bits={self.bits}, backend={self._backend})"


def create_turboquant_cache(
    bits: int = 4,
    key_bits: Optional[int] = None,
    value_bits: Optional[int] = None,
    protected_layers: Optional[List[int]] = None,
    **kwargs
) -> TurboQuant:
    """Factory function to create a TurboQuant cache.
    
    Args:
        bits: Number of bits for quantization (default: 4).
        key_bits: Asymmetric key bits.
        value_bits: Asymmetric value bits.
        protected_layers: Layers to exclude from quantization.
        **kwargs: Additional configuration.
        
    Returns:
        Configured TurboQuant instance.
        
    Example::
        
        from neuralnode.turboquant import create_turboquant_cache
        
        cache = create_turboquant_cache(bits=4)
    """
    return TurboQuant(
        bits=bits,
        key_bits=key_bits,
        value_bits=value_bits,
        protected_layers=protected_layers,
        **kwargs
    )


# Module-level configuration
def is_available() -> bool:
    """Check if TurboQuant is available."""
    return TURBOQUANT_AVAILABLE


def get_version() -> Optional[str]:
    """Get TurboQuant package version if available."""
    if not TURBOQUANT_AVAILABLE:
        return None
    try:
        import turboquant
        return getattr(turboquant, '__version__', 'unknown')
    except:
        return 'unknown'
