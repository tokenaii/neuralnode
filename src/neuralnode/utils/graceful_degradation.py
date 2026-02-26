"""
Graceful Degradation system for optional dependencies.
Provides fallbacks when optional libraries are not installed.
"""

import sys
from typing import Any, Optional, Callable, Dict, List
from functools import wraps


class OptionalImport:
    """
    Handle optional imports with graceful degradation.
    """
    
    def __init__(self, module_name: str, install_hint: Optional[str] = None):
        self.module_name = module_name
        self.install_hint = install_hint or f"pip install {module_name}"
        self.module = None
        self.available = False
        self._try_import()
    
    def _try_import(self):
        """Try to import the module."""
        try:
            self.module = __import__(self.module_name)
            self.available = True
        except ImportError:
            self.available = False
    
    def __call__(self, default=None):
        """Get module or default."""
        if self.available:
            return self.module
        return default
    
    def require(self, feature_name: str):
        """Require this module or raise informative error."""
        if not self.available:
            raise ImportError(
                f"'{feature_name}' requires {self.module_name}. "
                f"Install with: {self.install_hint}"
            )
        return self.module
    
    def fallback(self, fallback_func: Callable):
        """Decorator to use fallback when module not available."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if self.available:
                    return func(*args, **kwargs)
                else:
                    return fallback_func(*args, **kwargs)
            return wrapper
        return decorator


# Define common optional dependencies
OPTIONAL_DEPS = {
    "faiss": OptionalImport("faiss", "pip install faiss-cpu"),
    "chromadb": OptionalImport("chromadb", "pip install chromadb"),
    "sentence_transformers": OptionalImport("sentence_transformers", "pip install sentence-transformers"),
    "duckduckgo_search": OptionalImport("duckduckgo_search", "pip install duckduckgo-search"),
    "pypdf": OptionalImport("pypdf", "pip install pypdf"),
    "pandas": OptionalImport("pandas", "pip install pandas"),
    "numpy": OptionalImport("numpy", "pip install numpy"),
    "pillow": OptionalImport("PIL", "pip install pillow"),
    "cryptography": OptionalImport("cryptography", "pip install cryptography"),
    "gradio": OptionalImport("gradio", "pip install gradio"),
    "torch": OptionalImport("torch", "pip install torch"),
    "transformers": OptionalImport("transformers", "pip install transformers"),
    "psutil": OptionalImport("psutil", "pip install psutil"),
    "requests": OptionalImport("requests", "pip install requests"),
    "aiohttp": OptionalImport("aiohttp", "pip install aiohttp"),
    "tqdm": OptionalImport("tqdm", "pip install tqdm"),
    "yaml": OptionalImport("yaml", "pip install pyyaml"),
    " RestrictedPython": OptionalImport("RestrictedPython", "pip install RestrictedPython"),
}


def get_optional(module_name: str, default=None):
    """Get optional module or default."""
    if module_name in OPTIONAL_DEPS:
        return OPTIONAL_DEPS[module_name](default)
    
    # Dynamic import
    try:
        return __import__(module_name)
    except ImportError:
        return default


def require_optional(module_name: str, feature_name: str):
    """Require optional module or raise error."""
    if module_name in OPTIONAL_DEPS:
        return OPTIONAL_DEPS[module_name].require(feature_name)
    
    try:
        return __import__(module_name)
    except ImportError:
        raise ImportError(
            f"'{feature_name}' requires {module_name}. "
            f"Please install it first."
        )


def check_dependencies() -> Dict[str, bool]:
    """Check status of all optional dependencies."""
    return {
        name: dep.available
        for name, dep in OPTIONAL_DEPS.items()
    }


def print_dependency_status():
    """Print status of optional dependencies."""
    print("\nOptional Dependencies Status:")
    print("=" * 50)
    
    deps = check_dependencies()
    for name, available in sorted(deps.items()):
        status = "✅ Available" if available else "❌ Missing"
        print(f"  {name:20} {status}")
    
    print("=" * 50)


# Feature availability checks
def has_vector_store() -> bool:
    """Check if vector store features are available."""
    return OPTIONAL_DEPS["faiss"].available or OPTIONAL_DEPS["chromadb"].available


def has_embeddings() -> bool:
    """Check if embedding features are available."""
    return OPTIONAL_DEPS["sentence_transformers"].available


def has_web_search() -> bool:
    """Check if web search features are available."""
    return OPTIONAL_DEPS["duckduckgo_search"].available


def has_pdf_support() -> bool:
    """Check if PDF processing is available."""
    return OPTIONAL_DEPS["pypdf"].available


def has_secure_execution() -> bool:
    """Check if secure code execution is available."""
    return OPTIONAL_DEPS["RestrictedPython"].available


def has_monitoring() -> bool:
    """Check if system monitoring is available."""
    return OPTIONAL_DEPS["psutil"].available


class FeatureAvailability:
    """Check which features are available."""
    
    @staticmethod
    def get_available_features() -> Dict[str, List[str]]:
        """Get list of available features grouped by category."""
        features = {
            "Vector Store": [],
            "Embeddings": [],
            "Web Search": [],
            "Document Processing": [],
            "Security": [],
            "Monitoring": [],
            "Deep Learning": [],
        }
        
        if has_vector_store():
            features["Vector Store"].extend(["FAISS", "ChromaDB"])
        else:
            features["Vector Store"].append("Simple (fallback)")
        
        if has_embeddings():
            features["Embeddings"].append("Sentence Transformers")
        
        if has_web_search():
            features["Web Search"].append("DuckDuckGo")
        
        if has_pdf_support():
            features["Document Processing"].append("PDF")
        
        if has_secure_execution():
            features["Security"].append("Sandboxed Execution")
        
        if has_monitoring():
            features["Monitoring"].append("System Metrics")
        
        if OPTIONAL_DEPS["torch"].available:
            features["Deep Learning"].append("PyTorch")
        
        if OPTIONAL_DEPS["transformers"].available:
            features["Deep Learning"].append("Transformers")
        
        # Remove empty categories
        return {k: v for k, v in features.items() if v}
    
    @staticmethod
    def print_feature_matrix():
        """Print feature availability matrix."""
        features = FeatureAvailability.get_available_features()
        
        print("\n📊 Feature Availability Matrix:")
        print("=" * 60)
        
        for category, items in features.items():
            print(f"\n{category}:")
            for item in items:
                print(f"  ✅ {item}")
        
        # Show missing critical features
        missing = []
        if not has_vector_store():
            missing.append("Vector Store (install: faiss-cpu or chromadb)")
        if not has_embeddings():
            missing.append("Embeddings (install: sentence-transformers)")
        
        if missing:
            print("\n⚠️  Recommended for better performance:")
            for item in missing:
                print(f"  ⬜ {item}")
        
        print("=" * 60)


# Decorators for graceful degradation
def with_fallback(fallback_value: Any):
    """Decorator to return fallback value on failure."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ImportError as e:
                print(f"Warning: {e}")
                return fallback_value
            except Exception as e:
                print(f"Error in {func.__name__}: {e}")
                return fallback_value
        return wrapper
    return decorator


def with_optional_dep(module_name: str, fallback_func: Optional[Callable] = None):
    """Decorator to only run if optional dependency is available."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            dep = OPTIONAL_DEPS.get(module_name) or OptionalImport(module_name)
            
            if not dep.available:
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                else:
                    raise ImportError(
                        f"{func.__name__} requires {module_name}. "
                        f"Install with: {dep.install_hint}"
                    )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    print_dependency_status()
    FeatureAvailability.print_feature_matrix()
