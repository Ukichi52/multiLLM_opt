# src/utils/config_loader.py
import yaml
import os
from typing import Dict, Any
from pathlib import Path

class Config:
    """
    Global configuration manager
    
    Loads config from YAML and provides easy access
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_yaml()
        self._substitute_env_vars()
    
    def _load_yaml(self) -> Dict:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _substitute_env_vars(self):
        """
        Replace ${ENV_VAR} with actual environment variable values
        
        Example: "${OPENAI_API_KEY}" → "sk-..."
        """
        def _replace(obj):
            if isinstance(obj, dict):
                return {k: _replace(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_replace(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                env_var = obj[2:-1]
                return os.getenv(env_var, obj)  # Fallback to original if not found
            else:
                return obj
        
        self._config = _replace(self._config)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get config value using dot notation
        
        Args:
            key_path: Dot-separated path (e.g., "models.mutator.model_name")
            default: Default value if key not found
        
        Example:
            config.get("models.mutator.device")  # → "cuda"
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_model_config(self, model_type: str) -> Dict:
        """
        Get configuration for a specific model
        
        Args:
            model_type: 'mutator', 'target', 'judge', 'image_analyzer'
        
        Returns:
            Model config dict
        """
        return self.get(f"models.{model_type}", {})
    
    def __getitem__(self, key: str) -> Any:
        """Support dict-like access: config['models']"""
        return self._config.get(key)


# Singleton instance
_global_config = None

def get_config(config_path: str = "config/config.yaml") -> Config:
    """
    Get global config instance (singleton)
    
    Usage:
        from src.utils.config_loader import get_config
        config = get_config()
        device = config.get("models.mutator.device")
    """
    global _global_config
    if _global_config is None:
        _global_config = Config(config_path)
    return _global_config