# src/models/model_factory.py
import logging
from typing import Optional
from pathlib import Path

from src.models.api_client import APIBasedLLM
from src.utils.config_loader import Config, get_config

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory for creating all model instances
    
    Handles:
    - Loading from config
    - Choosing between API and local
    - Fallback mechanism
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
    
    def create_model(
        self, 
        model_type: str,
        enable_fallback: bool = True
    ):
        """
        Create a model instance
        
        Args:
            model_type: 'target', 'judge', 'mutator', or 'analyzer'
            enable_fallback: Whether to use local model as fallback
        
        Returns:
            Model instance (implements generate() method)
        """
        model_config = self.config.get_model_config(model_type)
        
        if not model_config:
            raise ValueError(f"Model config for '{model_type}' not found in config.yaml")
        
        model_kind = model_config.get('type', 'api')
        
        if model_kind == 'api':
            try:
                logger.info(f"Creating API model for {model_type}: {model_config['model_name']}")
                return APIBasedLLM(model_config)
                
            except Exception as e:
                if enable_fallback:
                    logger.warning(f"API model creation failed: {e}")
                    logger.info(f"Falling back to local model for {model_type}")
                    return self._create_local_fallback(model_type)
                else:
                    raise
        
        elif model_kind == 'local':
            return self._create_local_model(model_type, model_config)
        
        else:
            raise ValueError(f"Unknown model type: {model_kind}")
    
    def _create_local_fallback(self, model_type: str):
        """
        Create local model as fallback
        
        Note: Not implemented yet - will be added when needed
        """
        fallback_config = self.config.get(f"local_models.{model_type}_fallback", {})
        
        if not fallback_config:
            raise RuntimeError(f"No fallback config for {model_type}")
        
        logger.warning("Local model fallback not implemented yet")
        raise NotImplementedError(
            "Local model fallback will be implemented when API is unavailable. "
            "For now, please ensure your API is accessible."
        )
    
    def _create_local_model(self, model_type: str, model_config: dict):
        """
        Create local model (for future use)
        """
        raise NotImplementedError("Local model support will be added later")


# ========== Convenience Functions ==========

def create_target_model(config: Optional[Config] = None):
    """Create target model (the model being attacked)"""
    factory = ModelFactory(config)
    return factory.create_model('target')


def create_judge_model(config: Optional[Config] = None):
    """Create judge model (for scoring responses)"""
    factory = ModelFactory(config)
    return factory.create_model('judge')


def create_mutator_model(config: Optional[Config] = None):
    """Create mutator model (for rewriting queries)"""
    factory = ModelFactory(config)
    return factory.create_model('mutator')


def create_analyzer_model(config: Optional[Config] = None):
    """Create analyzer model (for analyzing images)"""
    factory = ModelFactory(config)
    return factory.create_model('analyzer')