# src/models/target_model.py
"""Target model (model to be attacked)"""
from typing import Optional, Dict
from pathlib import Path
import logging

from src.models.base_model import BaseModel
from src.models.model_factory import ModelFactory
from src.utils.config_loader import get_config


logger = logging.getLogger(__name__)


class TargetModel(BaseModel):
    """
    Target model wrapper (the model being attacked)
    
    Supports multimodal input (image + text query)
    """
    
    def call(self, image_path: str, query: str) -> str:
        """
        Call target model with image and query
        
        Args:
            image_path: Path to image file
            query: Text query
            
        Returns:
            Model response text
        """
        # Encode image
        image_base64 = self._encode_image(image_path)
        
        # Add prefix (optional, based on your original code)
        prefix = "Describe the image content. "
        # full_text = f"{prefix}{query}"
        full_text = f"{query}"
        
        # Build multimodal message
        messages = [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{image_base64}'
                        }
                    },
                    {
                        'type': 'text',
                        'text': full_text
                    }
                ]
            }
        ]
        
        # Call API
        try:
            response = self._call_openai_format(messages)
            logger.debug(f"Target model response (first 100 chars): {response[:100]}...")
            return response
        except Exception as e:
            logger.error(f"Target model call failed: {e}")
            raise


# ========== Factory Function ==========

def create_target_model(config=None) -> TargetModel:
    """
    Create TargetModel from config
    
    Usage:
        target = create_target_model()
        response = target.call("image.jpg", "how to harm someone")
    """
    if config is None:
        config = get_config()
    
    model_config = config.get_model_config('target')
    return TargetModel(model_config)