# src/models/base_model.py
import base64
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from pathlib import Path

from src.models.api_client import UnifiedAPIClient
from src.utils.config_loader import get_config


logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Base class for all models
    
    Provides common utilities:
    - Image encoding
    - API calling with OpenAI format
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize base model
        
        Args:
            config: Model config dict (with api_base, api_key, model_name, etc.)
        """
        if config is None:
            raise ValueError("Model config is required")
        
        # Create API client
        self.client = UnifiedAPIClient(
            api_base=config['api_base'],
            api_key=config['api_key'],
            model_name=config['model_name'],
            rate_limit=config.get('rate_limit', 1.0)
        )
        
        self.generation_config = config.get('generation_config', {})
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string
        
        Args:
            image_path: Path to image file
        
        Returns:
            Base64 encoded string
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        return base64.b64encode(image_data).decode('utf-8')
    
    def _call_openai_format(
        self, 
        messages: List[Dict],
        **kwargs
    ) -> str:
        """
        Call API using OpenAI message format
        
        Args:
            messages: List of message dicts, e.g.:
                [
                    {'role': 'system', 'content': '...'},
                    {'role': 'user', 'content': '...'}
                ]
            **kwargs: Additional generation parameters
        
        Returns:
            Model response text
        """
        # Merge default config with kwargs
        gen_params = {**self.generation_config, **kwargs}
        
        # Build prompt from messages
        # Note: Some APIs need message format, others need single prompt
        # We'll use the API client's internal handling
        
        # For simplicity, extract the last user message as prompt
        # (API client will handle full message structure)
        return self.client.generate(
            messages=messages,  # ← 直接传 messages（保留图片）
            **gen_params
        )
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """
        Convert OpenAI message format to single prompt
        
        This is a fallback for APIs that don't support message format.
        UnifiedAPIClient will handle the actual message structure.
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            # Handle multimodal content
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if item.get('type') == 'text':
                        text_parts.append(item['text'])
                    elif item.get('type') == 'image_url':
                        text_parts.append('[Image]')
                content = ' '.join(text_parts)
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        return '\n\n'.join(prompt_parts)
    
    @abstractmethod
    def call(self, *args, **kwargs):
        """
        Main interface for calling the model
        
        Each subclass implements its own signature
        """
        pass