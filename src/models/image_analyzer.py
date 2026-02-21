# src/models/image_analyzer.py
"""Image analyzer to extract scene context"""
import json
import logging
from typing import Dict, List, Tuple
from pathlib import Path

from src.models.base_model import BaseModel
from src.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class ImageAnalyzer(BaseModel):
    """
    Analyze image to extract:
    - Scene type (e.g., "laboratory", "office")
    - Detailed description
    - Key objects
    - Probable contexts (for tree search)
    """
    
    ANALYSIS_PROMPT = """
    Analyze this image and provide:
    1. **Scene Type**: One concise label (e.g., "laboratory", "office", "street", "kitchen")
    2. **Detailed Description**: 2-3 sentences describing the setting, objects, and atmosphere
    3. **Key Objects**: List 5-10 prominent objects visible in the image
    4. **Top 3 Probable Contexts**: Three interpretations with confidence scores (0-1)
    
    Return ONLY valid JSON:
    {
      "scene_type": "laboratory",
      "detailed_description": "A modern chemistry laboratory with...",
      "key_objects": ["beaker", "test tubes", "safety goggles", ...],
      "probable_contexts": [
        {"context": "research_lab", "confidence": 0.7},
        {"context": "educational_lab", "confidence": 0.2},
        {"context": "industrial_facility", "confidence": 0.1}
      ]
    }
    """
    
    def call(self, image_path: str) -> Dict:
        """
        Analyze image and return structured context
        
        Args:
            image_path: Path to image file
        
        Returns:
            {
                'scene_type': str,
                'detailed_description': str,
                'key_objects': List[str],
                'probable_contexts': List[Tuple[str, float]]
            }
        """
        # Encode image
        image_base64 = self._encode_image(image_path)
        
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
                        'text': self.ANALYSIS_PROMPT
                    }
                ]
            }
        ]
        
        # Call API
        try:
            result_raw = self._call_openai_format(messages, temperature=0.3)
            result = self._parse_analysis(result_raw)
            
            logger.info(f"Image analyzed: scene_type='{result['scene_type']}'")
            return result
        
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            # Fallback: return generic context
            return {
                'scene_type': 'unknown',
                'detailed_description': 'Image analysis failed',
                'key_objects': [],
                'probable_contexts': [('unknown', 1.0)]
            }
    
    def _parse_analysis(self, raw_text: str) -> Dict:
        """Parse JSON output from analyzer"""
        import re
        
        try:
            # Try direct parse
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            # Clean markdown
            clean_text = re.sub(r'```json\s*|\s*```', '', raw_text, flags=re.IGNORECASE).strip()
            try:
                data = json.loads(clean_text)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse analyzer JSON: {raw_text[:100]}...")
                raise
        
        # Extract and validate
        result = {
            'scene_type': data.get('scene_type', 'unknown'),
            'detailed_description': data.get('detailed_description', ''),
            'key_objects': data.get('key_objects', []),
            'probable_contexts': []
        }
        
        # Convert probable_contexts to list of tuples
        for ctx in data.get('probable_contexts', []):
            if isinstance(ctx, dict):
                result['probable_contexts'].append((
                    ctx.get('context', 'unknown'),
                    float(ctx.get('confidence', 0.0))
                ))
        
        return result


# ========== Factory Function ==========

def create_image_analyzer(config=None) -> ImageAnalyzer:
    """
    Create ImageAnalyzer from config
    
    Usage:
        analyzer = create_image_analyzer()
        context = analyzer.call("image.jpg")
    """
    if config is None:
        config = get_config()
    
    model_config = config.get_model_config('analyzer')
    return ImageAnalyzer(model_config)