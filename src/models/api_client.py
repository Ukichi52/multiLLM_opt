# src/models/api_client.py
import requests
import time
import logging
from typing import Dict, Optional, Any, List
from abc import ABC, abstractmethod
from threading import Lock
import json

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple rate limiter using token bucket algorithm
    
    Ensures we don't exceed API rate limits
    """
    
    def __init__(self, rate: float):
        """
        Args:
            rate: Requests per second (e.g., 1.0 = 1 req/sec, 0.5 = 1 req per 2 sec)
        """
        self.rate = rate
        self.interval = 1.0 / rate if rate > 0 else 0
        self.last_call = 0
        self.lock = Lock()
    
    def wait(self):
        """Wait if necessary to respect rate limit"""
        with self.lock:
            if self.interval > 0:
                elapsed = time.time() - self.last_call
                if elapsed < self.interval:
                    sleep_time = self.interval - elapsed
                    time.sleep(sleep_time)
                self.last_call = time.time()


class UnifiedAPIClient:
    """
    Unified API client for OpenAI-compatible APIs
    
    Supports:
    - Claude (via ai-yyds.com)
    - DeepSeek (via api.deepseek.com)
    - Gemini (via ai-yyds.com)
    - ChatGPT (via ai-yyds.com)
    - Any other OpenAI-compatible API
    
    Key features:
    - Rate limiting
    - Automatic retry on failure
    - Fallback to local model (optional)
    """
    
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model_name: str,
        rate_limit: float = 1.0,
        max_retries: int = 3,
        timeout: int = 60
    ):
        """
        Initialize API client
        
        Args:
            api_base: API base URL (e.g., "https://ai-yyds.com/v1")
            api_key: API key
            model_name: Model name (e.g., "claude-3-7-sonnet-20250219")
            rate_limit: Requests per second
            max_retries: Max retry attempts on failure
            timeout: Request timeout in seconds
        """
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Rate limiter
        self.rate_limiter = RateLimiter(rate_limit)
        
        logger.info(f"Initialized API client: {model_name} @ {api_base}")
    
    def generate(
        self,
        prompt: str = None, 
        messages: List[Dict] = None,  
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text using API
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
        
        Returns:
            Generated text
        
        Raises:
            RuntimeError: If all retries fail
        """
        # Respect rate limit
        self.rate_limiter.wait()
        
        # Build messages
        if messages is None:
            # Simple text-only mode
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        else:
            # Multimodal mode - use provided messages directly
            pass
        
        # Build request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        # Retry loop
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"API call attempt {attempt}/{self.max_retries}")
                
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                
                # Check HTTP status
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                
                # Extract content
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    return content.strip()
                else:
                    raise ValueError(f"Unexpected API response format: {result}")
                
            except requests.exceptions.Timeout:
                last_error = f"Request timeout ({self.timeout}s)"
                logger.warning(f"{last_error}, attempt {attempt}/{self.max_retries}")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP error: {e.response.status_code} - {e.response.text}"
                logger.warning(f"{last_error}, attempt {attempt}/{self.max_retries}")
                
                # Don't retry on 4xx errors (client errors)
                if 400 <= e.response.status_code < 500:
                    break
                
                time.sleep(2 ** attempt)
                
            except requests.exceptions.RequestException as e:
                last_error = f"Request failed: {e}"
                logger.warning(f"{last_error}, attempt {attempt}/{self.max_retries}")
                time.sleep(2 ** attempt)
                
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                last_error = f"Response parsing failed: {e}"
                logger.error(last_error)
                break  # Don't retry on parsing errors
        
        # All retries failed
        error_msg = f"API call failed after {self.max_retries} attempts. Last error: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def get_model_name(self) -> str:
        return self.model_name


# ========== BaseLLM 实现 ==========

class APIBasedLLM:
    """
    BaseLLM implementation using UnifiedAPIClient
    
    This is the adapter that makes our API client compatible with QueryMutator
    """
    
    def __init__(self, config: Dict):
        """
        Initialize from config dict
        
        Args:
            config: Model config from config.yaml, e.g.:
                {
                    'api_base': 'https://ai-yyds.com/v1',
                    'api_key': 'sk-xxx',
                    'model_name': 'claude-3-7-sonnet-20250219',
                    'rate_limit': 1.0,
                    'generation_config': {...}
                }
        """
        self.client = UnifiedAPIClient(
            api_base=config['api_base'],
            api_key=config['api_key'],
            model_name=config['model_name'],
            rate_limit=config.get('rate_limit', 1.0)
        )
        
        self.generation_config = config.get('generation_config', {})
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text (compatible with BaseLLM interface)
        """
        # Merge default config with kwargs
        gen_params = {**self.generation_config, **kwargs}
        
        return self.client.generate(prompt, **gen_params)
    
    def get_model_name(self) -> str:
        return self.client.model_name