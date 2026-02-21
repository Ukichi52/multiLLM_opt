# src/optimization/query_mutator.py
from typing import Dict, Optional, Any
from abc import ABC, abstractmethod
import logging
import hashlib
import json
from collections import OrderedDict

from src.storage.strategy_pool import StrategyPool
from src.utils.config_loader import Config

logger = logging.getLogger(__name__)


# ========== LLM Abstraction Layer ==========

class BaseLLM(ABC):
    """
    Abstract base class for Language Models
    
    Design Pattern: Strategy Pattern
    - QueryMutator depends on this abstraction
    - Concrete LLM implementations inherit from this
    - Easy to swap LLM without changing QueryMutator
    """
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text based on prompt
        
        Args:
            prompt: Input prompt
            **kwargs: Model-specific parameters (temperature, max_tokens, etc.)
        
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return model name for logging"""
        pass


class QwenVLModel(BaseLLM):
    """
    Qwen-VL model wrapper
    
    Supports both local inference and API calls
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        load_in_8bit: bool = False
    ):
        """
        Initialize Qwen-VL model
        
        Args:
            model_name: HuggingFace model name or local path
            device: 'cuda' or 'cpu'
            load_in_8bit: Use 8-bit quantization to save memory
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading {model_name} on {device}...")
        
        self.model_name = model_name
        self.device = device
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if device == "cuda" else None
        }
        
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        logger.info(f"Model loaded successfully")
    
    def generate(
        self, 
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Generate text using Qwen-VL
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
        
        Returns:
            Generated text
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Extract only the new part (remove input prompt)
        # Qwen models often echo the input, so we need to strip it
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def get_model_name(self) -> str:
        return self.model_name


class MockLLM(BaseLLM):
    """
    Mock LLM for testing
    
    Returns predictable output without calling real model
    """
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Simple mock: return query + sub_id
        if "Input:" in prompt:
            query = prompt.split("Input:")[-1].strip()
            return f"{query} [MUTATED]"
        return "Mock output"
    
    def get_model_name(self) -> str:
        return "MockLLM"


# ========== QueryMutator ==========

class QueryMutator:
    """
    Query Mutator: Execute query rewriting using Sub-policies
    
    Responsibilities:
    1. Retrieve Sub-policy prompt template
    2. Fill template with context variables
    3. Call LLM to generate mutated query
    4. Cache results to avoid redundant calls
    
    Design Patterns:
    - Strategy Pattern: Depends on BaseLLM abstraction
    - Template Method: Prompt template filling
    """
    
    def __init__(
        self,
        strategy_pool: StrategyPool,
        llm: BaseLLM,
        cache_size: int = 1000,
        enable_cache: bool = True
    ):
        """
        Initialize QueryMutator
        
        Args:
            strategy_pool: StrategyPool instance
            llm: LLM instance (must inherit from BaseLLM)
            cache_size: Maximum number of cached results
            enable_cache: Whether to enable caching
        
        Design decision:
        - Inject LLM dependency (Dependency Injection)
        - Caller decides which LLM to use
        """
        self.pool = strategy_pool
        self.llm = llm
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        
        # LRU Cache: OrderedDict maintains insertion order
        self._cache = OrderedDict()
        
        logger.info(f"QueryMutator initialized with {llm.get_model_name()}")
        logger.info(f"Cache: {'enabled' if enable_cache else 'disabled'} (size: {cache_size})")
    
    def mutate(
        self,
        query: str,
        sub_policy_id: str,
        **context
    ) -> str:
        """
        Mutate query using specified Sub-policy
        
        Args:
            query: Current query to mutate
            sub_policy_id: ID of Sub-policy to apply
            **context: Context variables including:
                - image_context: str (scene type, e.g., "laboratory")
                - image_description: str (detailed description)
                - key_objects: List[str] (optional)
                - trajectory: List[Dict] (optional, for context-aware mutation)
                - Any other variables needed by prompt template
        
        Returns:
            Mutated query string
        
        Raises:
            ValueError: If sub_policy_id not found
        """
        # Step 1: Retrieve Sub-policy
        sub = self.pool.get_sub_policy(sub_policy_id)
        if sub is None:
            raise ValueError(f"Sub-policy '{sub_policy_id}' not found")
        
        # Step 2: Check cache
        if self.enable_cache:
            cache_key = self._compute_cache_key(query, sub_policy_id, context)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for {sub_policy_id}")
                return self._cache[cache_key]
        
        # Step 3: Build prompt
        prompt = self._build_prompt(sub, query, context)
        
        # Step 4: Generate
        logger.debug(f"Calling LLM for {sub_policy_id}")
        try:
            mutated_query = self.llm.generate(prompt)
            
            # Post-process: strip extra whitespace
            mutated_query = mutated_query.strip()
            
            # Validation: ensure output is not empty
            if not mutated_query:
                logger.warning(f"LLM returned empty output for {sub_policy_id}, using original query")
                mutated_query = query
            
        except Exception as e:
            logger.error(f"Error calling LLM for {sub_policy_id}: {e}")
            # Fallback: return original query
            mutated_query = query
        
        # Step 5: Update cache
        if self.enable_cache:
            self._update_cache(cache_key, mutated_query)
        
        return mutated_query
    
    def _build_prompt(self, sub: Dict, query: str, context: Dict) -> str:
        """
        Build complete prompt from template
        
        Args:
            sub: Sub-policy dict (contains prompt_template)
            query: Current query
            context: Context variables
        
        Returns:
            Filled prompt string
        
        Design decision:
        - Use str.format() instead of Jinja2 (simpler for now)
        - Filter out empty context values to keep prompt clean
        - Future: upgrade to Jinja2 for conditional logic
        """
        template = sub['prompt_template']
        
        # Prepare variables
        prompt_vars = {
            'query': query,
            'context': context.get('image_context', ''),
            'image_description': context.get('image_description', ''),
            'key_objects': ', '.join(context.get('key_objects', [])),
        }
        
        # Add any other context variables
        for key, value in context.items():
            if key not in prompt_vars:
                prompt_vars[key] = value
        
        # Fill template
        try:
            prompt = template.format(**prompt_vars)
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            # Fallback: use template as-is
            prompt = template
        
        return prompt
    
    def _compute_cache_key(self, query: str, sub_id: str, context: Dict) -> str:
        """
        Compute cache key for a mutation request
        
        Cache key = hash(query + sub_id + sorted_context)
        
        Design decision:
        - Use hash instead of storing full (query, sub_id, context) tuple
        - Saves memory (store 64-byte hash instead of potentially large strings)
        - Hash collisions are extremely rare in practice
        """
        # Sort context to ensure consistent key
        context_str = json.dumps(context, sort_keys=True)
        
        # Combine all components
        cache_input = f"{query}|{sub_id}|{context_str}"
        
        # Hash
        cache_key = hashlib.sha256(cache_input.encode()).hexdigest()
        
        return cache_key
    
    def _update_cache(self, key: str, value: str):
        """
        Update LRU cache
        
        If cache is full, remove oldest entry
        """
        # If key already exists, move to end (most recent)
        if key in self._cache:
            self._cache.move_to_end(key)
        
        # Add new entry
        self._cache[key] = value
        
        # Evict oldest if over limit
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)  # Remove first (oldest)
    
    def clear_cache(self):
        """Clear all cached results"""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'size': len(self._cache),
            'max_size': self.cache_size,
            'enabled': self.enable_cache
        }


# ========== Factory Functions ==========

def create_qwen_mutator(
    strategy_pool: StrategyPool,
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
    device: str = "cuda",
    **kwargs
) -> QueryMutator:
    """
    Factory function: Create QueryMutator with Qwen-VL
    
    Args:
        strategy_pool: StrategyPool instance
        model_name: Qwen model name or path
        device: 'cuda' or 'cpu'
        **kwargs: Additional arguments for QueryMutator
    
    Returns:
        Configured QueryMutator instance
    
    Usage:
        pool = StrategyPool("config/strategies.yaml")
        mutator = create_qwen_mutator(pool, device="cuda")
    """
    llm = QwenVLModel(model_name=model_name, device=device)
    return QueryMutator(strategy_pool, llm, **kwargs)


def create_mock_mutator(strategy_pool: StrategyPool) -> QueryMutator:
    """
    Factory function: Create QueryMutator with MockLLM (for testing)
    
    Usage:
        pool = StrategyPool("config/strategies.yaml")
        mutator = create_mock_mutator(pool)
    """
    llm = MockLLM()
    return QueryMutator(strategy_pool, llm)

# src/optimization/query_mutator.py (添加新的工厂函数)

def create_mutator_from_config(
    strategy_pool: StrategyPool,
    config: Optional[Config] = None
) -> QueryMutator:
    """
    Create QueryMutator using model from config
    
    Automatically handles API or local model based on config
    
    Usage:
        pool = StrategyPool()
        mutator = create_mutator_from_config(pool)
    """
    from src.models.model_factory import create_mutator_model
    
    llm = create_mutator_model(config)
    return QueryMutator(strategy_pool, llm)