# src/evaluation/metrics.py（修改后的版本）
"""Individual metric calculators"""
import torch
import numpy as np
import logging
from typing import Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger(__name__)


class SemanticSimilarityCalculator:
    """
    Calculate semantic similarity using SentenceBERT
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize calculator
        
        Args:
            model_path: Path to local model (if None, download from HuggingFace)
            device: 'cuda' or 'cpu'
        """
        if model_path and Path(model_path).exists():
            logger.info(f"Loading SentenceBERT from local: {model_path}")
            self.model = SentenceTransformer(model_path, device=device)
        else:
            logger.info("Downloading SentenceBERT from HuggingFace")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        logger.info(f"SentenceBERT loaded on {device}")
    
    def calculate(self, query1: str, query2: str) -> float:
        """
        Calculate cosine similarity between two queries
        
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        # Encode
        emb1 = self.model.encode(query1, convert_to_tensor=True)
        emb2 = self.model.encode(query2, convert_to_tensor=True)
        
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), 
            emb2.unsqueeze(0)
        ).item()
        
        # Normalize to [0, 1]
        normalized_similarity = (similarity + 1) / 2
        
        return float(normalized_similarity)


class PerplexityCalculator:
    """
    Calculate perplexity using GPT-2
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize calculator
        
        Args:
            model_path: Path to local model (if None, download from HuggingFace)
            device: 'cuda' or 'cpu'
        """
        if model_path and Path(model_path).exists():
            logger.info(f"Loading GPT-2 from local: {model_path}")
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
        else:
            logger.info("Downloading GPT-2 from HuggingFace")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        self.model.eval()
        
        # Move to device
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        logger.info(f"GPT-2 loaded on {self.device}")
    
    def calculate(self, text: str) -> float:
        """
        Calculate normalized perplexity
        
        Returns:
            Normalized perplexity score (0-1, lower = better)
        """
        # Tokenize
        encodings = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        if self.device == "cuda":
            encodings = {k: v.cuda() for k, v in encodings.items()}
        
        # Calculate perplexity
        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings['input_ids'])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        # Use sigmoid-like normalization for better separation
        # Good text: PPL 10-30 → score 0.2-0.4
        # Bad text:  PPL 100+ → score 0.8-1.0
        
        # Log scale
        log_ppl = np.log(max(perplexity, 1.0))  # Prevent log(0)
        
        # Normalize: map [log(10), log(100)] to [0, 1]
        min_log_ppl = np.log(10)   # Good text baseline
        max_log_ppl = np.log(500)  # Bad text threshold
        
        normalized_ppl = (log_ppl - min_log_ppl) / (max_log_ppl - min_log_ppl)
        normalized_ppl = np.clip(normalized_ppl, 0.0, 1.0)
        
        return float(normalized_ppl)


class StepPenaltyCalculator:
    """Calculate step penalty (no model needed)"""
    
    @staticmethod
    def calculate(current_step: int, max_steps: int) -> float:
        return current_step / max_steps