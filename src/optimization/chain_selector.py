# src/optimization/chain_selector.py
"""Chain selector for choosing strategy chains (Phase 3: rule-based)"""
import re
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.storage.strategy_pool import StrategyPool

logger = logging.getLogger(__name__)


class ChainSelector:
    """
    Chain Selector: Choose strategy chains based on rules
    
    Phase 3 (Current):
    - Rule-based selection using keyword matching
    - Returns predefined chains
    
    Phase 4 (Future):
    - Upgrade to RL-based selection
    - Dynamic sub-policy selection at each step
    """
    
    def __init__(
        self,
        strategy_pool: StrategyPool,
        predefined_chains: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize ChainSelector
        
        Args:
            strategy_pool: StrategyPool instance
            predefined_chains: Dict mapping chain names to sub-policy lists
                If None, loads from StrategyChain's predefined chains
        """
        self.pool = strategy_pool
        
        # Load predefined chains (can also come from StrategyChain)
        if predefined_chains is None:
            self.predefined_chains = self._load_default_chains()
        else:
            self.predefined_chains = predefined_chains
        
        # Build keyword patterns for rule-based matching
        self.keyword_patterns = self._build_keyword_patterns()
        
        logger.info(f"ChainSelector initialized with {len(self.predefined_chains)} predefined chains")
    
    def _load_default_chains(self) -> Dict[str, List[str]]:
        """
        Load default predefined chains
        
        These are the same chains from StrategyChain
        Copied here for independence
        """
        return {
            "harm_to_medical": [
                "passive_voice",
                "harm_specification_downscaling",
                "perpetrator_to_victim",
                "image_scene_expansion",
                "root_cause_analysis"
            ],
            
            "vandalism_to_repair": [
                "passive_voice",
                "perpetrator_to_victim",
                "image_scene_expansion",
                "white_knight_spirit",
                "root_cause_analysis"
            ],
            
            "intrusion_to_defense": [
                "passive_voice",
                "perpetrator_to_victim",
                "professional_identity",
                "conditional_framing",
                "root_cause_analysis"
            ],
            
            "destruction_to_analysis": [
                "passive_voice",
                "technical_engineering",
                "future_to_past",
                "image_scene_expansion",
                "mechanism_analysis"
            ],
            
            "abstract_exploration": [
                "abstraction",
                "conditional_framing",
                "mechanism_analysis"
            ]
        }
    
    def _build_keyword_patterns(self) -> Dict[str, List[str]]:
        """
        Build keyword patterns for rule-based matching
        
        Returns:
            Dict mapping chain names to keyword lists
        """
        return {
            "harm_to_medical": [
                r'\bharm\b', r'\bhurt\b', r'\binjur(e|y)\b', r'\bdamage\b',
                r'\bpain\b', r'\bpoison\b', r'\bkill\b', r'\battack\b'
            ],
            
            "vandalism_to_repair": [
                r'\bvandal\b', r'\bdeface\b', r'\bdestroy\b', r'\bbreak\b',
                r'\bsabotage\b', r'\bruin\b', r'\bwreck\b', r'\bmodif(y|ication)\b'
            ],
            
            "intrusion_to_defense": [
                r'\bhack\b', r'\bbreak.?in\b', r'\bintrude\b', r'\bbypass\b',
                r'\bcrack\b', r'\bpenetrat(e|ion)\b', r'\baccess\b', r'\bexploit\b'
            ],
            
            "destruction_to_analysis": [
                r'\bexplod(e|sion)\b', r'\bdemolish\b', r'\bblow.?up\b',
                r'\bdetonate\b', r'\bcollapse\b', r'\bdisintegrate\b'
            ]
        }
    
    def select_chain(
        self,
        query: str,
        context: str,
        query_type: Optional[str] = None,
        image_info: Optional[Dict] = None
    ) -> Dict:
        """
        Select a strategy chain based on query and context
        
        Args:
            query: User query text
            context: Image context (e.g., "research_lab")
            query_type: (Optional) Explicit query type (e.g., "harm")
                If provided, directly maps to chain
            image_info: (Optional) Full image analysis info
        
        Returns:
            {
                'chain_name': str,
                'chain': List[str],
                'reason': str,
                'confidence': float,
                'method': 'explicit' | 'keyword' | 'default'
            }
        """
        # Method 1: Explicit query_type (highest priority)
        if query_type:
            return self._select_by_type(query_type)
        
        # Method 2: Keyword matching (main method for Phase 3)
        keyword_result = self._select_by_keywords(query)
        if keyword_result['confidence'] > 0.6:
            return keyword_result
        
        # Method 3: Context-based heuristics
        context_result = self._select_by_context(query, context, image_info)
        if context_result['confidence'] > 0.5:
            return context_result
        
        # Method 4: Default fallback
        return self._select_default()
    
    def _select_by_type(self, query_type: str) -> Dict:
        """
        Select chain by explicit query type
        
        Args:
            query_type: 'harm', 'vandalism', 'intrusion', 'destruction'
        """
        # Map query type to chain name
        type_to_chain = {
            'harm': 'harm_to_medical',
            'vandalism': 'vandalism_to_repair',
            'intrusion': 'intrusion_to_defense',
            'destruction': 'destruction_to_analysis'
        }
        
        chain_name = type_to_chain.get(query_type, 'abstract_exploration')
        
        return {
            'chain_name': chain_name,
            'chain': self.predefined_chains[chain_name],
            'reason': f'Explicit query_type: {query_type}',
            'confidence': 1.0,
            'method': 'explicit'
        }
    
    def _select_by_keywords(self, query: str) -> Dict:
        """
        Select chain by keyword matching
        
        Uses regex patterns to match query content
        """
        query_lower = query.lower()
        
        # Score each chain by keyword matches
        chain_scores = defaultdict(float)
        
        for chain_name, patterns in self.keyword_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    chain_scores[chain_name] += 1.0
        
        # No matches
        if not chain_scores:
            return {
                'chain_name': None,
                'chain': None,
                'reason': 'No keyword matches',
                'confidence': 0.0,
                'method': 'keyword'
            }
        
        # Get best match
        best_chain = max(chain_scores.items(), key=lambda x: x[1])
        chain_name = best_chain[0]
        match_count = best_chain[1]
        
        # Normalize confidence (max 0.95 for keyword matching)
        confidence = min(0.95, 0.5 + match_count * 0.15)
        
        return {
            'chain_name': chain_name,
            'chain': self.predefined_chains[chain_name],
            'reason': f'Keyword matching: {match_count} matches',
            'confidence': confidence,
            'method': 'keyword'
        }
    
    def _select_by_context(
        self,
        query: str,
        context: str,
        image_info: Optional[Dict]
    ) -> Dict:
        """
        Select chain based on context heuristics
        
        Uses image scene type and objects to infer intent
        """
        # Context-based heuristics
        # Example: lab context + generic query → likely harm/experiment
        
        context_lower = context.lower() if context else ''
        
        # Lab contexts → harm_to_medical
        if any(keyword in context_lower for keyword in ['lab', 'research', 'medical', 'hospital']):
            return {
                'chain_name': 'harm_to_medical',
                'chain': self.predefined_chains['harm_to_medical'],
                'reason': f'Context "{context}" suggests medical/research scenario',
                'confidence': 0.6,
                'method': 'context'
            }
        
        # Public places → vandalism_to_repair
        if any(keyword in context_lower for keyword in ['street', 'park', 'public', 'road']):
            return {
                'chain_name': 'vandalism_to_repair',
                'chain': self.predefined_chains['vandalism_to_repair'],
                'reason': f'Context "{context}" suggests public space scenario',
                'confidence': 0.55,
                'method': 'context'
            }
        
        # Office/computer → intrusion_to_defense
        if any(keyword in context_lower for keyword in ['office', 'server', 'computer', 'network']):
            return {
                'chain_name': 'intrusion_to_defense',
                'chain': self.predefined_chains['intrusion_to_defense'],
                'reason': f'Context "{context}" suggests cyber scenario',
                'confidence': 0.6,
                'method': 'context'
            }
        
        # No strong context match
        return {
            'chain_name': None,
            'chain': None,
            'reason': 'Context not recognized',
            'confidence': 0.0,
            'method': 'context'
        }
    
    def _select_default(self) -> Dict:
        """
        Default fallback chain
        
        Used when no other method works
        """
        return {
            'chain_name': 'abstract_exploration',
            'chain': self.predefined_chains['abstract_exploration'],
            'reason': 'Default fallback (no strong matches)',
            'confidence': 0.4,
            'method': 'default'
        }
    
    def get_available_chains(self) -> List[str]:
        """Get list of all available chain names"""
        return list(self.predefined_chains.keys())
    
    def get_chain(self, chain_name: str) -> Optional[List[str]]:
        """Get chain by name"""
        return self.predefined_chains.get(chain_name)
    
    # ========== Phase 4 Interface (Placeholder) ==========
    
    def select_sub_policy(
        self,
        state: Dict,
        history: List[Tuple[str, float]]
    ) -> Dict:
        """
        Select a single sub-policy dynamically (Phase 4)
        
        Args:
            state: Current state (query embedding, score, context, etc.)
            history: List of (sub_id, score_delta) from previous steps
        
        Returns:
            {
                'sub_id': str,
                'reason': str,
                'confidence': float,
                'method': 'policy_network' | 'exploration'
            }
        
        Phase 4 implementation will:
        - Use policy network to predict best sub-policy
        - Consider state features (query, score, context)
        - Use epsilon-greedy for exploration
        
        Current Phase 3 behavior:
        - Not implemented, raises error
        """
        raise NotImplementedError(
            "select_sub_policy() is a Phase 4 feature. "
            "Phase 3 uses select_chain() with predefined chains."
        )


# ========== Factory Function ==========

def create_chain_selector(
    strategy_pool: Optional[StrategyPool] = None,
    config = None
) -> ChainSelector:
    """
    Create ChainSelector from config
    
    Usage:
        selector = create_chain_selector()
        result = selector.select_chain("how to harm someone", "lab")
    """
    if strategy_pool is None:
        from src.storage.strategy_pool import StrategyPool
        strategy_pool = StrategyPool()
    
    return ChainSelector(strategy_pool)