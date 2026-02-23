# src/optimization/adaptive_chain_selector.py
"""
Adaptive Chain Selector: Uses StrategyAnalytics to dynamically select chains

Phase 4 enhancement over the rule-based ChainSelector
"""
import random
import logging
from typing import Dict, List, Optional
from collections import defaultdict

from src.storage.strategy_pool import StrategyPool
from src.optimization.chain_selector import ChainSelector

logger = logging.getLogger(__name__)


class AdaptiveChainSelector(ChainSelector):
    """
    Adaptive Chain Selector with exploration strategy
    
    Enhancements over base ChainSelector:
    1. Uses historical performance data (StrategyAnalytics)
    2. Exploration strategy to avoid repetition
    3. Context-aware chain selection
    """
    
    def __init__(
        self,
        strategy_pool: StrategyPool,
        predefined_chains: Optional[Dict[str, List[str]]] = None,
        analytics=None,
        exploration_rate: float = 0.2
    ):
        """
        Initialize Adaptive Chain Selector
        
        Args:
            strategy_pool: StrategyPool instance
            predefined_chains: Dict of predefined chains
            analytics: StrategyAnalytics instance (None = no analytics)
            exploration_rate: Probability of exploring (vs exploiting best chain)
        """
        super().__init__(strategy_pool, predefined_chains, analytics)
        
        self.exploration_rate = exploration_rate
        self.selection_history = defaultdict(int)  # Track chain usage
        
        logger.info(
            f"AdaptiveChainSelector initialized: "
            f"exploration_rate={exploration_rate}, "
            f"analytics={'enabled' if analytics else 'disabled'}"
        )
    
    def select_chain(
        self,
        query: str,
        context: str,
        query_type: Optional[str] = None,
        image_info: Optional[Dict] = None,
        used_chains: Optional[List[str]] = None
    ) -> Dict:
        """
        Select chain with exploration strategy
        
        Args:
            query: Query text
            context: Image context
            query_type: Optional query type
            image_info: Image analysis info
            used_chains: Chains already tried (to avoid repetition)
        
        Returns:
            Chain selection dict
        """
        # Method 1: Explicit query_type (highest priority)
        if query_type:
            return self._select_by_type(query_type)
        
        # Method 2: Analytics-based selection (if available)
        if self.analytics:
            analytics_result = self._select_by_analytics(
                query, context, used_chains
            )
            
            # Decide: exploit (use analytics) or explore (random)
            if random.random() > self.exploration_rate:
                # Exploit: use analytics recommendation
                if analytics_result['confidence'] > 0.4:
                    logger.info(f"🎯 Exploiting: {analytics_result['chain_name']} (confidence={analytics_result['confidence']:.2f})")
                    self.selection_history[analytics_result['chain_name']] += 1
                    return analytics_result
        
        # Method 3: Explore - use rule-based selection or random
        exploration_result = self._explore_chains(query, context, used_chains)
        logger.info(f"🔍 Exploring: {exploration_result['chain_name']}")
        self.selection_history[exploration_result['chain_name']] += 1
        return exploration_result
    
    def _select_by_analytics(
        self,
        query: str,
        context: str,
        used_chains: Optional[List[str]] = None
    ) -> Dict:
        """
        Select chain based on historical analytics
        
        Strategy:
        1. Get top-performing chains
        2. Filter out already-used chains
        3. Return best available
        """
        # Get top chains by success rate
        top_chains = self.analytics.get_top_chains(n=5)
        
        if not top_chains:
            # No analytics data yet, fallback to rules
            return self._select_by_keywords(query)
        
        # Filter out used chains
        if used_chains:
            top_chains = [
                (name, rate) for name, rate in top_chains
                if name not in used_chains
            ]
        
        if not top_chains:
            # All top chains used, try random
            return self._explore_chains(query, context, used_chains)
        
        # Select best available chain
        best_chain_name, success_rate = top_chains[0]
        
        if best_chain_name not in self.predefined_chains:
            logger.warning(f"Analytics recommended unknown chain: {best_chain_name}")
            return self._select_by_keywords(query)
        
        return {
            'chain_name': best_chain_name,
            'chain': self.predefined_chains[best_chain_name],
            'reason': f'Analytics (historical success rate: {success_rate:.1%})',
            'confidence': success_rate,
            'method': 'analytics'
        }
    
    def _explore_chains(
        self,
        query: str,
        context: str,
        used_chains: Optional[List[str]] = None
    ) -> Dict:
        """
        Exploration: try different chains
        
        Strategy:
        1. Get all available chains
        2. Filter out used chains
        3. Use rule-based selection or random
        """
        # Get available chains
        available_chains = list(self.predefined_chains.keys())
        
        # Filter out used
        if used_chains:
            available_chains = [c for c in available_chains if c not in used_chains]
        
        if not available_chains:
            # All chains used, allow repetition
            available_chains = list(self.predefined_chains.keys())
        
        # Try rule-based selection first
        rule_result = self._select_by_keywords(query)
        
        if rule_result['chain_name'] and rule_result['chain_name'] in available_chains:
            return {
                **rule_result,
                'method': 'exploration_rule'
            }
        
        # Random selection
        random_chain = random.choice(available_chains)
        
        return {
            'chain_name': random_chain,
            'chain': self.predefined_chains[random_chain],
            'reason': 'Random exploration (diversification)',
            'confidence': 0.3,
            'method': 'exploration_random'
        }
    
    def get_selection_stats(self) -> Dict:
        """Get statistics on chain selection"""
        total = sum(self.selection_history.values())
        
        return {
            'total_selections': total,
            'chain_usage': dict(self.selection_history),
            'chain_distribution': {
                name: count / total if total > 0 else 0.0
                for name, count in self.selection_history.items()
            }
        }


# ========== Factory Function ==========

def create_adaptive_chain_selector(
    strategy_pool = None,
    analytics = None,
    exploration_rate: float = 0.2,
    config = None
) -> AdaptiveChainSelector:
    """
    Create Adaptive Chain Selector
    
    Usage:
        # Without analytics (first run)
        selector = create_adaptive_chain_selector()
        
        # With analytics (after collecting data)
        from src.analysis.strategy_analytics import StrategyAnalytics
        analytics = StrategyAnalytics()
        analytics.load_trajectories()
        selector = create_adaptive_chain_selector(analytics=analytics)
    """
    if strategy_pool is None:
        from src.storage.strategy_pool import StrategyPool
        strategy_pool = StrategyPool()
    
    if config is None:
        from src.utils.config_loader import get_config
        config = get_config()
    
    # Get exploration rate from config
    exploration_config = config.get('exploration', {})
    exploration_rate = exploration_config.get('epsilon', exploration_rate)
    
    return AdaptiveChainSelector(
        strategy_pool=strategy_pool,
        analytics=analytics,
        exploration_rate=exploration_rate
    )