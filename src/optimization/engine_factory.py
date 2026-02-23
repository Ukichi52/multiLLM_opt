# src/optimization/engine_factory.py
"""
Factory functions for creating OptimizationEngine with different configurations

Phase 3: Rule-based chain selection
Phase 4: Analytics-driven adaptive selection
"""
import logging
from typing import Optional

from src.optimization.optimization_engine import OptimizationEngine
from src.models.image_analyzer import create_image_analyzer
from src.models.target_model import create_target_model
from src.models.judge_model import create_judge_model
from src.evaluation.evaluator import create_evaluator
from src.optimization.chain_selector import create_chain_selector
from src.optimization.adaptive_chain_selector import create_adaptive_chain_selector
from src.optimization.query_mutator import create_mutator_from_config
from src.optimization.strategy_chain import StrategyChain
from src.storage.strategy_pool import StrategyPool
from src.utils.config_loader import get_config, Config

logger = logging.getLogger(__name__)


def create_baseline_engine(config: Optional[Config] = None) -> OptimizationEngine:
    """
    Create baseline OptimizationEngine (Phase 3 - rule-based)
    
    Usage:
        engine = create_baseline_engine()
        result = engine.optimize_single_query(image, query)
    """
    if config is None:
        config = get_config()
    
    logger.info("Creating BASELINE OptimizationEngine (rule-based chain selection)")
    
    # Models
    image_analyzer = create_image_analyzer(config)
    target_model = create_target_model(config)
    
    # Evaluation
    judge_model = create_judge_model(config, enable_knn=True)
    evaluator = create_evaluator(judge_model, config)
    
    # Strategy
    strategy_pool = StrategyPool()
    chain_selector = create_chain_selector(strategy_pool, config)  # ← Rule-based
    query_mutator = create_mutator_from_config(strategy_pool, config)
    
    # Chain executor
    strategy_chain = StrategyChain(strategy_pool, query_mutator, evaluator)
    
    # Create engine
    engine = OptimizationEngine(
        image_analyzer=image_analyzer,
        target_model=target_model,
        evaluator=evaluator,
        chain_selector=chain_selector,
        strategy_chain=strategy_chain,
        query_mutator=query_mutator,
        config=config
    )
    
    logger.info("Baseline engine created (Phase 3)")
    return engine


def create_adaptive_engine(
    analytics=None,
    exploration_rate: float = 0.2,
    config: Optional[Config] = None
) -> OptimizationEngine:
    """
    Create adaptive OptimizationEngine (Phase 4 - analytics-driven)
    
    Args:
        analytics: StrategyAnalytics instance (None = will use best available)
        exploration_rate: Exploration probability (0.0-1.0)
        config: Configuration
    
    Usage:
        # First, collect data with baseline
        baseline_engine = create_baseline_engine()
        baseline_engine.optimize_batch(dataset, 0, 30)
        
        # Then, analyze and create adaptive engine
        from src.analysis.strategy_analytics import StrategyAnalytics
        analytics = StrategyAnalytics()
        analytics.load_trajectories()
        
        adaptive_engine = create_adaptive_engine(analytics=analytics)
        adaptive_engine.optimize_batch(dataset, 30, 60)
    """
    if config is None:
        config = get_config()
    
    logger.info("Creating ADAPTIVE OptimizationEngine (analytics-driven chain selection)")
    
    # Load analytics if not provided
    if analytics is None:
        logger.info("No analytics provided, attempting to load from trajectories...")
        from src.analysis.strategy_analytics import StrategyAnalytics
        analytics = StrategyAnalytics()
        try:
            analytics.load_trajectories(limit=100)
            logger.info(f"Loaded {len(analytics.trajectories)} trajectories for analytics")
        except Exception as e:
            logger.warning(f"Failed to load analytics: {e}. Falling back to baseline.")
            return create_baseline_engine(config)
    
    # Models (same as baseline)
    image_analyzer = create_image_analyzer(config)
    target_model = create_target_model(config)
    
    # Evaluation (same as baseline)
    judge_model = create_judge_model(config, enable_knn=True)
    evaluator = create_evaluator(judge_model, config)
    
    # Strategy (ADAPTIVE)
    strategy_pool = StrategyPool()
    chain_selector = create_adaptive_chain_selector(  # ← ADAPTIVE!
        strategy_pool=strategy_pool,
        analytics=analytics,
        exploration_rate=exploration_rate,
        config=config
    )
    query_mutator = create_mutator_from_config(strategy_pool, config)
    
    # Chain executor
    strategy_chain = StrategyChain(strategy_pool, query_mutator, evaluator)
    
    # Create engine
    engine = OptimizationEngine(
        image_analyzer=image_analyzer,
        target_model=target_model,
        evaluator=evaluator,
        chain_selector=chain_selector,
        strategy_chain=strategy_chain,
        query_mutator=query_mutator,
        config=config
    )
    
    logger.info(f"Adaptive engine created (Phase 4, exploration_rate={exploration_rate})")
    return engine


def create_optimization_engine(
    mode: str = "adaptive",
    analytics=None,
    config: Optional[Config] = None
) -> OptimizationEngine:
    """
    Unified factory function
    
    Args:
        mode: "baseline" or "adaptive"
        analytics: StrategyAnalytics (for adaptive mode)
        config: Configuration
    
    Returns:
        OptimizationEngine instance
    """
    if mode == "baseline":
        return create_baseline_engine(config)
    elif mode == "adaptive":
        return create_adaptive_engine(analytics, config=config)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'baseline' or 'adaptive'")