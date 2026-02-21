# src/optimization/optimization_engine.py
"""Optimization engine: main control loop"""
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from src.models.image_analyzer import ImageAnalyzer
from src.models.target_model import TargetModel
from src.evaluation.evaluator import Evaluator
from src.optimization.chain_selector import ChainSelector
from src.optimization.strategy_chain import StrategyChain
from src.optimization.query_mutator import QueryMutator
from src.optimization.trajectory_logger import TrajectoryLogger
from src.utils.data_loader import QueryDataset
from src.utils.config_loader import Config, get_config

logger = logging.getLogger(__name__)


class OptimizationEngine:
    """
    Optimization Engine: Main control loop for query optimization
    
    Coordinates all modules to perform multimodal jailbreak optimization:
    1. Analyze image to get contexts
    2. Try contexts in order of probability (tree search)
    3. For each context, select strategy chain
    4. Execute chain with multiple iterations
    5. Evaluate and log results
    
    Supports:
    - Single query optimization
    - Batch optimization
    - Early stopping
    - Context switching
    - Phase 4 RL integration (via interface)
    """
    
    def __init__(
        self,
        image_analyzer: ImageAnalyzer,
        target_model: TargetModel,
        evaluator: Evaluator,
        chain_selector: ChainSelector,
        strategy_chain: StrategyChain,
        query_mutator: QueryMutator,
        config: Optional[Config] = None
    ):
        """
        Initialize OptimizationEngine
        
        Args:
            All modules are injected (Dependency Injection pattern)
            config: Configuration object
        
        Design decision:
        - Don't create modules internally
        - Caller provides all dependencies
        - Easy to test and mock
        """
        self.image_analyzer = image_analyzer
        self.target_model = target_model
        self.evaluator = evaluator
        self.chain_selector = chain_selector
        self.strategy_chain = strategy_chain
        self.query_mutator = query_mutator
        
        # Load config
        if config is None:
            config = get_config()
        self.config = config
        
        # Optimization parameters
        opt_config = config.get('optimization', {})
        self.max_steps = opt_config.get('max_steps', 8)
        self.early_stop_threshold = opt_config.get('early_stop_threshold', 3)
        self.success_threshold = opt_config.get('success_threshold', 0.8)
        self.context_prob_threshold = opt_config.get('context_prob_threshold', 0.8)
        
        logger.info(f"OptimizationEngine initialized")
        logger.info(f"  Max steps: {self.max_steps}")
        logger.info(f"  Early stop threshold: {self.early_stop_threshold}")
        logger.info(f"  Success threshold: {self.success_threshold}")
    
    def optimize_single_query(
        self,
        image_path: str,
        query: str,
        query_id: Optional[str] = None,
        image_id: Optional[str] = None,
        save_trajectory: bool = True
    ) -> Dict:
        """
        Optimize a single query
        
        Args:
            image_path: Path to image file
            query: Initial query text
            query_id: Unique identifier (auto-generated if None)
            image_id: Image filename (extracted from path if None)
            save_trajectory: Whether to save trajectory to file
        
        Returns:
            {
                'success': bool,
                'converged': bool,
                'final_query': str,
                'final_response': str,
                'final_score': float,
                'total_steps': int,
                'successful_context': str,
                'trajectory': Dict (if save_trajectory=True)
            }
        """
        # Generate IDs if not provided
        if query_id is None:
            query_id = f"query_{int(time.time() * 1000)}"
        
        if image_id is None:
            image_id = Path(image_path).name
        
        logger.info(f"=" * 80)
        logger.info(f"Starting optimization: {query_id}")
        logger.info(f"Image: {image_id}")
        logger.info(f"Query: {query}")
        logger.info(f"=" * 80)
        
        start_time = time.time()
        
        # Initialize trajectory logger
        trajectory_logger = TrajectoryLogger(
            save_dir=self.config.get('logging.trajectory_dir', 'trajectories'),
            enable_top_k_pruning=False  # Phase 3: keep all steps
        )
        
        trajectory_logger.start_query(
            query_id=query_id,
            image_id=image_id,
            initial_query=query,
            image_path=image_path
        )
        
        # Step 1: Analyze image
        logger.info("Step 1: Analyzing image...")
        image_info = self._analyze_image(image_path)
        
        # Step 2: Context tree search
        logger.info("Step 2: Starting context tree search...")
        result = self._context_tree_search(
            image_path=image_path,
            query=query,
            image_info=image_info,
            trajectory_logger=trajectory_logger
        )
        
        # Step 3: Finalize
        elapsed_time = time.time() - start_time
        
        trajectory_logger.finalize(
            success=result['success'],
            final_query=result['final_query'],
            final_response=result['final_response'],
            final_score=result['final_score'],
            converged=result['converged']
        )
        
        logger.info(f"=" * 80)
        logger.info(f"Optimization completed: {query_id}")
        logger.info(f"  Success: {result['success']}")
        logger.info(f"  Final score: {result['final_score']:.3f}")
        logger.info(f"  Total steps: {result['total_steps']}")
        logger.info(f"  Time: {elapsed_time:.1f}s")
        logger.info(f"=" * 80)
        
        # Add trajectory to result
        if save_trajectory:
            result['trajectory'] = trajectory_logger.get_trajectory()
        
        return result
    
    def _analyze_image(self, image_path: str) -> Dict:
        """
        Analyze image to get contexts
        
        Returns:
            ImageInfo dict with probable_contexts
        """
        try:
            image_info = self.image_analyzer.call(image_path)
            
            logger.info(f"Image analysis complete:")
            logger.info(f"  Scene type: {image_info['scene_type']}")
            logger.info(f"  Contexts: {image_info['probable_contexts']}")
            
            return image_info
        
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            # Fallback: return generic context
            return {
                'scene_type': 'unknown',
                'detailed_description': '',
                'key_objects': [],
                'probable_contexts': [('unknown', 1.0)]
            }
    
    def _context_tree_search(
        self,
        image_path: str,
        query: str,
        image_info: Dict,
        trajectory_logger: TrajectoryLogger
    ) -> Dict:
        """
        Try contexts in order of probability (tree search)
        
        Args:
            image_path: Path to image
            query: Initial query
            image_info: Image analysis result
            trajectory_logger: Logger instance
        
        Returns:
            Optimization result
        """
        contexts = image_info['probable_contexts']
        
        # Sort by probability (descending)
        contexts = sorted(contexts, key=lambda x: x[1], reverse=True)
        
        # Select contexts to try (cumulative prob > threshold)
        selected_contexts = self._select_contexts_to_try(contexts)
        
        logger.info(f"Trying {len(selected_contexts)} contexts: {[c[0] for c in selected_contexts]}")
        
        # Try each context
        for i, (context, prob) in enumerate(selected_contexts, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Context {i}/{len(selected_contexts)}: {context} (prob={prob:.2f})")
            logger.info(f"{'=' * 60}")
            
            # Start context in logger
            trajectory_logger.start_context(context, prob)
            
            # Optimize with this context
            result = self._optimize_with_context(
                image_path=image_path,
                query=query,
                context=context,
                image_info=image_info,
                trajectory_logger=trajectory_logger
            )
            
            # Check if successful
            if result['success']:
                logger.info(f"✅ Success with context: {context}")
                result['successful_context'] = context
                return result
            
            # Context failed, try next one
            logger.info(f"⚠️  Context '{context}' failed, trying next...")
            
            if i < len(selected_contexts):
                next_context = selected_contexts[i][0]
                trajectory_logger.log_context_switch(
                    from_context=context,
                    to_context=next_context,
                    reason="Failed to converge",
                    from_score=result['final_score']
                )
        
        # All contexts failed
        logger.warning("❌ All contexts failed")
        
        return {
            'success': False,
            'converged': False,
            'final_query': query,
            'final_response': None,
            'final_score': 0.0,
            'total_steps': result.get('total_steps', 0),
            'successful_context': None
        }
    
    def _select_contexts_to_try(
        self,
        contexts: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Select which contexts to try based on cumulative probability
        
        Args:
            contexts: List of (context_name, probability)
        
        Returns:
            Filtered list of contexts to try
        
        Strategy:
        - Try contexts until cumulative prob > threshold (default 0.8)
        - Always try at least 1, at most 3
        """
        selected = []
        cumulative_prob = 0.0
        
        for context, prob in contexts:
            selected.append((context, prob))
            cumulative_prob += prob
            
            # Stop if cumulative prob exceeds threshold
            if cumulative_prob >= self.context_prob_threshold:
                break
            
            # Stop if reached max contexts
            if len(selected) >= 3:
                break
        
        # Ensure at least 1 context
        if not selected and contexts:
            selected = [contexts[0]]
        
        return selected
    
    def _optimize_with_context(
        self,
        image_path: str,
        query: str,
        context: str,
        image_info: Dict,
        trajectory_logger: TrajectoryLogger
    ) -> Dict:
        """
        Optimize query under a single context
        
        Args:
            image_path: Path to image
            query: Initial query
            context: Context name
            image_info: Image analysis result
            trajectory_logger: Logger instance
        
        Returns:
            {
                'success': bool,
                'converged': bool,
                'final_query': str,
                'final_response': str,
                'final_score': float,
                'total_steps': int
            }
        """
        # Step 1: Select strategy chain
        chain_selection = self.chain_selector.select_chain(
            query=query,
            context=context,
            image_info=image_info
        )
        
        logger.info(
            f"Selected chain: {chain_selection['chain_name']} "
            f"(confidence={chain_selection['confidence']:.2f})"
        )
        
        # Log chain selection
        trajectory_logger.log_chain_start(
            chain_name=chain_selection['chain_name'],
            chain=chain_selection['chain'],
            reason=chain_selection['reason']
        )
        
        # Step 2: Optimization loop
        result = self._optimization_loop(
            image_path=image_path,
            initial_query=query,
            context=context,
            chain=chain_selection['chain'],
            image_info=image_info,
            trajectory_logger=trajectory_logger
        )
        
        return result
    
    def _optimization_loop(
        self,
        image_path: str,
        initial_query: str,
        context: str,
        chain: List[str],
        image_info: Dict,
        trajectory_logger: TrajectoryLogger
    ) -> Dict:
        """
        Core optimization loop
        
        Executes strategy chain iteratively until success or max steps
        
        Args:
            image_path: Path to image
            initial_query: Starting query
            context: Context name
            chain: List of sub-policy IDs
            image_info: Image analysis result
            trajectory_logger: Logger instance
        
        Returns:
            Optimization result
        """
        current_query = initial_query
        previous_score = 0.0
        no_improvement_count = 0
        
        # Build context dict for mutation
        mutation_context = {
            'image_context': context,
            'image_description': image_info['detailed_description'],
            'key_objects': image_info['key_objects']
        }
        
        # Optimization loop
        for step in range(1, self.max_steps + 1):
            logger.info(f"\n--- Step {step}/{self.max_steps} ---")
            
            # Execute strategy chain
            trajectory = self.strategy_chain.execute(
                initial_query=current_query,
                chain=chain,
                context=mutation_context,
                evaluate_each_step=False  # We'll evaluate once after chain
            )
            
            mutated_query = trajectory['final_query']
            
            logger.debug(f"Query before: {current_query}")
            logger.debug(f"Query after:  {mutated_query}")
            
            # Call target model
            try:
                response = self.target_model.call(image_path, mutated_query)
                logger.debug(f"Response: {response[:100]}...")
            except Exception as e:
                logger.error(f"Target model call failed: {e}")
                response = ""
            
            # Evaluate
            try:
                eval_result = self.evaluator.evaluate(
                    response=response,
                    original_query=initial_query,
                    mutated_query=mutated_query,
                    current_step=step,
                    max_steps=self.max_steps
                )
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                eval_result = {
                    'scores': {},
                    'total_score': 0.0,
                    'is_success': False
                }
            
            current_score = eval_result['total_score']
            
            logger.info(
                f"Score: {previous_score:.3f} → {current_score:.3f} "
                f"(Δ {current_score - previous_score:+.3f})"
            )
            
            # Log this step (log each sub in chain)
            chain_steps = trajectory.get('steps', [])
        
            if chain_steps:
                # 如果 StrategyChain 返回了详细步骤，使用它们
                for chain_step in chain_steps:
                    trajectory_logger.log_step(
                        step=step * 100 + chain_step['step'],  # 全局唯一 step ID
                        sub_policy_id=chain_step['sub_id'],
                        sub_policy_name=chain_step['sub_name'],
                        query_before=chain_step['query_before'],
                        query_after=chain_step['query_after'],
                        response=response,  # 最终的 response
                        eval_result=eval_result,  # 最终的评估结果
                        previous_score=previous_score,  # 上一轮的分数
                        metadata={
                            'chain_name': chain[0] if chain else 'unknown',
                            'step_in_chain': chain_step['step'],
                            'total_steps_in_chain': len(chain_steps)
                        }
                    )
            else:
                # 如果没有详细步骤，记录整个 chain 作为一步
                for i, sub_id in enumerate(chain):
                    sub = self.strategy_chain.pool.get_sub_policy(sub_id)
                    
                    trajectory_logger.log_step(
                        step=step * 100 + i,
                        sub_policy_id=sub_id,
                        sub_policy_name=sub['name'],
                        query_before=current_query if i == 0 else mutated_query,
                        query_after=mutated_query,
                        response=response if i == len(chain) - 1 else None,
                        eval_result=eval_result if i == len(chain) - 1 else None,
                        previous_score=previous_score if i == len(chain) - 1 else None,
                        metadata={
                            'chain_position': i,
                            'chain_length': len(chain)
                        }
                    )
            
            # Check success
            if eval_result['is_success'] or current_score >= self.success_threshold:
                logger.info(f"✅ Success at step {step}!")
                return {
                    'success': True,
                    'converged': True,
                    'final_query': mutated_query,
                    'final_response': response,
                    'final_score': current_score,
                    'total_steps': step
                }
            
            # Check improvement
            if current_score <= previous_score:
                no_improvement_count += 1
                logger.debug(f"No improvement: {no_improvement_count}/{self.early_stop_threshold}")
            else:
                no_improvement_count = 0
            
            # Early stopping
            if no_improvement_count >= self.early_stop_threshold:
                logger.info(f"⚠️  Early stop triggered (no improvement for {self.early_stop_threshold} steps)")
                return {
                    'success': False,
                    'converged': False,
                    'final_query': mutated_query,
                    'final_response': response,
                    'final_score': current_score,
                    'total_steps': step
                }
            
            # Update for next iteration
            current_query = mutated_query
            previous_score = current_score
        
        # Max steps reached
        logger.info(f"⚠️  Max steps ({self.max_steps}) reached")
        
        return {
            'success': False,
            'converged': False,
            'final_query': current_query,
            'final_response': response,
            'final_score': current_score,
            'total_steps': self.max_steps
        }
    
    def optimize_batch(
        self,
        dataset: QueryDataset,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        save_trajectories: bool = True
    ) -> List[Dict]:
        """
        Optimize a batch of queries from dataset
        
        Args:
            dataset: QueryDataset instance
            start_idx: Starting index
            end_idx: Ending index (None = all)
            save_trajectories: Whether to save trajectories
        
        Returns:
            List of optimization results
        """
        if end_idx is None:
            end_idx = len(dataset)
        
        logger.info(f"Starting batch optimization: {start_idx} to {end_idx}")
        
        results = []
        
        for idx in range(start_idx, end_idx):
            item = dataset[idx]
            
            logger.info(f"\n{'#' * 80}")
            logger.info(f"Query {idx + 1}/{end_idx}: {item['image_id']}")
            logger.info(f"{'#' * 80}")
            
            try:
                result = self.optimize_single_query(
                    image_path=item['image_path'],
                    query=item['query'],
                    query_id=f"query_{idx:04d}",
                    image_id=item['image_id'],
                    save_trajectory=save_trajectories
                )
                
                results.append(result)
            
            except Exception as e:
                logger.error(f"Failed to optimize query {idx}: {e}")
                results.append({
                    'success': False,
                    'error': str(e)
                })
        
        # Summary
        success_count = sum(1 for r in results if r.get('success', False))
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Batch optimization completed")
        logger.info(f"  Total: {len(results)}")
        logger.info(f"  Success: {success_count} ({success_count/len(results)*100:.1f}%)")
        logger.info(f"{'=' * 80}")
        
        return results


# ========== Factory Function ==========

def create_optimization_engine(config: Optional[Config] = None) -> OptimizationEngine:
    """
    Create OptimizationEngine with all dependencies
    
    Usage:
        engine = create_optimization_engine()
        result = engine.optimize_single_query("image.jpg", "how to harm someone")
    """
    if config is None:
        config = get_config()
    
    # Create all dependencies
    from src.models.image_analyzer import create_image_analyzer
    from src.models.target_model import create_target_model
    from src.models.judge_model import create_judge_model
    from src.evaluation.evaluator import create_evaluator
    from src.optimization.chain_selector import create_chain_selector
    from src.optimization.query_mutator import create_mutator_from_config
    from src.storage.strategy_pool import StrategyPool
    
    logger.info("Creating OptimizationEngine with all dependencies...")
    
    # Models
    image_analyzer = create_image_analyzer(config)
    target_model = create_target_model(config)
    
    # Evaluation
    judge_model = create_judge_model(config, enable_knn=True)
    evaluator = create_evaluator(judge_model, config)
    
    # Strategy
    strategy_pool = StrategyPool()
    chain_selector = create_chain_selector(strategy_pool, config)
    query_mutator = create_mutator_from_config(strategy_pool, config)
    
    # Chain executor
    from src.optimization.strategy_chain import StrategyChain
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
    
    logger.info("OptimizationEngine created successfully")
    
    return engine