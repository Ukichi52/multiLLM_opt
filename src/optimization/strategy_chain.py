# src/optimization/strategy_chain.py
from typing import List, Dict, Optional, Callable
import logging
from pathlib import Path

from src.storage.strategy_pool import StrategyPool

logger = logging.getLogger(__name__)


class StrategyChain:
    """
    Strategy Chain Executor
    
    Responsibilities:
    1. Execute a sequence of Sub-policies in order
    2. Pass context (image info, trajectory) between steps
    3. Record intermediate results for each step
    4. Support both predefined chains and dynamic chains
    
    Design Pattern: Chain of Responsibility
    - Each Sub-policy processes the query and passes to next
    - Each step can access all previous steps' results
    """
    
    def __init__(
        self, 
        strategy_pool: StrategyPool,
        mutator: 'QueryMutator',  # Forward reference
        evaluator: Optional['Evaluator'] = None  # Optional: for scoring each step
    ):
        """
        Initialize StrategyChain
        
        Args:
            strategy_pool: StrategyPool instance for querying Sub-policies
            mutator: QueryMutator instance for executing mutations
            evaluator: (Optional) Evaluator for scoring each step
        
        Design decision:
        - Inject dependencies rather than create them internally
        - Follows Dependency Injection principle
        - Easier to test (can mock mutator/evaluator)
        """
        self.pool = strategy_pool
        self.mutator = mutator
        self.evaluator = evaluator
        
        # Load predefined chains
        self.predefined_chains = self._load_predefined_chains()
        
        logger.info(f"StrategyChain initialized with {len(self.predefined_chains)} predefined chains")
    
    def _load_predefined_chains(self) -> Dict[str, List[str]]:
        """
        Load predefined strategy chains
        
        These are "expert-designed" chains based on domain knowledge.
        Will be used in early-stage optimization (cold start).
        
        Returns:
            Dict mapping chain_name to list of sub_policy_ids
        
        Future enhancement:
        - Load from YAML file (config/predefined_chains.yaml)
        - Allow users to define custom chains
        """
        return {
            "harm_to_medical": [
                "passive_voice",                    # Step 1: Reduce agency
                "harm_specification_downscaling",   # Step 2: Downscale to medical terms
                "perpetrator_to_victim",            # Step 3: Shift to victim perspective
                "image_scene_expansion",            # Step 4: Add image context
                "root_cause_analysis"               # Step 5: Ask for causes
            ],
            
            "vandalism_to_repair": [
                "passive_voice",                    # Reduce agency
                "perpetrator_to_victim",            # I was affected
                "image_scene_expansion",            # Describe the scene
                "white_knight_spirit",              # Moral reversal
                "root_cause_analysis"               # How to fix
            ],
            
            "intrusion_to_defense": [
                "passive_voice",
                "perpetrator_to_victim",
                "professional_identity",            # Security engineer
                "conditional_framing",              # Authorized context
                "root_cause_analysis"
            ],
            
            "destruction_to_analysis": [
                "passive_voice",
                "technical_engineering",            # Engineering terminology
                "future_to_past",                   # Case study
                "image_scene_expansion",
                "mechanism_analysis"                # How it works
            ],
            
            "abstract_exploration": [
                "abstraction",                      # Theoretical level
                "conditional_framing",              # Research context
                "mechanism_analysis"
            ]
        }
    
    def execute(
        self, 
        initial_query: str,
        chain: List[str],
        context: Dict,
        evaluate_each_step: bool = False
    ) -> Dict:
        """
        Execute a strategy chain
        
        Args:
            initial_query: Starting query
            chain: List of sub_policy_ids to execute in order
            context: Shared context including:
                - image_context: str (scene type)
                - image_description: str (detailed description)
                - key_objects: List[str] (optional)
                - Any other info needed by Sub-policies
            evaluate_each_step: If True, score query after each step
        
        Returns:
            {
                'initial_query': str,
                'final_query': str,
                'chain': List[str],
                'steps': [
                    {
                        'step': int,
                        'sub_id': str,
                        'query_before': str,
                        'query_after': str,
                        'scores': Dict (if evaluated),
                        'delta_score': float (if evaluated)
                    },
                    ...
                ],
                'total_score_gain': float (if evaluated)
            }
        
        Design decision:
        - Return full trajectory rather than just final query
        - Enables credit assignment and analysis
        """
        logger.info(f"Executing chain with {len(chain)} steps: {chain}")
        
        # Initialize trajectory
        trajectory = {
            'initial_query': initial_query,
            'final_query': initial_query,  # Will be updated
            'chain': chain,
            'steps': [],
            'total_score_gain': 0.0
        }
        
        current_query = initial_query
        previous_score = None
        
        # Execute each Sub-policy in sequence
        for step_idx, sub_id in enumerate(chain, start=1):
            logger.debug(f"Step {step_idx}/{len(chain)}: Executing {sub_id}")
            
            # Validate Sub-policy exists
            sub = self.pool.get_sub_policy(sub_id)
            if sub is None:
                logger.warning(f"Sub-policy '{sub_id}' not found, skipping")
                continue
            
            # Execute mutation
            try:
                query_before = current_query
                query_after = self.mutator.mutate(
                    query=current_query,
                    sub_policy_id=sub_id,
                    **context  # Unpack image_context, image_description, etc.
                )
                
                # Update current query
                current_query = query_after
                
            except Exception as e:
                logger.error(f"Error executing {sub_id}: {e}")
                query_after = current_query  # Fallback: keep unchanged
            
            # Evaluate (optional)
            step_scores = None
            delta_score = 0.0
            
            if evaluate_each_step and self.evaluator:
                try:
                    step_scores = self.evaluator.evaluate(query_after)
                    current_score = step_scores.get('total', 0.0)
                    
                    if previous_score is not None:
                        delta_score = current_score - previous_score
                    
                    previous_score = current_score
                    
                except Exception as e:
                    logger.error(f"Error evaluating step {step_idx}: {e}")
            
            # Record step
            step_record = {
                'step': step_idx,
                'sub_id': sub_id,
                'sub_name': sub.get('name', sub_id),
                'query_before': query_before,
                'query_after': query_after,
                'scores': step_scores,
                'delta_score': delta_score
            }
            
            trajectory['steps'].append(step_record)
            
            logger.debug(f"Step {step_idx} completed. Delta score: {delta_score:+.3f}")
        
        # Finalize trajectory
        trajectory['final_query'] = current_query
        
        if evaluate_each_step and len(trajectory['steps']) > 0:
            # Calculate total score gain
            first_score = trajectory['steps'][0].get('scores', {}).get('total', 0.0)
            last_score = trajectory['steps'][-1].get('scores', {}).get('total', 0.0)
            trajectory['total_score_gain'] = last_score - first_score
        
        logger.info(f"Chain execution completed. Total score gain: {trajectory['total_score_gain']:+.3f}")
        
        return trajectory
    
    def execute_predefined(
        self,
        chain_name: str,
        initial_query: str,
        context: Dict,
        evaluate_each_step: bool = False
    ) -> Dict:
        """
        Execute a predefined chain by name
        
        Convenience method for using expert-designed chains.
        
        Args:
            chain_name: Name of predefined chain (e.g., "harm_to_medical")
            initial_query: Starting query
            context: Shared context
            evaluate_each_step: Whether to score each step
        
        Returns:
            Same as execute()
        
        Raises:
            ValueError: If chain_name not found
        """
        if chain_name not in self.predefined_chains:
            available = list(self.predefined_chains.keys())
            raise ValueError(f"Chain '{chain_name}' not found. Available: {available}")
        
        chain = self.predefined_chains[chain_name]
        return self.execute(initial_query, chain, context, evaluate_each_step)
    
    def get_predefined_chains(self) -> List[str]:
        """Get list of available predefined chain names"""
        return list(self.predefined_chains.keys())
    
    def add_predefined_chain(self, name: str, chain: List[str]) -> bool:
        """
        Add a new predefined chain (for dynamic learning)
        
        Args:
            name: Chain name
            chain: List of sub_policy_ids
        
        Returns:
            True if added successfully, False if name already exists
        
        Future use case:
        - After discovering high-performing chains during exploration
        - User can add them as predefined chains for future use
        """
        if name in self.predefined_chains:
            logger.warning(f"Chain '{name}' already exists")
            return False
        
        # Validate all Sub-policies exist
        for sub_id in chain:
            if self.pool.get_sub_policy(sub_id) is None:
                logger.error(f"Invalid chain: Sub-policy '{sub_id}' not found")
                return False
        
        self.predefined_chains[name] = chain
        logger.info(f"Added predefined chain '{name}' with {len(chain)} steps")
        return True
    
    # ========== Future Extensions (Placeholder) ==========
    
    def execute_with_branching(self, *args, **kwargs):
        """
        Execute chain with conditional branching
        
        Not implemented yet. Future enhancement for:
        - "If step X fails, try alternative Y"
        - "If score < threshold, skip next step"
        """
        raise NotImplementedError("Branching not implemented yet")
    
    def optimize_chain_order(self, *args, **kwargs):
        """
        Use RL to optimize the order of Sub-policies
        
        Not implemented yet. Future enhancement for:
        - Learning optimal chain order for different query types
        - Genetic algorithm or gradient-based optimization
        """
        raise NotImplementedError("Chain optimization not implemented yet")


# ========== Helper Functions ==========

def visualize_trajectory(trajectory: Dict) -> str:
    """
    Pretty-print a trajectory for debugging
    
    Args:
        trajectory: Output from StrategyChain.execute()
    
    Returns:
        Formatted string
    
    Example output:
        Step 1: passive_voice (Δ: +0.02)
          Before: "how to harm someone"
          After:  "how can someone be harmed"
        
        Step 2: harm_specification_downscaling (Δ: +0.15)
          Before: "how can someone be harmed"
          After:  "how can skin breakage occur"
        ...
    """
    lines = []
    lines.append(f"Initial Query: {trajectory['initial_query']}")
    lines.append(f"Chain: {' → '.join(trajectory['chain'])}")
    lines.append("-" * 80)
    
    for step in trajectory['steps']:
        delta = step.get('delta_score', 0.0)
        lines.append(f"\nStep {step['step']}: {step['sub_name']} (Δ: {delta:+.3f})")
        lines.append(f"  Before: \"{step['query_before']}\"")
        lines.append(f"  After:  \"{step['query_after']}\"")
        
        if step.get('scores'):
            scores_str = ", ".join(f"{k}={v:.2f}" for k, v in step['scores'].items())
            lines.append(f"  Scores: {scores_str}")
    
    lines.append("\n" + "=" * 80)
    lines.append(f"Final Query: {trajectory['final_query']}")
    lines.append(f"Total Score Gain: {trajectory['total_score_gain']:+.3f}")
    
    return "\n".join(lines)


def compare_chains(trajectories: List[Dict]) -> str:
    """
    Compare multiple chain executions side-by-side
    
    Args:
        trajectories: List of trajectory dicts from different chains
    
    Returns:
        Comparison table as string
    
    Use case:
    - Compare "harm_to_medical" vs "abstract_exploration"
    - See which chain achieves higher score gain
    """
    lines = []
    lines.append("Chain Comparison")
    lines.append("=" * 100)
    
    # Header
    header = f"{'Chain':<30} {'Steps':<10} {'Score Gain':<15} {'Final Query (preview)':<45}"
    lines.append(header)
    lines.append("-" * 100)
    
    # Sort by score gain (descending)
    sorted_traj = sorted(trajectories, key=lambda t: t.get('total_score_gain', 0), reverse=True)
    
    for traj in sorted_traj:
        chain_name = " → ".join(traj['chain'][:3]) + "..."  # First 3 steps
        if len(traj['chain']) <= 3:
            chain_name = " → ".join(traj['chain'])
        
        steps = len(traj['steps'])
        gain = traj.get('total_score_gain', 0.0)
        final_preview = traj['final_query'][:40] + "..." if len(traj['final_query']) > 40 else traj['final_query']
        
        row = f"{chain_name:<30} {steps:<10} {gain:+.3f}          {final_preview:<45}"
        lines.append(row)
    
    return "\n".join(lines)
