# src/optimization/trajectory_logger.py
"""Trajectory logger for recording optimization paths"""
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class TrajectoryLogger:
    """
    Trajectory Logger: Record detailed optimization paths
    
    Records:
    - Each Sub-policy's effect (query change + score delta)
    - Context switches
    - Strategy chain performance
    - Final results
    
    Supports:
    - Top-K pruning (keep only best K paths)
    - Phase 4 RL learning (stores all info needed for credit assignment)
    """
    
    def __init__(
        self,
        save_dir: str = "trajectories",
        enable_top_k_pruning: bool = False,
        top_k: int = 2
    ):
        """
        Initialize logger
        
        Args:
            save_dir: Directory to save trajectory files
            enable_top_k_pruning: Whether to keep only top-K branches
            top_k: Number of best branches to keep
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_top_k_pruning = enable_top_k_pruning
        self.top_k = top_k
        
        # Current trajectory state
        self.current_trajectory = None
        self.current_context = None
        self.context_histories = []  # For tracking context switches
        
        logger.info(f"TrajectoryLogger initialized: save_dir={save_dir}, top_k={top_k if enable_top_k_pruning else 'disabled'}")
    
    def start_query(
        self,
        query_id: str,
        image_id: str,
        initial_query: str,
        image_path: Optional[str] = None
    ):
        """
        Start recording a new query optimization
        
        Args:
            query_id: Unique identifier for this query
            image_id: Image filename
            initial_query: Original query text
            image_path: Full path to image
        """
        self.current_trajectory = {
            'query_id': query_id,
            'image_id': image_id,
            'image_path': image_path,
            'initial_query': initial_query,
            'start_time': datetime.now().isoformat(),
            
            # Context exploration
            'contexts_explored': [],
            'current_context': None,
            'successful_context': None,
            
            # Optimization steps
            'steps': [],
            'current_step': 0,
            
            # Results
            'success': False,
            'converged': False,
            'final_query': initial_query,
            'final_response': None,
            'final_score': 0.0,
            
            # Statistics
            'total_steps': 0,
            'total_api_calls': 0,
            'total_time_seconds': 0.0
        }
        
        self.context_histories = []
        
        logger.info(f"Started logging query: {query_id}")
    
    def start_context(self, context: str, context_prob: float):
        """
        Start exploring a new context branch
        
        Args:
            context: Context name (e.g., "research_lab")
            context_prob: Probability/confidence of this context
        """
        if self.current_trajectory is None:
            raise RuntimeError("Must call start_query() before start_context()")
        
        self.current_context = {
            'context_name': context,
            'context_prob': context_prob,
            'start_step': self.current_trajectory['current_step'],
            'steps': [],
            'success': False,
            'final_score': 0.0
        }
        
        self.current_trajectory['current_context'] = context
        self.current_trajectory['contexts_explored'].append(context)
        
        logger.debug(f"Started context: {context} (prob={context_prob:.2f})")
    
    def log_chain_start(self, chain_name: str, chain: List[str], reason: str):
        """
        Log the start of a strategy chain execution
        
        Args:
            chain_name: Name of the chain (e.g., "harm_to_medical")
            chain: List of Sub-policy IDs
            reason: Why this chain was selected
        """
        chain_info = {
            'chain_name': chain_name,
            'chain': chain,
            'reason': reason,
            'start_step': self.current_trajectory['current_step']
        }
        
        if self.current_context:
            self.current_context['chain_used'] = chain_info
        
        logger.debug(f"Chain started: {chain_name} with {len(chain)} sub-policies")
    
    def log_step(
        self,
        step: int,
        sub_policy_id: str,
        sub_policy_name: str,
        query_before: str,
        query_after: str,
        response: Optional[str] = None,
        eval_result: Optional[Dict] = None,
        previous_score: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Log a single optimization step (one Sub-policy application)
        
        Args:
            step: Global step number
            sub_policy_id: ID of the Sub-policy used
            sub_policy_name: Human-readable name
            query_before: Query before this Sub-policy
            query_after: Query after this Sub-policy
            response: Target model's response (if evaluated)
            eval_result: Evaluation result dict
            previous_score: Score from previous step (for delta calculation)
            metadata: Additional metadata (e.g., selection reason for Phase 4)
        
        This is the KEY method for Phase 4 credit assignment!
        """
        if self.current_trajectory is None:
            raise RuntimeError("Must call start_query() before log_step()")
        
        # Calculate score delta
        current_score = eval_result['total_score'] if eval_result else 0.0
        score_delta = current_score - previous_score if previous_score is not None else 0.0
        
        # Build step record
        step_record = {
            'step': step,
            'sub_policy_id': sub_policy_id,
            'sub_policy_name': sub_policy_name,
            
            # Query evolution
            'query_before': query_before,
            'query_after': query_after,
            'query_changed': query_before != query_after,
            
            # Response (if evaluated)
            'response': response,
            
            # Evaluation
            'eval_result': eval_result,
            'score_before': previous_score,
            'score_after': current_score,
            'score_delta': score_delta,
            
            # Context
            'context': self.current_trajectory['current_context'],
            
            # Metadata (for Phase 4)
            'metadata': metadata or {},
            
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to trajectory
        self.current_trajectory['steps'].append(step_record)
        self.current_trajectory['current_step'] = step
        
        # Add to current context
        if self.current_context:
            self.current_context['steps'].append(step_record)
            self.current_context['final_score'] = current_score
        
        # Log
        if previous_score is not None:
            logger.debug(
                f"Step {step}: {sub_policy_name} | "
                f"Score: {previous_score:.3f} → {current_score:.3f} (Δ {score_delta:+.3f})"
            )
        else:
            logger.debug(
                f"Step {step}: {sub_policy_name} | "
                f"Score: N/A → {current_score:.3f} (first step)"
            )
    
    def log_context_switch(
        self,
        from_context: str,
        to_context: str,
        reason: str,
        from_score: float
    ):
        """
        Log switching from one context to another
        
        Args:
            from_context: Previous context
            to_context: New context
            reason: Why switching (e.g., "Failed", "Early stop")
            from_score: Final score achieved in previous context
        """
        if self.current_context:
            self.current_context['success'] = False
            self.current_context['exit_reason'] = reason
            self.context_histories.append(self.current_context)
        
        switch_record = {
            'from_context': from_context,
            'to_context': to_context,
            'reason': reason,
            'from_score': from_score,
            'step': self.current_trajectory['current_step']
        }
        
        if 'context_switches' not in self.current_trajectory:
            self.current_trajectory['context_switches'] = []
        
        self.current_trajectory['context_switches'].append(switch_record)
        
        logger.info(f"Context switch: {from_context} → {to_context} (reason: {reason})")
    
    def finalize(
        self,
        success: bool,
        final_query: str,
        final_response: Optional[str],
        final_score: float,
        converged: bool = False
    ):
        """
        Finalize trajectory and save to file
        
        Args:
            success: Whether optimization succeeded
            final_query: Final mutated query
            final_response: Final target model response
            final_score: Final evaluation score
            converged: Whether optimization converged (vs max steps)
        """
        if self.current_trajectory is None:
            raise RuntimeError("Must call start_query() before finalize()")
        
        # Finalize current context
        if self.current_context:
            self.current_context['success'] = success
            if success:
                self.current_trajectory['successful_context'] = self.current_context['context_name']
            self.context_histories.append(self.current_context)
        
        # Update trajectory
        self.current_trajectory['success'] = success
        self.current_trajectory['converged'] = converged
        self.current_trajectory['final_query'] = final_query
        self.current_trajectory['final_response'] = final_response
        self.current_trajectory['final_score'] = final_score
        self.current_trajectory['end_time'] = datetime.now().isoformat()
        
        # Calculate statistics
        start_time = datetime.fromisoformat(self.current_trajectory['start_time'])
        end_time = datetime.fromisoformat(self.current_trajectory['end_time'])
        
        self.current_trajectory['total_steps'] = len(self.current_trajectory['steps'])
        self.current_trajectory['total_time_seconds'] = (end_time - start_time).total_seconds()
        self.current_trajectory['total_api_calls'] = self._count_api_calls()
        
        # Context history
        self.current_trajectory['context_history'] = self.context_histories
        
        # Top-K pruning (if enabled)
        if self.enable_top_k_pruning:
            self._apply_top_k_pruning()
        
        # Save to file
        self._save_trajectory()
        
        logger.info(
            f"Finalized trajectory: {self.current_trajectory['query_id']} | "
            f"Success: {success} | Steps: {self.current_trajectory['total_steps']} | "
            f"Score: {final_score:.3f}"
        )
    
    def _count_api_calls(self) -> int:
        """Count total API calls (mutator + target + judge)"""
        # Simplified: 3 calls per step (mutator, target, judge)
        return len(self.current_trajectory['steps']) * 3
    
    def _apply_top_k_pruning(self):
        """
        Keep only top-K steps by score delta
        
        Used to reduce storage and focus analysis on impactful steps
        """
        steps = self.current_trajectory['steps']
        
        if len(steps) <= self.top_k:
            return
        
        # Sort by score_delta (descending)
        sorted_steps = sorted(steps, key=lambda s: s['score_delta'], reverse=True)
        
        # Keep top-K + always keep first and last
        top_k_steps = sorted_steps[:self.top_k]
        
        # Ensure first and last are included
        if steps[0] not in top_k_steps:
            top_k_steps.append(steps[0])
        if steps[-1] not in top_k_steps:
            top_k_steps.append(steps[-1])
        
        # Re-sort by step number
        top_k_steps = sorted(top_k_steps, key=lambda s: s['step'])
        
        # Mark pruned steps
        self.current_trajectory['pruned_steps'] = len(steps) - len(top_k_steps)
        self.current_trajectory['steps'] = top_k_steps
        
        logger.debug(f"Applied top-{self.top_k} pruning: {len(steps)} → {len(top_k_steps)} steps")
        
    def _clean_for_json(self, obj):
        """
        Recursively clean object for JSON serialization
        
        Simplified version - most conversions done at source
        """
        if obj is None:
            return None
        
        # Handle remaining edge cases
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # NumPy scalar
            return obj.item()
        
        # Collections
        elif isinstance(obj, dict):
            return {key: self._clean_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        
        # Base types
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        
        # Fallback
        else:
            return str(obj)
    
    def _save_trajectory(self):
        """Save trajectory to JSON file"""
        query_id = self.current_trajectory['query_id']
        filename = f"{query_id}.json"
        filepath = self.save_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.current_trajectory, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved trajectory to: {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to save trajectory: {e}")
    
    def get_trajectory(self) -> Dict:
        """
        Get current trajectory (for Phase 4 learning)
        
        Returns:
            Complete trajectory dict
        """
        return self.current_trajectory


# ========== Utility Functions ==========

def load_trajectory(filepath: str) -> Dict:
    """
    Load a saved trajectory from file
    
    Usage:
        traj = load_trajectory("trajectories/query_001.json")
        print(traj['success'])
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_trajectory(trajectory: Dict) -> Dict:
    """
    Analyze a trajectory to extract insights
    
    Returns statistics like:
    - Which Sub-policy contributed most
    - Average score gain per step
    - Best performing context
    
    Usage:
        traj = load_trajectory("trajectories/query_001.json")
        stats = analyze_trajectory(traj)
        print(stats['best_sub_policy'])
    """
    steps = trajectory['steps']
    
    if not steps:
        return {'error': 'No steps in trajectory'}
    
    # Sub-policy contributions
    sub_contributions = defaultdict(lambda: {'count': 0, 'total_delta': 0.0})
    
    for step in steps:
        sub_id = step['sub_policy_id']
        delta = step['score_delta']
        
        sub_contributions[sub_id]['count'] += 1
        sub_contributions[sub_id]['total_delta'] += delta
    
    # Calculate averages
    for sub_id, stats in sub_contributions.items():
        stats['avg_delta'] = stats['total_delta'] / stats['count']
    
    # Find best Sub-policy
    best_sub = max(
        sub_contributions.items(),
        key=lambda x: x[1]['avg_delta']
    )
    
    # Overall statistics
    total_score_gain = steps[-1]['score_after'] - steps[0]['score_before'] if len(steps) > 1 else 0.0
    avg_score_gain = total_score_gain / len(steps) if len(steps) > 0 else 0.0
    
    return {
        'total_steps': len(steps),
        'total_score_gain': total_score_gain,
        'avg_score_gain_per_step': avg_score_gain,
        'best_sub_policy': {
            'id': best_sub[0],
            'avg_delta': best_sub[1]['avg_delta'],
            'usage_count': best_sub[1]['count']
        },
        'sub_contributions': dict(sub_contributions),
        'success': trajectory['success'],
        'successful_context': trajectory.get('successful_context')
    }


def visualize_trajectory(trajectory: Dict) -> str:
    """
    Create a text visualization of trajectory
    
    Usage:
        traj = load_trajectory("trajectories/query_001.json")
        print(visualize_trajectory(traj))
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"Trajectory: {trajectory['query_id']}")
    lines.append(f"Success: {trajectory['success']} | Final Score: {trajectory['final_score']:.3f}")
    lines.append("=" * 80)
    
    lines.append(f"\nInitial Query: {trajectory['initial_query']}")
    lines.append(f"Final Query:   {trajectory['final_query']}")
    
    lines.append(f"\n{'Step':<6} {'Sub-Policy':<30} {'Score Before':<15} {'Score After':<15} {'Delta':<10}")
    lines.append("-" * 80)
    
    for step in trajectory['steps']:
        lines.append(
            f"{step['step']:<6} "
            f"{step['sub_policy_name'][:28]:<30} "
            f"{step['score_before']:<15.3f} "
            f"{step['score_after']:<15.3f} "
            f"{step['score_delta']:+.3f}"
        )
    
    lines.append("=" * 80)
    
    return "\n".join(lines)
