# src/analysis/strategy_analytics.py
"""Strategy analytics: Learn from historical trajectories"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class StrategyAnalytics:
    """
    Analyze historical trajectories to extract insights
    
    Learns:
    - Which Sub-policies are most effective (avg score delta)
    - Position preferences (which step is best for each Sub)
    - Chain performance (which chains succeed most)
    - Context-specific patterns (which Sub works in which context)
    """
    
    def __init__(self, trajectory_dir: str = "trajectories"):
        self.trajectory_dir = Path(trajectory_dir)
        self.trajectories = []
        
        # Statistics
        self.sub_stats = defaultdict(lambda: {
            'usage_count': 0,
            'total_delta': 0.0,
            'avg_delta': 0.0,
            'success_contribution': 0.0,
            'position_distribution': defaultdict(int),
            'context_performance': defaultdict(lambda: {'count': 0, 'total_delta': 0.0})
        })
        
        self.chain_stats = defaultdict(lambda: {
            'usage_count': 0,
            'success_count': 0,
            'total_score': 0.0,
            'avg_score': 0.0
        })
        
        self.context_stats = defaultdict(lambda: {
            'usage_count': 0,
            'success_count': 0
        })
        
        logger.info(f"StrategyAnalytics initialized: {trajectory_dir}")
    
    def load_trajectories(self, limit: Optional[int] = None):
        """
        Load trajectories from JSON files
        
        Args:
            limit: Maximum number to load (None = all)
        """
        if not self.trajectory_dir.exists():
            logger.warning(f"Trajectory directory not found: {self.trajectory_dir}")
            return
        
        files = sorted(self.trajectory_dir.glob("*.json"))
        
        if limit:
            files = files[:limit]
        
        logger.info(f"Found {len(files)} trajectory files")
        
        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    traj = json.load(f)
                    self.trajectories.append(traj)
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
        
        logger.info(f"Loaded {len(self.trajectories)} trajectories")
        
        # Analyze
        if self.trajectories:
            self._analyze()
    
    def _analyze(self):
        """Analyze all loaded trajectories"""
        logger.info("Analyzing trajectories...")
        
        for traj in self.trajectories:
            self._analyze_single(traj)
        
        # Normalize statistics
        for sub_id, stats in self.sub_stats.items():
            if stats['usage_count'] > 0:
                stats['avg_delta'] = stats['total_delta'] / stats['usage_count']
                
                # Normalize position distribution to probabilities
                total_pos = sum(stats['position_distribution'].values())
                if total_pos > 0:
                    stats['position_probs'] = {
                        pos: count / total_pos
                        for pos, count in stats['position_distribution'].items()
                    }
                
                # Context-specific averages
                for ctx, ctx_stats in stats['context_performance'].items():
                    if ctx_stats['count'] > 0:
                        ctx_stats['avg_delta'] = ctx_stats['total_delta'] / ctx_stats['count']
        
        for chain_name, stats in self.chain_stats.items():
            if stats['usage_count'] > 0:
                stats['success_rate'] = stats['success_count'] / stats['usage_count']
                stats['avg_score'] = stats['total_score'] / stats['usage_count']
        
        logger.info("Analysis complete")
    
    def _analyze_single(self, traj: Dict):
        """Analyze a single trajectory"""
        steps = traj.get('steps', [])
        success = traj.get('success', False)
        final_score = traj.get('final_score', 0.0)
        
        # Track context usage
        for context_hist in traj.get('context_history', []):
            context = context_hist.get('context_name')
            if context:
                self.context_stats[context]['usage_count'] += 1
                if context_hist.get('success'):
                    self.context_stats[context]['success_count'] += 1
            
            # Track chain used
            chain_info = context_hist.get('chain_used', {})
            chain_name = chain_info.get('chain_name')
            
            if chain_name:
                self.chain_stats[chain_name]['usage_count'] += 1
                self.chain_stats[chain_name]['total_score'] += final_score
                if context_hist.get('success'):
                    self.chain_stats[chain_name]['success_count'] += 1
        
        # Analyze each step
        for step_record in steps:
            sub_id = step_record.get('sub_policy_id')
            if not sub_id:
                continue
            
            delta = step_record.get('score_delta', 0.0)
            step_num = step_record.get('step', 0)
            context = step_record.get('context', 'unknown')
            
            # Update stats
            self.sub_stats[sub_id]['usage_count'] += 1
            self.sub_stats[sub_id]['total_delta'] += delta
            
            # Position distribution (normalize step to 0-9)
            # step 101 → 1, step 201 → 2, etc.
            normalized_step = min((step_num // 100) if step_num >= 100 else step_num, 9)
            self.sub_stats[sub_id]['position_distribution'][normalized_step] += 1
            
            # Context-specific performance
            self.sub_stats[sub_id]['context_performance'][context]['count'] += 1
            self.sub_stats[sub_id]['context_performance'][context]['total_delta'] += delta
            
            # Success contribution (credit assignment)
            if success and delta > 0:
                self.sub_stats[sub_id]['success_contribution'] += delta
    
    def get_top_subs(
        self,
        n: int = 10,
        metric: str = 'avg_delta',
        context: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Get top N Sub-policies by metric
        
        Args:
            n: Number to return
            metric: 'avg_delta', 'success_contribution', 'usage_count'
            context: Filter by context (None = all contexts)
        
        Returns:
            List of (sub_id, score) tuples
        """
        if context:
            # Context-specific ranking
            ranked = []
            for sub_id, stats in self.sub_stats.items():
                ctx_perf = stats['context_performance'].get(context, {})
                if ctx_perf.get('count', 0) > 0:
                    score = ctx_perf.get('avg_delta', 0.0)
                    ranked.append((sub_id, score))
            ranked.sort(key=lambda x: x[1], reverse=True)
        else:
            # Global ranking
            ranked = sorted(
                self.sub_stats.items(),
                key=lambda x: x[1].get(metric, 0.0),
                reverse=True
            )
            ranked = [(sub_id, stats[metric]) for sub_id, stats in ranked]
        
        return ranked[:n]
    
    def get_top_chains(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N chains by success rate"""
        chains_with_rate = []
        
        for chain_name, stats in self.chain_stats.items():
            if stats['usage_count'] > 0:
                success_rate = stats['success_rate']
                chains_with_rate.append((chain_name, success_rate))
        
        return sorted(chains_with_rate, key=lambda x: x[1], reverse=True)[:n]
    
    def get_sub_best_position(self, sub_id: str) -> int:
        """Get the position where this Sub performs best"""
        if sub_id not in self.sub_stats:
            return 0
        
        pos_dist = self.sub_stats[sub_id]['position_distribution']
        if not pos_dist:
            return 0
        
        # Return position with highest usage
        return max(pos_dist.items(), key=lambda x: x[1])[0]
    
    def get_sub_best_context(self, sub_id: str) -> Optional[str]:
        """Get the context where this Sub performs best"""
        if sub_id not in self.sub_stats:
            return None
        
        ctx_perf = self.sub_stats[sub_id]['context_performance']
        if not ctx_perf:
            return None
        
        # Return context with highest avg_delta
        best_ctx = max(
            ctx_perf.items(),
            key=lambda x: x[1].get('avg_delta', 0.0)
        )
        
        return best_ctx[0] if best_ctx[1].get('count', 0) > 0 else None
    
    def recommend_chain(self, context: Optional[str] = None) -> Optional[str]:
        """
        Recommend best chain based on historical performance
        
        Args:
            context: Image context (None = use global best)
        
        Returns:
            Chain name or None
        """
        if context and context in self.context_stats:
            # Find chains that performed well in this context
            # (simplified: use global best chain for now)
            pass
        
        # Return global best chain
        top_chains = self.get_top_chains(n=1)
        if top_chains and top_chains[0][1] > 0.3:  # At least 30% success rate
            return top_chains[0][0]
        
        return None
    
    def print_report(self):
        """Print comprehensive analysis report"""
        print("\n" + "=" * 80)
        print("STRATEGY ANALYTICS REPORT")
        print("=" * 80)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Total Trajectories: {len(self.trajectories)}")
        
        if self.trajectories:
            success_count = sum(1 for t in self.trajectories if t.get('success', False))
            print(f"  Successful: {success_count} ({success_count/len(self.trajectories)*100:.1f}%)")
            
            avg_score = sum(t.get('final_score', 0.0) for t in self.trajectories) / len(self.trajectories)
            print(f"  Average Final Score: {avg_score:.3f}")
        
        print(f"\n🏆 Top 10 Sub-Policies (by avg score delta):")
        for i, (sub_id, delta) in enumerate(self.get_top_subs(10, 'avg_delta'), 1):
            stats = self.sub_stats[sub_id]
            best_pos = self.get_sub_best_position(sub_id)
            print(f"  {i:2d}. {sub_id:35s} | Δ{delta:+.3f} | Uses: {stats['usage_count']:3d} | Best@Step{best_pos}")
        
        print(f"\n📈 Top 5 Strategy Chains (by success rate):")
        for i, (chain_name, rate) in enumerate(self.get_top_chains(5), 1):
            stats = self.chain_stats[chain_name]
            print(f"  {i}. {chain_name:25s} | Success: {rate:.1%} | Uses: {stats['usage_count']:3d} | Avg Score: {stats['avg_score']:.3f}")
        
        print(f"\n🗺️  Context Performance:")
        for context, stats in sorted(self.context_stats.items(), key=lambda x: x[1]['success_count'], reverse=True):
            if stats['usage_count'] > 0:
                rate = stats['success_count'] / stats['usage_count']
                print(f"  {context:25s} | Success: {rate:.1%} ({stats['success_count']}/{stats['usage_count']})")
        
        print("\n" + "=" * 80)
    
    def export_recommendations(self) -> Dict:
        """
        Export actionable recommendations for ChainSelector
        
        Returns:
            Dict with recommendations
        """
        return {
            'top_chains': {
                name: {'success_rate': rate, 'recommendation': 'preferred' if rate > 0.5 else 'consider'}
                for name, rate in self.get_top_chains(10)
            },
            'top_subs': {
                sub_id: {
                    'avg_delta': delta,
                    'best_position': self.get_sub_best_position(sub_id),
                    'best_context': self.get_sub_best_context(sub_id)
                }
                for sub_id, delta in self.get_top_subs(20, 'avg_delta')
            },
            'context_success_rates': {
                ctx: stats['success_count'] / stats['usage_count']
                for ctx, stats in self.context_stats.items()
                if stats['usage_count'] > 0
            }
        }