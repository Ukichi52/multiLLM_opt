#!/usr/bin/env python3
"""
Main entry point for multimodal jailbreak optimization
"""
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.optimization.optimization_engine import create_optimization_engine
from src.utils.data_loader import load_dataset
from src.utils.config_loader import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def optimize_single(args):
    """Optimize a single query"""
    logger.info("=" * 80)
    logger.info("MODE: Single Query Optimization")
    logger.info("=" * 80)
    
    # Load dataset
    dataset = load_dataset()
    
    # Get query by index or random
    if args.query_index is not None:
        item = dataset[args.query_index]
        logger.info(f"Using query at index {args.query_index}")
    else:
        from src.utils.data_loader import get_random_query
        item = get_random_query(seed=args.seed)
        logger.info(f"Using random query (seed={args.seed})")
    
    logger.info(f"Image: {item['image_id']}")
    logger.info(f"Query: {item['query']}")
    
    # Create engine
    logger.info("Creating optimization engine...")
    engine = create_optimization_engine()
    
    # Optimize
    result = engine.optimize_single_query(
        image_path=item['image_path'],
        query=item['query'],
        query_id=args.query_id,
        image_id=item['image_id'],
        save_trajectory=True
    )
    
    # Print results
    logger.info("=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info(f"Success: {result['success']}")
    logger.info(f"Final Score: {result['final_score']:.3f}")
    logger.info(f"Total Steps: {result['total_steps']}")
    logger.info(f"Successful Context: {result.get('successful_context', 'N/A')}")
    
    if result['success']:
        logger.info(f"\nInitial Query: {item['query']}")
        logger.info(f"Final Query: {result['final_query'][:200]}...")
        if 'llm_tier' in result:
            logger.info(f"LLM Tier: {result['llm_tier']}")
    
    logger.info("=" * 80)


def optimize_batch(args):
    """Optimize a batch of queries"""
    logger.info("=" * 80)
    logger.info("MODE: Batch Optimization")
    logger.info("=" * 80)
    
    # Load dataset
    dataset = load_dataset()
    
    # Determine range
    start_idx = args.start_index
    end_idx = args.end_index if args.end_index else min(start_idx + args.batch_size, len(dataset))
    
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Processing: {start_idx} to {end_idx} ({end_idx - start_idx} queries)")
    
    # Create engine
    logger.info("Creating optimization engine...")
    engine = create_optimization_engine()
    
    # Run batch
    results = engine.optimize_batch(
        dataset=dataset,
        start_idx=start_idx,
        end_idx=end_idx,
        save_trajectories=True
    )
    
    # Summary statistics
    success_count = sum(1 for r in results if r.get('success', False))
    avg_score = sum(r.get('final_score', 0.0) for r in results) / len(results)
    avg_steps = sum(r.get('total_steps', 0) for r in results) / len(results)
    
    logger.info("=" * 80)
    logger.info("BATCH RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total Queries: {len(results)}")
    logger.info(f"Successes: {success_count} ({success_count/len(results)*100:.1f}%)")
    logger.info(f"Average Score: {avg_score:.3f}")
    logger.info(f"Average Steps: {avg_steps:.1f}")
    logger.info("=" * 80)
    
    # Save summary
    import json
    summary_path = f"batch_summary_{start_idx}_{end_idx}.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'start_index': start_idx,
            'end_index': end_idx,
            'total': len(results),
            'success_count': success_count,
            'success_rate': success_count / len(results),
            'avg_score': avg_score,
            'avg_steps': avg_steps,
            'results': results
        }, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_path}")


def analyze_trajectories(args):
    """Analyze collected trajectories"""
    logger.info("=" * 80)
    logger.info("MODE: Trajectory Analysis")
    logger.info("=" * 80)
    
    from src.analysis.strategy_analytics import StrategyAnalytics
    
    # Create analytics
    analytics = StrategyAnalytics(trajectory_dir=args.trajectory_dir)
    
    # Load trajectories
    logger.info(f"Loading trajectories from: {args.trajectory_dir}")
    analytics.load_trajectories(limit=args.limit)
    
    # Print report
    analytics.print_report()
    
    # Save detailed stats
    if args.save_stats:
        import json
        stats_path = "strategy_stats.json"
        with open(stats_path, 'w') as f:
            json.dump({
                'sub_stats': dict(analytics.sub_stats),
                'chain_stats': dict(analytics.chain_stats),
                'top_subs': analytics.get_top_subs(20, 'avg_delta'),
                'top_chains': analytics.get_top_chains(10)
            }, f, indent=2, default=str)
        logger.info(f"Detailed stats saved to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Multimodal Jailbreak Optimization Framework'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # ========== Single Query Mode ==========
    single_parser = subparsers.add_parser('single', help='Optimize a single query')
    single_parser.add_argument(
        '--query-index', type=int, default=None,
        help='Index of query in dataset (default: random)'
    )
    single_parser.add_argument(
        '--query-id', type=str, default=None,
        help='Custom query ID (default: auto-generated)'
    )
    single_parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for query selection (default: 42)'
    )
    
    # ========== Batch Mode ==========
    batch_parser = subparsers.add_parser('batch', help='Optimize a batch of queries')
    batch_parser.add_argument(
        '--start-index', type=int, default=0,
        help='Start index in dataset (default: 0)'
    )
    batch_parser.add_argument(
        '--end-index', type=int, default=None,
        help='End index in dataset (default: start + batch_size)'
    )
    batch_parser.add_argument(
        '--batch-size', type=int, default=30,
        help='Number of queries to process (default: 30)'
    )
    
    # ========== Analysis Mode ==========
    analyze_parser = subparsers.add_parser('analyze', help='Analyze collected trajectories')
    analyze_parser.add_argument(
        '--trajectory-dir', type=str, default='trajectories',
        help='Directory containing trajectory files (default: trajectories)'
    )
    analyze_parser.add_argument(
        '--limit', type=int, default=None,
        help='Maximum number of trajectories to load (default: all)'
    )
    analyze_parser.add_argument(
        '--save-stats', action='store_true',
        help='Save detailed statistics to JSON file'
    )
    
    args = parser.parse_args()
    
    if args.command == 'single':
        optimize_single(args)
    elif args.command == 'batch':
        optimize_batch(args)
    elif args.command == 'analyze':
        analyze_trajectories(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()