# experiments/scripts/run_benchmark.py
#!/usr/bin/env python3
"""
Main benchmark runner for Efficient NAS project.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import yaml
import argparse
from pathlib import Path
from src.benchmarking.baselines import BaselineManager
from src.benchmarking.ablation import AblationStudy, Visualization
from src.utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description='Run Efficient NAS Benchmarking')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use for training')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger(output_dir / 'benchmark.log')
    
    logger.info("Starting Efficient NAS Benchmarking")
    logger.info(f"Configuration: {config['experiment']['name']}")
    
    # Run baseline evaluations
    if 'baselines' in config:
        logger.info("Evaluating baseline models...")
        baseline_manager = BaselineManager(device=args.device)
        
        baseline_results = {}
        for dataset in ['cifar10', 'cifar100']:
            baseline_results[dataset] = baseline_manager.evaluate_all_baselines(dataset)
        
        # Save baseline results
        import json
        with open(output_dir / 'baseline_results.json', 'w') as f:
            json.dump(baseline_results, f, indent=2)
    
    # Run ablation studies
    if config['ablation_studies']['search_strategy']:
        logger.info("Running search strategy ablation...")
        ablation_study = AblationStudy(device=args.device)
        
        strategy_results = ablation_study.search_strategy_ablation()
        Visualization.plot_pareto_fronts(strategy_results['individual_results'],
                                       output_dir / 'search_strategy_comparison.png')
    
    if config['ablation_studies']['proxy_ablation']:
        logger.info("Running proxy ablation study...")
        proxy_results = ablation_study.proxy_ablation()
        Visualization.plot_proxy_ablation(proxy_results,
                                        output_dir / 'proxy_ablation.png')
    
    # Generate comprehensive report
    report = ablation_study.generate_report()
    with open(output_dir / 'ablation_report.txt', 'w') as f:
        f.write(report)
    
    logger.info("Benchmarking completed successfully!")
    logger.info(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()