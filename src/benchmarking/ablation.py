import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import scipy.stats as stats
from scipy import spatial
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ..evolution.nsga2 import NSGA2
from ..evaluation.zero_cost_proxies import ZeroCostProxies
from ..search_space.encoding import ArchitectureEncoder, ConstraintAwareInitializer

class AblationStudy:
    """Comprehensive ablation studies for research quality validation."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.results = {}
    
    def search_strategy_ablation(self, 
                               population_size: int = 50,
                               generations: int = 40,
                               dataset: str = 'cifar10') -> Dict[str, Any]:
        """
        Compare different search strategies:
        - NSGA-II (our method)
        - NSGA-III (reference)
        - Single-objective GA
        - Random search
        - Grid search
        """
        strategies = ['nsga2', 'nsga3', 'single_objective', 'random', 'grid']
        results = {}
        
        for strategy in strategies:
            print(f"Running {strategy} search strategy...")
            
            if strategy == 'nsga2':
                # Our main method
                from ..evolution.nsga2 import NSGA2
                algorithm = NSGA2(population_size=population_size,
                                generations=generations)
                result = self._run_nsga2_search(algorithm, dataset)
            
            elif strategy == 'nsga3':
                # NSGA-III reference implementation
                result = self._run_nsga3_search(population_size, generations, dataset)
            
            elif strategy == 'single_objective':
                # Single-objective genetic algorithm
                result = self._run_single_objective_ga(population_size, generations, dataset)
            
            elif strategy == 'random':
                # Random search baseline
                result = self._run_random_search(population_size * generations, dataset)
            
            elif strategy == 'grid':
                # Grid search (limited scale)
                result = self._run_grid_search(dataset)
            
            results[strategy] = result
        
        # Compare Pareto fronts
        comparison = self._compare_pareto_fronts(results)
        
        self.results['search_strategy'] = {
            'individual_results': results,
            'comparison': comparison
        }
        
        return self.results['search_strategy']
    
    def proxy_ablation(self, 
                      population_size: int = 50,
                      generations: int = 40) -> Dict[str, Any]:
        """
        Ablation study on zero-cost proxies:
        - Individual proxies
        - Ensemble combinations
        - Proxy-free baseline (full training)
        """
        proxy_configs = {
            'synflow_only': ['synflow'],
            'naswot_only': ['naswot'],
            'grad_norm_only': ['grad_norm'],
            'zen_only': ['zen_score'],
            'ensemble_all': ['synflow', 'naswot', 'grad_norm', 'zen_score', 'params'],
            'no_proxies': []  # Full training baseline
        }
        
        results = {}
        
        for config_name, proxies in proxy_configs.items():
            print(f"Running proxy ablation: {config_name}")
            
            # Configure zero-cost evaluator
            from ..evaluation.zero_cost_proxies import ZeroCostProxies
            zc_evaluator = ZeroCostProxies(self.device)
            
            if proxies:
                # Use specified proxies
                zc_evaluator.proxy_weights = {proxy: 1.0/len(proxies) for proxy in proxies}
            else:
                # No proxies - will use full training
                pass
            
            # Run search
            result = self._run_search_with_proxy_config(zc_evaluator, 
                                                      population_size, generations)
            results[config_name] = result
        
        # Statistical comparison
        stats_comparison = self._statistical_comparison(results)
        
        self.results['proxy_ablation'] = {
            'config_results': results,
            'statistics': stats_comparison
        }
        
        return self.results['proxy_ablation']
    
    def search_space_ablation(self) -> Dict[str, Any]:
        """
        Ablate different search space configurations:
        - Operation pools (with/without expensive operations)
        - Cell structures (number of nodes)
        - Parameter constraints
        """
        space_configs = {
            'efficient_ops': {
                'operations': ['sep_conv_3x3', 'sep_conv_5x5', 'dil_sep_conv_3x3',
                              'skip_connect', 'avg_pool_3x3', 'max_pool_3x3'],
                'steps': 5,
                'param_budget': 2e6
            },
            'extended_ops': {
                'operations': ['sep_conv_3x3', 'sep_conv_5x5', 'sep_conv_7x7',
                              'dil_sep_conv_3x3', 'dil_sep_conv_5x5',
                              'group_conv_2x2', 'group_conv_4x4',
                              'inv_bottleneck_2', 'inv_bottleneck_6',
                              'se_inv_bottleneck', 'skip_connect',
                              'avg_pool_3x3', 'max_pool_3x3'],
                'steps': 7,
                'param_budget': 5e6
            },
            'minimal_ops': {
                'operations': ['sep_conv_3x3', 'skip_connect', 'avg_pool_3x3'],
                'steps': 3,
                'param_budget': 1e6
            }
        }
        
        results = {}
        
        for config_name, config in space_configs.items():
            print(f"Running search space ablation: {config_name}")
            
            # Configure search space
            from ..search_space.encoding import ArchitectureEncoder
            encoder = ArchitectureEncoder(steps=config['steps'])
            
            # Modify operation pool based on config
            from ..search_space.operations import OPS
            available_ops = {op: OPS[op] for op in config['operations'] if op in OPS}
            
            # Run search with modified space
            result = self._run_search_with_space_config(encoder, available_ops,
                                                      config['param_budget'])
            results[config_name] = result
        
        # Compare search space diversity and quality
        space_comparison = self._compare_search_spaces(results)
        
        self.results['search_space_ablation'] = {
            'space_results': results,
            'comparison': space_comparison
        }
        
        return self.results['search_space_ablation']
    
    def population_dynamics_ablation(self) -> Dict[str, Any]:
        """
        Study population dynamics:
        - Population sizes: 20, 50, 100
        - Generation counts: 20, 40, 60
        - Genetic operator rates
        """
        dynamics_configs = {
            'pop20_gen20': {'population_size': 20, 'generations': 20},
            'pop50_gen40': {'population_size': 50, 'generations': 40},
            'pop100_gen60': {'population_size': 100, 'generations': 60},
            'high_mutation': {'mutation_rate': 0.5, 'crossover_rate': 0.7},
            'low_mutation': {'mutation_rate': 0.1, 'crossover_rate': 0.9},
        }
        
        results = {}
        
        for config_name, config in dynamics_configs.items():
            print(f"Running population dynamics: {config_name}")
            
            # Configure NSGA-II with specific parameters
            from ..evolution.nsga2 import NSGA2
            
            base_params = {
                'population_size': 50,
                'generations': 40,
                'crossover_rate': 0.9,
                'mutation_rate': 0.3
            }
            base_params.update(config)
            
            algorithm = NSGA2(**base_params)
            result = self._run_nsga2_search(algorithm, 'cifar10')
            results[config_name] = result
        
        # Analyze convergence and diversity
        dynamics_analysis = self._analyze_population_dynamics(results)
        
        self.results['population_dynamics'] = {
            'dynamics_results': results,
            'analysis': dynamics_analysis
        }
        
        return self.results['population_dynamics']
    
    def hardware_config_ablation(self) -> Dict[str, Any]:
        """
        Test different hardware configurations:
        - CPU-only
        - CPU with mixed precision (BFloat16)
        - Batched inference optimization
        """
        hardware_configs = {
            'cpu_fp32': {'device': 'cpu', 'dtype': torch.float32, 'batch_size': 1},
            'cpu_bf16': {'device': 'cpu', 'dtype': torch.bfloat16, 'batch_size': 1},
            'cpu_batch8': {'device': 'cpu', 'dtype': torch.float32, 'batch_size': 8},
            'cpu_batch32': {'device': 'cpu', 'dtype': torch.float32, 'batch_size': 32},
        }
        
        results = {}
        
        # Test on a set of discovered architectures
        architectures = self._get_sample_architectures(10)
        
        for config_name, config in hardware_configs.items():
            print(f"Testing hardware configuration: {config_name}")
            
            latency_results = []
            throughput_results = []
            
            for arch in architectures:
                latency, throughput = self._benchmark_hardware_config(arch, config)
                latency_results.append(latency)
                throughput_results.append(throughput)
            
            results[config_name] = {
                'mean_latency': np.mean(latency_results),
                'std_latency': np.std(latency_results),
                'mean_throughput': np.mean(throughput_results),
                'std_throughput': np.std(throughput_results),
                'latency_values': latency_results,
                'throughput_values': throughput_results
            }
        
        self.results['hardware_config'] = results
        return self.results['hardware_config']
    
    def _compare_pareto_fronts(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare Pareto fronts using hypervolume and statistical tests."""
        comparison = {}
        
        # Calculate hypervolume indicators
        ref_point = np.array([0, 100, 5])  # [accuracy, latency, params]
        
        for strategy, result in results.items():
            if 'pareto_front' in result:
                front = result['pareto_front']
                hypervolume = self._calculate_hypervolume(front, ref_point)
                comparison[strategy] = {
                    'hypervolume': hypervolume,
                    'front_size': len(front)
                }
        
        # Statistical significance tests
        strategies = list(results.keys())
        for i, strat1 in enumerate(strategies):
            for strat2 in strategies[i+1:]:
                if 'accuracy_values' in results[strat1] and 'accuracy_values' in results[strat2]:
                    # Wilcoxon signed-rank test
                    stat, p_value = stats.wilcoxon(
                        results[strat1]['accuracy_values'],
                        results[strat2]['accuracy_values']
                    )
                    comparison[f'{strat1}_vs_{strat2}'] = {
                        'wilcoxon_stat': stat,
                        'wilcoxon_p': p_value
                    }
        
        return comparison
    
    def _calculate_hypervolume(self, front: np.ndarray, ref_point: np.ndarray) -> float:
        """Calculate hypervolume indicator for Pareto front."""
        # Simple hypervolume calculation (for 3 objectives)
        if len(front) == 0:
            return 0.0
        
        # Normalize objectives
        front_norm = front.copy()
        front_norm[:, 0] = front_norm[:, 0] / 100  # Accuracy (maximize)
        front_norm[:, 1] = 1 - front_norm[:, 1] / 100  # Latency (minimize)
        front_norm[:, 2] = 1 - front_norm[:, 2] / 5  # Params (minimize)
        
        # Calculate dominated hypervolume
        hypervolume = 0.0
        for point in front_norm:
            # Volume of hyperrectangle dominated by this point
            volume = np.prod(point)
            hypervolume += volume
        
        return hypervolume / len(front)
    
    def _statistical_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical comparison of different configurations."""
        stats_results = {}
        
        # Collect accuracy distributions
        accuracy_distributions = {}
        for config, result in results.items():
            if 'accuracy_values' in result:
                accuracy_distributions[config] = result['accuracy_values']
        
        # ANOVA test for multiple comparisons
        if len(accuracy_distributions) >= 2:
            f_stat, p_value = stats.f_oneway(*accuracy_distributions.values())
            stats_results['anova'] = {
                'f_statistic': f_stat,
                'p_value': p_value
            }
        
        # Pairwise t-tests
        configs = list(accuracy_distributions.keys())
        for i, config1 in enumerate(configs):
            for config2 in configs[i+1:]:
                t_stat, p_value = stats.ttest_ind(
                    accuracy_distributions[config1],
                    accuracy_distributions[config2]
                )
                stats_results[f'{config1}_vs_{config2}'] = {
                    't_statistic': t_stat,
                    'p_value': p_value
                }
        
        return stats_results
    
    def _compare_search_spaces(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different search space configurations."""
        comparison = {}
        
        for space_name, result in results.items():
            if 'architectures' in result:
                architectures = result['architectures']
                
                # Calculate diversity metrics
                diversity = self._calculate_architecture_diversity(architectures)
                
                # Calculate quality metrics
                if 'accuracy_values' in result:
                    quality = {
                        'mean_accuracy': np.mean(result['accuracy_values']),
                        'std_accuracy': np.std(result['accuracy_values']),
                        'best_accuracy': np.max(result['accuracy_values'])
                    }
                else:
                    quality = {}
                
                comparison[space_name] = {
                    'diversity': diversity,
                    'quality': quality
                }
        
        return comparison
    
    def _calculate_architecture_diversity(self, architectures: List[Any]) -> Dict[str, float]:
        """Calculate diversity metrics for a set of architectures."""
        if len(architectures) < 2:
            return {'diversity': 0.0}
        
        # Convert architectures to feature vectors
        encoder = ArchitectureEncoder()
        feature_vectors = []
        
        for arch in architectures:
            int_encoding = encoder.genome_to_integer(arch)
            feature_vectors.append(int_encoding)
        
        feature_vectors = np.array(feature_vectors)
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(feature_vectors)):
            for j in range(i+1, len(feature_vectors)):
                dist = spatial.distance.hamming(feature_vectors[i], feature_vectors[j])
                distances.append(dist)
        
        return {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        }
    
    def _analyze_population_dynamics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze population convergence and diversity over generations."""
        analysis = {}
        
        for config_name, result in results.items():
            if 'history' in result:
                history = result['history']
                
                # Analyze convergence
                convergence = self._analyze_convergence(history)
                
                # Analyze diversity
                diversity = self._analyze_diversity(history)
                
                analysis[config_name] = {
                    'convergence': convergence,
                    'diversity': diversity
                }
        
        return analysis
    
    def _analyze_convergence(self, history: Dict[str, Any]) -> Dict[str, float]:
        """Analyze convergence metrics from search history."""
        if 'fitness' not in history or len(history['fitness']) == 0:
            return {}
        
        fitness_history = history['fitness']
        
        # Calculate improvement over generations
        initial_best = np.max(fitness_history[0][:, 0])  # Best accuracy in gen 0
        final_best = np.max(fitness_history[-1][:, 0])   # Best accuracy in final gen
        
        improvement = final_best - initial_best
        
        # Calculate convergence rate
        improvements = []
        for i in range(1, len(fitness_history)):
            prev_best = np.max(fitness_history[i-1][:, 0])
            curr_best = np.max(fitness_history[i][:, 0])
            improvements.append(curr_best - prev_best)
        
        convergence_rate = np.mean(improvements) if improvements else 0
        
        return {
            'initial_best': initial_best,
            'final_best': final_best,
            'absolute_improvement': improvement,
            'relative_improvement': improvement / initial_best if initial_best > 0 else 0,
            'convergence_rate': convergence_rate
        }
    
    def _analyze_diversity(self, history: Dict[str, Any]) -> Dict[str, List[float]]:
        """Analyze population diversity over generations."""
        if 'population' not in history or len(history['population']) == 0:
            return {}
        
        population_history = history['population']
        diversity_metrics = {
            'unique_architectures': [],
            'average_distance': []
        }
        
        encoder = ArchitectureEncoder()
        
        for gen_population in population_history:
            # Count unique architectures
            arch_strings = [str(arch) for arch in gen_population]
            unique_count = len(set(arch_strings))
            diversity_metrics['unique_architectures'].append(unique_count)
            
            # Calculate average pairwise distance
            if len(gen_population) > 1:
                feature_vectors = [encoder.genome_to_integer(arch) for arch in gen_population]
                distances = []
                for i in range(len(feature_vectors)):
                    for j in range(i+1, len(feature_vectors)):
                        dist = spatial.distance.hamming(feature_vectors[i], feature_vectors[j])
                        distances.append(dist)
                diversity_metrics['average_distance'].append(np.mean(distances))
            else:
                diversity_metrics['average_distance'].append(0.0)
        
        return diversity_metrics
    
    def _run_nsga2_search(self, algorithm: NSGA2, dataset: str) -> Dict[str, Any]:
        """Run NSGA-II search with given configuration."""
        # This would integrate with your existing search infrastructure
        # Placeholder implementation
        return {
            'pareto_front': np.random.random((10, 3)) * np.array([100, 100, 5]),
            'accuracy_values': np.random.normal(85, 5, 10),
            'architectures': [{'test': 'architecture'} for _ in range(10)]
        }
    
    def _get_sample_architectures(self, n: int) -> List[Dict]:
        """Get sample architectures for hardware testing."""
        encoder = ArchitectureEncoder()
        initializer = ConstraintAwareInitializer(encoder)
        return initializer.initialize_population(n)
    
    def _benchmark_hardware_config(self, arch: Dict, config: Dict) -> Tuple[float, float]:
        """Benchmark architecture on specific hardware configuration."""
        # Placeholder implementation
        return np.random.uniform(1, 10), np.random.uniform(100, 1000)

    def generate_report(self) -> str:
        """Generate comprehensive ablation study report."""
        report = []
        report.append("EFFICIENT NAS ABLATION STUDY REPORT")
        report.append("=" * 50)
        
        for study_name, study_results in self.results.items():
            report.append(f"\n{study_name.upper()} RESULTS")
            report.append("-" * 30)
            
            if 'comparison' in study_results:
                for key, value in study_results['comparison'].items():
                    report.append(f"{key}: {value}")
            
            if 'statistics' in study_results:
                for test, results in study_results['statistics'].items():
                    report.append(f"{test}: p={results.get('p_value', 'N/A'):.4f}")
        
        return "\n".join(report)

class Visualization:
    """Visualization tools for ablation study results."""
    
    @staticmethod
    def plot_pareto_fronts(ablation_results: Dict[str, Any], 
                          save_path: str = None):
        """Plot comparison of Pareto fronts from different strategies."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        # Accuracy vs Latency
        for i, (strategy, results) in enumerate(ablation_results.items()):
            if 'pareto_front' in results:
                front = results['pareto_front']
                axes[0].scatter(front[:, 1], front[:, 0], label=strategy, alpha=0.7)
        
        axes[0].set_xlabel('Latency (ms)')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Accuracy vs Latency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy vs Parameters
        for i, (strategy, results) in enumerate(ablation_results.items()):
            if 'pareto_front' in results:
                front = results['pareto_front']
                axes[1].scatter(front[:, 2], front[:, 0], label=strategy, alpha=0.7)
        
        axes[1].set_xlabel('Parameters (M)')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy vs Parameters')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Hypervolume comparison
        hypervolumes = []
        strategies = []
        for strategy, results in ablation_results.items():
            if 'comparison' in results and 'hypervolume' in results['comparison'].get(strategy, {}):
                hypervolumes.append(results['comparison'][strategy]['hypervolume'])
                strategies.append(strategy)
        
        if hypervolumes:
            axes[2].bar(strategies, hypervolumes)
            axes[2].set_ylabel('Hypervolume')
            axes[2].set_title('Hypervolume Comparison')
            axes[2].tick_params(axis='x', rotation=45)
        
        # Convergence plot
        if 'population_dynamics' in ablation_results:
            dynamics = ablation_results['population_dynamics']
            for config, results in dynamics.get('dynamics_results', {}).items():
                if 'history' in results and 'fitness' in results['history']:
                    fitness_history = results['history']['fitness']
                    best_accuracies = [np.max(f[:, 0]) for f in fitness_history]
                    axes[3].plot(best_accuracies, label=config)
            
            axes[3].set_xlabel('Generation')
            axes[3].set_ylabel('Best Accuracy (%)')
            axes[3].set_title('Convergence Comparison')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_proxy_ablation(proxy_results: Dict[str, Any], 
                           save_path: str = None):
        """Plot proxy ablation study results."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy distribution
        accuracy_data = []
        labels = []
        for config, results in proxy_results.get('config_results', {}).items():
            if 'accuracy_values' in results:
                accuracy_data.append(results['accuracy_values'])
                labels.append(config)
        
        if accuracy_data:
            axes[0].boxplot(accuracy_data, labels=labels)
            axes[0].set_ylabel('Accuracy (%)')
            axes[0].set_title('Accuracy Distribution by Proxy Config')
            axes[0].tick_params(axis='x', rotation=45)
        
        # Statistical significance
        if 'statistics' in proxy_results:
            p_values = []
            comparisons = []
            for key, stats in proxy_results['statistics'].items():
                if 'p_value' in stats:
                    p_values.append(stats['p_value'])
                    comparisons.append(key)
            
            if p_values:
                axes[1].bar(comparisons, -np.log10(p_values))
                axes[1].axhline(-np.log10(0.05), color='red', linestyle='--', 
                               label='p=0.05 threshold')
                axes[1].set_ylabel('-log10(p-value)')
                axes[1].set_title('Statistical Significance')
                axes[1].tick_params(axis='x', rotation=45)
                axes[1].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()