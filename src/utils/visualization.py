import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE, UMAP
from sklearn.decomposition import PCA
import networkx as nx
from pathlib import Path
import json

class NASVisualizer:
    """Comprehensive visualization suite for Neural Architecture Search."""
    
    def __init__(self, output_dir: str = "./visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Style configuration
        plt.style.use('seaborn-v0_8')
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    def plot_pareto_front_evolution(self, 
                                  history: Dict[str, Any],
                                  save_path: Optional[str] = None,
                                  create_animation: bool = True):
        """
        Animate 3D Pareto front evolution across generations.
        
        Args:
            history: NSGA-II history containing fitness and populations
            save_path: Path to save visualization
            create_animation: Whether to create MP4 animation
        """
        generations = history['generation']
        fitness_history = history['fitness']
        pareto_fronts = history['pareto_front']
        
        if create_animation:
            self._create_pareto_animation(generations, fitness_history, pareto_fronts, save_path)
        else:
            self._create_pareto_static(generations, fitness_history, pareto_fronts, save_path)
    
    def _create_pareto_animation(self, generations: List[int],
                               fitness_history: List[np.ndarray],
                               pareto_fronts: List[Dict],
                               save_path: Optional[str]):
        """Create animated 3D Pareto front evolution."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            
            # Current generation data
            fitness = fitness_history[frame]
            pareto_front = pareto_fronts[frame]
            
            # Plot all architectures
            accuracy = fitness[:, 0]
            latency = fitness[:, 1]
            params = fitness[:, 2]
            
            ax.scatter(latency, params, accuracy, 
                      c='blue', alpha=0.3, s=20, label='All Architectures')
            
            # Plot Pareto front
            if pareto_front and len(pareto_front['fitness']) > 0:
                pf_accuracy = pareto_front['fitness'][:, 0]
                pf_latency = pareto_front['fitness'][:, 1]
                pf_params = pareto_front['fitness'][:, 2]
                
                ax.scatter(pf_latency, pf_params, pf_accuracy,
                          c='red', s=50, label='Pareto Front')
                
                # Convex hull for Pareto front
                from scipy.spatial import ConvexHull
                try:
                    points = np.column_stack([pf_latency, pf_params, pf_accuracy])
                    hull = ConvexHull(points)
                    
                    for simplex in hull.simplices:
                        ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2],
                               'r-', alpha=0.5)
                except:
                    pass  # Skip if convex hull fails
            
            ax.set_xlabel('Latency (ms)')
            ax.set_ylabel('Parameters (M)')
            ax.set_zlabel('Accuracy (%)')
            ax.set_title(f'Pareto Front Evolution - Generation {generations[frame]}')
            ax.legend()
            
            # Set consistent limits
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 5)
            ax.set_zlim(50, 100)
        
        anim = animation.FuncAnimation(fig, update, frames=len(generations),
                                     interval=500, repeat=True)
        
        if save_path:
            anim_path = self.output_dir / 'pareto_evolution.mp4'
            anim.save(anim_path, writer='ffmpeg', dpi=100)
            print(f"Animation saved to {anim_path}")
        
        plt.show()
    
    def _create_pareto_static(self, generations: List[int],
                            fitness_history: List[np.ndarray],
                            pareto_fronts: List[Dict],
                            save_path: Optional[str]):
        """Create static multi-generation Pareto front plot."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        # Select key generations to display
        key_generations = [0, len(generations)//4, len(generations)//2, 
                          3*len(generations)//4, len(generations)-1]
        
        for i, gen_idx in enumerate(key_generations[:6]):
            if gen_idx >= len(generations):
                continue
                
            fitness = fitness_history[gen_idx]
            pareto_front = pareto_fronts[gen_idx]
            
            # Accuracy vs Latency
            axes[i].scatter(fitness[:, 1], fitness[:, 0], 
                          alpha=0.5, s=30, label='All')
            
            if pareto_front and len(pareto_front['fitness']) > 0:
                pf_fitness = pareto_front['fitness']
                axes[i].scatter(pf_fitness[:, 1], pf_fitness[:, 0],
                              c='red', s=50, label='Pareto Front')
            
            axes[i].set_xlabel('Latency (ms)')
            axes[i].set_ylabel('Accuracy (%)')
            axes[i].set_title(f'Generation {generations[gen_idx]}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            static_path = self.output_dir / 'pareto_static.png'
            plt.savefig(static_path, dpi=300, bbox_inches='tight')
            print(f"Static plot saved to {static_path}")
        
        plt.show()
    
    def plot_architecture_genealogy(self, 
                                  genealogy_data: Dict[str, Any],
                                  save_path: Optional[str] = None):
        """
        Visualize parent-child relationships in evolutionary search.
        
        Args:
            genealogy_data: Dictionary containing parent-child mappings
            save_path: Path to save visualization
        """
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for child, parents in genealogy_data.items():
            G.add_node(child, architecture=child)
            for parent in parents:
                G.add_node(parent, architecture=parent)
                G.add_edge(parent, child)
        
        # Create layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        plt.figure(figsize=(15, 10))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=500, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, alpha=0.6)
        
        # Draw labels
        labels = {node: f"Arch_{i}" for i, node in enumerate(G.nodes())}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title('Architecture Genealogy Tree')
        plt.axis('off')
        
        if save_path:
            genealogy_path = self.output_dir / 'architecture_genealogy.png'
            plt.savefig(genealogy_path, dpi=300, bbox_inches='tight')
            print(f"Genealogy plot saved to {genealogy_path}")
        
        plt.show()
        
        # Also create interactive plotly version
        self._create_interactive_genealogy(G, genealogy_data)
    
    def _create_interactive_genealogy(self, G: nx.DiGraph, genealogy_data: Dict):
        """Create interactive genealogy plot with Plotly."""
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f'Architecture: {node[:20]}...')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                size=10,
                color='lightblue',
                line_width=2))
        
        node_trace.text = node_text
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Interactive Architecture Genealogy',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Interactive genealogy tree",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002) ],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        interactive_path = self.output_dir / 'interactive_genealogy.html'
        fig.write_html(interactive_path)
        print(f"Interactive genealogy saved to {interactive_path}")
    
    def plot_operation_frequency_heatmap(self,
                                       search_history: Dict[str, Any],
                                       save_path: Optional[str] = None):
        """
        Create heatmap showing operation selection frequency across generations.
        
        Args:
            search_history: History containing population genomes
            save_path: Path to save visualization
        """
        # Operation names (from operations.py)
        op_names = [
            'skip_connect', 'avg_pool_3x3', 'max_pool_3x3', 'sep_conv_3x3',
            'sep_conv_5x5', 'sep_conv_7x7', 'dil_sep_conv_3x3', 'dil_sep_conv_5x5',
            'group_conv_2x2', 'group_conv_4x4', 'inv_bottleneck_2', 'inv_bottleneck_4',
            'inv_bottleneck_6', 'se_inv_bottleneck'
        ]
        
        generations = search_history['generation']
        populations = search_history['population']
        
        # Count operation frequencies per generation
        frequency_matrix = np.zeros((len(op_names), len(generations)))
        
        for gen_idx, population in enumerate(populations):
            op_counts = {op: 0 for op in range(len(op_names))}
            total_ops = 0
            
            for individual in population:
                # Count operations in normal cell
                for gene in individual['normal_cell']:
                    op_idx = gene[1]
                    if op_idx < len(op_names):
                        op_counts[op_idx] += 1
                        total_ops += 1
                
                # Count operations in reduction cell
                for gene in individual['reduction_cell']:
                    op_idx = gene[1]
                    if op_idx < len(op_names):
                        op_counts[op_idx] += 1
                        total_ops += 1
            
            # Normalize frequencies
            for op_idx, count in op_counts.items():
                if total_ops > 0:
                    frequency_matrix[op_idx, gen_idx] = count / total_ops
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        im = plt.imshow(frequency_matrix, aspect='auto', cmap='YlOrRd',
                       interpolation='nearest')
        
        plt.colorbar(im, label='Operation Frequency')
        plt.xlabel('Generation')
        plt.ylabel('Operation Type')
        plt.title('Operation Selection Frequency Across Generations')
        
        # Set ticks
        plt.xticks(range(len(generations)), 
                  [f'Gen {g}' for g in generations], rotation=45)
        plt.yticks(range(len(op_names)), op_names, rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            heatmap_path = self.output_dir / 'operation_frequency_heatmap.png'
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {heatmap_path}")
        
        plt.show()
        
        # Create interactive heatmap
        self._create_interactive_heatmap(frequency_matrix, op_names, generations)
    
    def _create_interactive_heatmap(self, frequency_matrix: np.ndarray,
                                  op_names: List[str], generations: List[int]):
        """Create interactive heatmap with Plotly."""
        fig = go.Figure(data=go.Heatmap(
            z=frequency_matrix,
            x=[f'Gen {g}' for g in generations],
            y=op_names,
            colorscale='YlOrRd',
            hoverongaps=False))
        
        fig.update_layout(
            title='Interactive Operation Frequency Heatmap',
            xaxis_title='Generation',
            yaxis_title='Operation Type',
            width=800,
            height=600
        )
        
        interactive_path = self.output_dir / 'interactive_heatmap.html'
        fig.write_html(interactive_path)
        print(f"Interactive heatmap saved to {interactive_path}")
    
    def plot_search_space_coverage(self,
                                 architectures: List[Dict],
                                 performances: List[np.ndarray],
                                 save_path: Optional[str] = None):
        """
        Visualize search space coverage using t-SNE/UMAP projection.
        
        Args:
            architectures: List of explored architectures
            performances: Corresponding performance metrics
            save_path: Path to save visualization
        """
        # Convert architectures to feature vectors
        from ..search_space.encoding import ArchitectureEncoder
        encoder = ArchitectureEncoder()
        
        features = []
        for arch in architectures:
            int_encoding = encoder.genome_to_integer(arch)
            features.append(int_encoding)
        
        features = np.array(features)
        performances = np.array(performances)
        
        # Apply dimensionality reduction
        methods = {
            't-SNE': TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1)),
            'UMAP': UMAP(n_components=2, random_state=42)
        }
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for idx, (method_name, reducer) in enumerate(methods.items()):
            try:
                # Reduce dimensionality
                embeddings = reducer.fit_transform(features)
                
                # Plot with color coding by accuracy
                scatter = axes[idx].scatter(
                    embeddings[:, 0], embeddings[:, 1],
                    c=performances[:, 0],  # Accuracy
                    cmap='viridis', alpha=0.7, s=50)
                
                axes[idx].set_title(f'{method_name} Projection of Search Space')
                axes[idx].set_xlabel('Component 1')
                axes[idx].set_ylabel('Component 2')
                
                # Add colorbar
                plt.colorbar(scatter, ax=axes[idx], label='Accuracy (%)')
                
            except Exception as e:
                print(f"Error with {method_name}: {e}")
                axes[idx].text(0.5, 0.5, f'{method_name} failed', 
                              ha='center', va='center', transform=axes[idx].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            coverage_path = self.output_dir / 'search_space_coverage.png'
            plt.savefig(coverage_path, dpi=300, bbox_inches='tight')
            print(f"Search space coverage plot saved to {coverage_path}")
        
        plt.show()
    
    def plot_proxy_correlation_analysis(self,
                                      proxy_data: Dict[str, np.ndarray],
                                      true_performances: np.ndarray,
                                      save_path: Optional[str] = None):
        """
        Analyze correlation between proxy scores and true performance.
        
        Args:
            proxy_data: Dictionary with proxy names as keys and scores as values
            true_performances: Array of true accuracy values
            save_path: Path to save visualization
        """
        import scipy.stats as stats
        
        proxy_names = list(proxy_data.keys())
        n_proxies = len(proxy_names)
        
        # Calculate correlations
        correlations = []
        p_values = []
        
        for proxy_name in proxy_names:
            proxy_scores = proxy_data[proxy_name]
            corr, p_value = stats.spearmanr(proxy_scores, true_performances)
            correlations.append(corr)
            p_values.append(p_value)
        
        # Create correlation plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        # 1. Correlation bar plot
        y_pos = np.arange(len(proxy_names))
        bars = axes[0].barh(y_pos, correlations, color='skyblue')
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(proxy_names)
        axes[0].set_xlabel('Spearman Correlation')
        axes[0].set_title('Proxy Correlation with True Accuracy')
        
        # Add correlation values on bars
        for bar, corr in zip(bars, correlations):
            axes[0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{corr:.3f}', ha='left', va='center')
        
        # 2. Scatter plots for top proxies
        top_indices = np.argsort(np.abs(correlations))[-4:]  # Top 4 proxies
        
        for i, proxy_idx in enumerate(top_indices):
            if i < 4:  # Ensure we don't exceed subplot count
                proxy_name = proxy_names[proxy_idx]
                proxy_scores = proxy_data[proxy_name]
                
                axes[i+1].scatter(proxy_scores, true_performances, alpha=0.6)
                axes[i+1].set_xlabel(f'{proxy_name} Score')
                axes[i+1].set_ylabel('True Accuracy')
                axes[i+1].set_title(f'{proxy_name} vs True Accuracy\n'
                                  f'Correlation: {correlations[proxy_idx]:.3f}')
                
                # Add trend line
                z = np.polyfit(proxy_scores, true_performances, 1)
                p = np.poly1d(z)
                axes[i+1].plot(proxy_scores, p(proxy_scores), "r--", alpha=0.8)
        
        plt.tight_layout()
        
        if save_path:
            correlation_path = self.output_dir / 'proxy_correlation_analysis.png'
            plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
            print(f"Proxy correlation analysis saved to {correlation_path}")
        
        plt.show()
        
        # Print correlation table
        print("\nProxy Correlation Analysis:")
        print("=" * 50)
        for name, corr, p_val in zip(proxy_names, correlations, p_values):
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{name:20} | {corr:7.3f} | p-value: {p_val:.4f} {significance}")
    
    def create_comprehensive_dashboard(self, search_history: Dict[str, Any],
                                     genealogy_data: Dict[str, Any],
                                     proxy_data: Dict[str, np.ndarray]):
        """Create a comprehensive dashboard with all visualizations."""
        print("Generating Comprehensive NAS Dashboard...")
        
        # 1. Pareto Front Evolution
        self.plot_pareto_front_evolution(search_history, 
                                       create_animation=False)
        
        # 2. Architecture Genealogy
        self.plot_architecture_genealogy(genealogy_data)
        
        # 3. Operation Frequency Heatmap
        self.plot_operation_frequency_heatmap(search_history)
        
        # 4. Search Space Coverage (if we have enough architectures)
        if 'population' in search_history and len(search_history['population']) > 10:
            # Sample some architectures for visualization
            sample_architectures = []
            sample_performances = []
            
            for gen_idx, population in enumerate(search_history['population']):
                if gen_idx % 5 == 0:  # Sample every 5th generation
                    sample_architectures.extend(population[:5])  # First 5 architectures
                    if gen_idx < len(search_history['fitness']):
                        sample_performances.extend(search_history['fitness'][gen_idx][:5])
            
            if len(sample_architectures) > 10:
                self.plot_search_space_coverage(sample_architectures, 
                                              np.array(sample_performances))
        
        # 5. Proxy Correlation Analysis
        if proxy_data and 'accuracy' in proxy_data:
            # Use final generation performances as true values
            final_fitness = search_history['fitness'][-1]
            true_accuracies = final_fitness[:, 0]
            self.plot_proxy_correlation_analysis(proxy_data, true_accuracies)
        
        print("Dashboard generation completed!")
        
        # Generate summary report
        self._generate_dashboard_summary(search_history)

    def _generate_dashboard_summary(self, search_history: Dict[str, Any]):
        """Generate text summary of search results."""
        summary_path = self.output_dir / 'search_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("EFFICIENT NAS SEARCH SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic statistics
            total_generations = len(search_history['generation'])
            f.write(f"Total Generations: {total_generations}\n")
            
            # Final Pareto front
            final_pareto = search_history['pareto_front'][-1]
            if final_pareto and len(final_pareto['fitness']) > 0:
                pf_size = len(final_pareto['fitness'])
                f.write(f"Final Pareto Front Size: {pf_size}\n")
                
                # Performance ranges
                acc_range = (final_pareto['fitness'][:, 0].min(), 
                           final_pareto['fitness'][:, 0].max())
                lat_range = (final_pareto['fitness'][:, 1].min(),
                           final_pareto['fitness'][:, 1].max())
                param_range = (final_pareto['fitness'][:, 2].min(),
                             final_pareto['fitness'][:, 2].max())
                
                f.write(f"Accuracy Range: {acc_range[0]:.2f}% - {acc_range[1]:.2f}%\n")
                f.write(f"Latency Range: {lat_range[0]:.2f}ms - {lat_range[1]:.2f}ms\n")
                f.write(f"Parameter Range: {param_range[0]:.2f}M - {param_range[1]:.2f}M\n\n")
            
            # Search efficiency
            total_architectures = sum(len(pop) for pop in search_history['population'])
            f.write(f"Total Architectures Evaluated: {total_architectures}\n")
            
            # Convergence analysis
            if len(search_history['fitness']) > 1:
                initial_best = search_history['fitness'][0][:, 0].max()
                final_best = search_history['fitness'][-1][:, 0].max()
                improvement = final_best - initial_best
                f.write(f"Best Accuracy Improvement: {improvement:.2f}%\n")
        
        print(f"Search summary saved to {summary_path}")

# Utility function for real-time monitoring
class RealTimeMonitor:
    """Real-time monitoring of NAS search progress."""
    
    def __init__(self, update_interval: int = 5):
        self.update_interval = update_interval
        self.metrics_history = {
            'generation': [],
            'best_accuracy': [],
            'average_accuracy': [],
            'pareto_size': [],
            'diversity': []
        }
    
    def update(self, generation: int, population: List[Dict], 
              fitness: np.ndarray, pareto_front: Dict):
        """Update monitoring metrics."""
        self.metrics_history['generation'].append(generation)
        self.metrics_history['best_accuracy'].append(fitness[:, 0].max())
        self.metrics_history['average_accuracy'].append(fitness[:, 0].mean())
        self.metrics_history['pareto_size'].append(
            len(pareto_front['fitness']) if pareto_front else 0)
        
        # Calculate diversity (simplified)
        diversity = self._calculate_diversity(population)
        self.metrics_history['diversity'].append(diversity)
    
    def _calculate_diversity(self, population: List[Dict]) -> float:
        """Calculate population diversity."""
        if len(population) < 2:
            return 0.0
        
        # Simple diversity metric based on unique architectures
        unique_archs = len(set(str(arch) for arch in population))
        return unique_archs / len(population)
    
    def plot_progress(self):
        """Plot real-time search progress."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()
        
        metrics = ['best_accuracy', 'average_accuracy', 'pareto_size', 'diversity']
        titles = ['Best Accuracy', 'Average Accuracy', 'Pareto Front Size', 'Population Diversity']
        ylabels = ['Accuracy (%)', 'Accuracy (%)', 'Size', 'Diversity']
        
        for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
            axes[i].plot(self.metrics_history['generation'], 
                        self.metrics_history[metric], 
                        marker='o', linewidth=2)
            axes[i].set_xlabel('Generation')
            axes[i].set_ylabel(ylabel)
            axes[i].set_title(title)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()