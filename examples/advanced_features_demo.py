# examples/advanced_features_demo.py
#!/usr/bin/env python3
"""
Demonstration of Advanced Features:
1. Performance Predictor (GNN Surrogate)
2. Transfer Learning with Network Morphism  
3. Comprehensive Visualization Suite
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
from src.predictor.performance_predictor import PerformancePredictor
from src.evolution.network_morphism import NetworkMorphism
from src.utils.visualization import NASVisualizer, RealTimeMonitor
from src.search_space.encoding import ArchitectureEncoder, ConstraintAwareInitializer

def demo_performance_predictor():
    """Demonstrate GNN performance predictor."""
    print("=== Performance Predictor Demo ===")
    
    # Generate training data
    encoder = ArchitectureEncoder()
    initializer = ConstraintAwareInitializer(encoder)
    
    training_genomes = initializer.initialize_population(100)  # 100 architectures
    training_accuracies = np.random.uniform(70, 95, 100)  # Mock accuracies
    training_latencies = np.random.uniform(5, 50, 100)    # Mock latencies
    
    # Train predictor
    predictor = PerformancePredictor(device='cpu')
    history = predictor.train_predictor(
        training_genomes, training_accuracies, training_latencies, epochs=50)
    
    # Test prediction
    test_genomes = initializer.initialize_population(10)
    predictions = predictor.predict_performance(test_genomes, with_uncertainty=True)
    
    print("Prediction results:")
    for i, genome in enumerate(test_genomes[:3]):  # Show first 3
        print(f"Architecture {i+1}: "
              f"Predicted Accuracy = {predictions['accuracy_mean'][i]:.2f}% "
              f"(±{predictions['accuracy_uncertainty'][i]:.2f})")
    
    return predictor

def demo_network_morphism():
    """Demonstrate weight inheritance through network morphism."""
    print("\n=== Network Morphism Demo ===")
    
    morphism = NetworkMorphism()
    encoder = ArchitectureEncoder()
    
    # Create parent architecture and mock trained weights
    parent_genome = encoder.create_random_genome()
    print("Parent genome created")
    
    # Create similar child genome (one operation different)
    child_genome = encoder.mutate_genome(parent_genome, mutation_rate=0.1)
    print("Child genome created through mutation")
    
    # Check if morphism is possible
    can_morph = morphism.can_morph(parent_genome, child_genome)
    print(f"Network morphism possible: {can_morph}")
    
    if can_morph:
        # In real usage, we would have actual trained models
        print("Weight inheritance would be applied here")
    
    return morphism

def demo_visualization_suite():
    """Demonstrate comprehensive visualization capabilities."""
    print("\n=== Visualization Suite Demo ===")
    
    visualizer = NASVisualizer()
    
    # Generate mock search history for demonstration
    mock_history = generate_mock_search_history()
    
    # Generate mock genealogy data
    mock_genealogy = {
        'arch_1': ['arch_0'],
        'arch_2': ['arch_0', 'arch_1'],
        'arch_3': ['arch_1'],
        'arch_4': ['arch_2', 'arch_3'],
        'arch_5': ['arch_3']
    }
    
    # Generate mock proxy data
    mock_proxy_data = {
        'synflow': np.random.uniform(0, 100, 50),
        'naswot': np.random.uniform(0, 50, 50),
        'grad_norm': np.random.uniform(0, 10, 50),
        'zen_score': np.random.uniform(0, 100, 50)
    }
    
    # Create comprehensive dashboard
    visualizer.create_comprehensive_dashboard(
        mock_history, mock_genealogy, mock_proxy_data)
    
    print("Visualizations generated in ./visualizations/ directory")

def generate_mock_search_history() -> dict:
    """Generate mock search history for demonstration."""
    generations = list(range(0, 50, 5))
    history = {
        'generation': generations,
        'fitness': [],
        'population': [],
        'pareto_front': []
    }
    
    encoder = ArchitectureEncoder()
    initializer = ConstraintAwareInitializer(encoder)
    
    for gen in generations:
        # Mock population
        population = initializer.initialize_population(20)
        
        # Mock fitness (improving over generations)
        base_accuracy = 70 + gen * 0.5
        accuracy = np.random.normal(base_accuracy, 5, 20)
        latency = np.random.uniform(5, 50, 20)
        params = np.random.uniform(0.5, 3.0, 20)
        
        fitness = np.column_stack([accuracy, latency, params])
        
        # Mock Pareto front (top 5 architectures)
        pareto_indices = np.argsort(accuracy)[-5:]
        pareto_front = {
            'individuals': [population[i] for i in pareto_indices],
            'fitness': fitness[pareto_indices]
        }
        
        history['fitness'].append(fitness)
        history['population'].append(population)
        history['pareto_front'].append(pareto_front)
    
    return history

def demo_integrated_advanced_search():
    """Demonstrate integrated advanced features in NAS search."""
    print("\n=== Integrated Advanced Search Demo ===")
    
    # Initialize components
    predictor = PerformancePredictor()
    morphism = NetworkMorphism()
    visualizer = NASVisualizer()
    monitor = RealTimeMonitor()
    
    # In a real scenario, we would:
    # 1. Train predictor on initial architectures
    # 2. Use predictor-enhanced search for early generations
    # 3. Apply network morphism for weight inheritance
    # 4. Use visualizer for real-time monitoring
    
    print("Advanced features integrated and ready for NAS search!")
    print("Components available:")
    print("- Performance Predictor (GNN Surrogate)")
    print("- Network Morphism (Weight Inheritance)") 
    print("- Comprehensive Visualization Suite")
    print("- Real-time Progress Monitoring")

if __name__ == "__main__":
    # Run demonstrations
    predictor = demo_performance_predictor()
    morphism = demo_network_morphism()
    demo_visualization_suite()
    demo_integrated_advanced_search()
