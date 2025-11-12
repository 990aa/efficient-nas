import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn

class ArchitectureEncoder:
    """Encodes/decodes neural architectures to/from integer genomes."""
    
    def __init__(self, steps: int = 5, num_ops: int = 13):
        """
        Args:
            steps: Number of intermediate nodes in each cell
            num_ops: Number of available operations
        """
        self.steps = steps
        self.num_ops = num_ops
        self.genes_per_cell = steps * 2  # 2 connections per node
        
    def create_random_genome(self, num_cells: int = 4, init_channels: int = 32, 
                           channel_multiplier: float = 2.0) -> Dict[str, Any]:
        """Create a random architecture genome."""
        # Normal cell genome: each gene is (input_node, operation, weight)
        normal_genome = []
        for i in range(self.genes_per_cell):
            input_node = np.random.randint(0, 2 + i // 2)  # Only previous nodes
            operation = np.random.randint(0, self.num_ops)
            weight = np.random.randint(0, 3)  # Optional connection strength
            normal_genome.append((input_node, operation, weight))
            
        # Reduction cell genome
        reduce_genome = []
        for i in range(self.genes_per_cell):
            input_node = np.random.randint(0, 2 + i // 2)
            operation = np.random.randint(0, self.num_ops)
            weight = np.random.randint(0, 3)
            reduce_genome.append((input_node, operation, weight))
            
        # Hyperparameters
        hyperparams = {
            'num_cells': np.random.randint(3, 7),  # 3-6 cells
            'init_channels': np.random.randint(16, 49),  # 16-48
            'channel_multiplier': np.random.uniform(1.5, 3.0),
            'steps': self.steps
        }
        
        return {
            'normal_cell': normal_genome,
            'reduction_cell': reduce_genome,
            'hyperparams': hyperparams
        }
    
    def genome_to_integer(self, genome: Dict[str, Any]) -> List[int]:
        """Convert genome to flat integer list for evolutionary algorithms."""
        integer_encoding = []
        
        # Encode normal cell
        for gene in genome['normal_cell']:
            integer_encoding.extend(gene[:2])  # input_node, operation
        
        # Encode reduction cell
        for gene in genome['reduction_cell']:
            integer_encoding.extend(gene[:2])
            
        # Encode hyperparameters
        hyper = genome['hyperparams']
        integer_encoding.extend([
            hyper['num_cells'],
            hyper['init_channels'],
            int(hyper['channel_multiplier'] * 10)  # Scale for integer encoding
        ])
        
        return integer_encoding
    
    def integer_to_genome(self, integer_list: List[int]) -> Dict[str, Any]:
        """Convert integer list back to genome dictionary."""
        idx = 0
        
        # Decode normal cell
        normal_genome = []
        for i in range(self.genes_per_cell):
            input_node = integer_list[idx]
            idx += 1
            operation = integer_list[idx]
            idx += 1
            normal_genome.append((input_node, operation, 1))  # Default weight
        
        # Decode reduction cell
        reduce_genome = []
        for i in range(self.genes_per_cell):
            input_node = integer_list[idx]
            idx += 1
            operation = integer_list[idx]
            idx += 1
            reduce_genome.append((input_node, operation, 1))
            
        # Decode hyperparameters
        hyperparams = {
            'num_cells': integer_list[idx],
            'init_channels': integer_list[idx + 1],
            'channel_multiplier': integer_list[idx + 2] / 10.0,  # Scale back
            'steps': self.steps
        }
        
        return {
            'normal_cell': normal_genome,
            'reduction_cell': reduce_genome,
            'hyperparams': hyperparams
        }
    
    def mutate_genome(self, genome: Dict[str, Any], mutation_rate: float = 0.1) -> Dict[str, Any]:
        """Apply mutations to a genome."""
        mutated = genome.copy()
        integer_genome = self.genome_to_integer(genome)
        
        # Apply random mutations
        for i in range(len(integer_genome)):
            if np.random.random() < mutation_rate:
                if i < self.genes_per_cell * 4:  # Cell genes
                    # Mutation for cell connections and operations
                    if i % 2 == 0:  # Input node
                        max_input = 2 + (i // 2) // 2
                        integer_genome[i] = np.random.randint(0, max_input)
                    else:  # Operation
                        integer_genome[i] = np.random.randint(0, self.num_ops)
                else:  # Hyperparameters
                    if i == len(integer_genome) - 3:  # num_cells
                        integer_genome[i] = np.random.randint(3, 7)
                    elif i == len(integer_genome) - 2:  # init_channels
                        integer_genome[i] = np.random.randint(16, 49)
                    else:  # channel_multiplier
                        integer_genome[i] = np.random.randint(15, 31)
        
        return self.integer_to_genome(integer_genome)
    
    def crossover_genomes(self, parent1: Dict[str, Any], 
                         parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform crossover between two parent genomes."""
        int1 = self.genome_to_integer(parent1)
        int2 = self.genome_to_integer(parent2)
        
        # Single-point crossover
        crossover_point = np.random.randint(1, len(int1) - 1)
        child_int = int1[:crossover_point] + int2[crossover_point:]
        
        return self.integer_to_genome(child_int)

class ConstraintAwareInitializer:
    """Early Exit Population Initialization (EEPI) for constraint satisfaction."""
    
    def __init__(self, encoder: ArchitectureEncoder, param_budget: float = 2e6):
        self.encoder = encoder
        self.param_budget = param_budget
        
    def estimate_parameters(self, genome: Dict[str, Any]) -> float:
        """Quick parameter estimation without building full model."""
        hyper = genome['hyperparams']
        C = hyper['init_channels']
        L = hyper['num_cells']
        multiplier = 4  # Conservative multiplier
        
        # Stem parameters
        stem_params = 3 * C * 3 * 3  # First conv
        
        # Cell parameters estimation (simplified)
        cell_params = 0
        for i in range(L):
            if i in [L // 3, 2 * L // 3]:
                C *= hyper['channel_multiplier']
            
            # Estimate based on connections and operations
            cell_params += C * C * 3 * 3 * self.encoder.steps * 2
        
        # Classifier
        classifier_params = C * multiplier * 10  # Assume 10 classes
        
        total_est = stem_params + cell_params + classifier_params
        return total_est
    
    def initialize_population(self, population_size: int, 
                            max_attempts: int = 1000) -> List[Dict[str, Any]]:
        """Initialize population with constraint satisfaction."""
        population = []
        attempts = 0
        
        while len(population) < population_size and attempts < max_attempts:
            genome = self.encoder.create_random_genome()
            param_estimate = self.estimate_parameters(genome)
            
            if param_estimate <= self.param_budget:
                population.append(genome)
            
            attempts += 1
        
        if len(population) < population_size:
            print(f"Warning: Only generated {len(population)} valid architectures "
                  f"out of {population_size} after {max_attempts} attempts")
        
        return population

# Utility function to create model from genome
def create_model_from_genome(genome: Dict[str, Any], num_classes: int = 10) -> nn.Module:
    """Create a PyTorch model from a genome encoding."""
    from .cell_structure import Network
    
    hyper = genome['hyperparams']
    
    model = Network(
        C=hyper['init_channels'],
        num_classes=num_classes,
        layers=hyper['num_cells'],
        steps=hyper['steps'],
        multiplier=4,  # Fixed multiplier
        stem_multiplier=3,  # Fixed stem multiplier
        normal_genome=genome['normal_cell'],
        reduce_genome=genome['reduction_cell']
    )
    
    return model