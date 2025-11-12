import numpy as np
from typing import Tuple, Dict
import copy

class GeneticOperators:
    """Genetic operators for architecture evolution."""
    
    def __init__(self, encoder, param_budget: float = 2e6):
        self.encoder = encoder
        self.param_budget = param_budget
        
    def crossover(self, parent1: Dict, parent2: Dict, 
                 method: str = 'uniform') -> Tuple[Dict, Dict]:
        """
        Crossover between two parent genomes.
        
        Args:
            parent1, parent2: Parent genomes
            method: 'uniform', 'single_point', or 'cell_level'
            
        Returns:
            Two child genomes
        """
        if method == 'uniform':
            return self._uniform_crossover(parent1, parent2)
        elif method == 'single_point':
            return self._single_point_crossover(parent1, parent2)
        elif method == 'cell_level':
            return self._cell_level_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover method: {method}")
    
    def _uniform_crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Uniform crossover - swap random genes with 50% probability."""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # Crossover normal cell genes
        for i in range(len(child1['normal_cell'])):
            if np.random.random() < 0.5:
                child1['normal_cell'][i], child2['normal_cell'][i] = \
                    child2['normal_cell'][i], child1['normal_cell'][i]
        
        # Crossover reduction cell genes
        for i in range(len(child1['reduction_cell'])):
            if np.random.random() < 0.5:
                child1['reduction_cell'][i], child2['reduction_cell'][i] = \
                    child2['reduction_cell'][i], child1['reduction_cell'][i]
        
        # Crossover hyperparameters
        if np.random.random() < 0.5:
            child1['hyperparams']['num_cells'], child2['hyperparams']['num_cells'] = \
                child2['hyperparams']['num_cells'], child1['hyperparams']['num_cells']
        
        if np.random.random() < 0.5:
            child1['hyperparams']['init_channels'], child2['hyperparams']['init_channels'] = \
                child2['hyperparams']['init_channels'], child1['hyperparams']['init_channels']
        
        if np.random.random() < 0.5:
            child1['hyperparams']['channel_multiplier'], child2['hyperparams']['channel_multiplier'] = \
                child2['hyperparams']['channel_multiplier'], child1['hyperparams']['channel_multiplier']
        
        # Repair if needed
        child1 = self.repair_genome(child1)
        child2 = self.repair_genome(child2)
        
        return child1, child2
    
    def _single_point_crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Single-point crossover at a random position."""
        # Convert to integer representation for easier crossover
        int1 = self.encoder.genome_to_integer(parent1)
        int2 = self.encoder.genome_to_integer(parent2)
        
        crossover_point = np.random.randint(1, len(int1) - 1)
        
        child1_int = int1[:crossover_point] + int2[crossover_point:]
        child2_int = int2[:crossover_point] + int1[crossover_point:]
        
        child1 = self.encoder.integer_to_genome(child1_int)
        child2 = self.encoder.integer_to_genome(child2_int)
        
        # Repair if needed
        child1 = self.repair_genome(child1)
        child2 = self.repair_genome(child2)
        
        return child1, child2
    
    def _cell_level_crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Exchange entire cells between parents."""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # Randomly decide which cell type to swap
        if np.random.random() < 0.5:
            # Swap normal cells
            child1['normal_cell'], child2['normal_cell'] = \
                child2['normal_cell'], child1['normal_cell']
        else:
            # Swap reduction cells
            child1['reduction_cell'], child2['reduction_cell'] = \
                child2['reduction_cell'], child1['reduction_cell']
        
        # Repair if needed
        child1 = self.repair_genome(child1)
        child2 = self.repair_genome(child2)
        
        return child1, child2
    
    def mutate(self, individual: Dict, mutation_rate: float = 0.3) -> Dict:
        """
        Mutate an individual genome with various mutation operators.
        
        Args:
            individual: Genome to mutate
            mutation_rate: Base mutation rate
            
        Returns:
            Mutated genome
        """
        mutated = copy.deepcopy(individual)
        
        # Operation replacement (20% per gene)
        for i in range(len(mutated['normal_cell'])):
            if np.random.random() < 0.2 * mutation_rate:
                # Mutate operation
                new_op = np.random.randint(0, self.encoder.num_ops)
                mutated['normal_cell'][i] = (mutated['normal_cell'][i][0], new_op, 1)
        
        for i in range(len(mutated['reduction_cell'])):
            if np.random.random() < 0.2 * mutation_rate:
                # Mutate operation
                new_op = np.random.randint(0, self.encoder.num_ops)
                mutated['reduction_cell'][i] = (mutated['reduction_cell'][i][0], new_op, 1)
        
        # Connection rewiring (15% probability)
        if np.random.random() < 0.15 * mutation_rate:
            mutated = self._rewire_connections(mutated)
        
        # Hyperparameter mutation with Gaussian noise
        mutated = self._mutate_hyperparameters(mutated, mutation_rate)
        
        # Repair if needed
        mutated = self.repair_genome(mutated)
        
        return mutated
    
    def _rewire_connections(self, individual: Dict) -> Dict:
        """Randomly rewire connections in the genome."""
        mutated = copy.deepcopy(individual)
        
        # Rewire random connection in normal cell
        if len(mutated['normal_cell']) > 0:
            idx = np.random.randint(0, len(mutated['normal_cell']))
            node_idx = idx // 2
            max_input = 2 + node_idx  # Only previous nodes are allowed
            new_input = np.random.randint(0, max_input)
            mutated['normal_cell'][idx] = (new_input, mutated['normal_cell'][idx][1], 1)
        
        # Rewire random connection in reduction cell
        if len(mutated['reduction_cell']) > 0:
            idx = np.random.randint(0, len(mutated['reduction_cell']))
            node_idx = idx // 2
            max_input = 2 + node_idx
            new_input = np.random.randint(0, max_input)
            mutated['reduction_cell'][idx] = (new_input, mutated['reduction_cell'][idx][1], 1)
        
        return mutated
    
    def _mutate_hyperparameters(self, individual: Dict, mutation_rate: float) -> Dict:
        """Mutate hyperparameters with Gaussian noise."""
        mutated = copy.deepcopy(individual)
        hyper = mutated['hyperparams']
        
        # Number of cells (discrete)
        if np.random.random() < 0.3 * mutation_rate:
            hyper['num_cells'] = np.random.randint(3, 7)
        
        # Initial channels (discrete)
        if np.random.random() < 0.3 * mutation_rate:
            hyper['init_channels'] = np.random.randint(16, 49)
        
        # Channel multiplier (continuous with Gaussian noise)
        if np.random.random() < 0.3 * mutation_rate:
            noise = np.random.normal(0, 0.1)
            new_multiplier = hyper['channel_multiplier'] + noise
            hyper['channel_multiplier'] = np.clip(new_multiplier, 1.5, 3.0)
        
        return mutated
    
    def repair_genome(self, individual: Dict) -> Dict:
        """
        Repair genome to ensure validity and constraint satisfaction.
        
        - Ensure no disconnected nodes
        - Satisfy parameter budget
        - Valid connection patterns
        """
        repaired = copy.deepcopy(individual)
        
        # Check parameter budget
        from ..search_space.encoding import ConstraintAwareInitializer
        initializer = ConstraintAwareInitializer(self.encoder, self.param_budget)
        param_estimate = initializer.estimate_parameters(repaired)
        
        if param_estimate > self.param_budget:
            # Reduce model complexity
            repaired['hyperparams']['init_channels'] = max(16, 
                repaired['hyperparams']['init_channels'] // 1.2)
            repaired['hyperparams']['num_cells'] = max(3,
                repaired['hyperparams']['num_cells'] - 1)
        
        # Ensure valid connections
        repaired = self._ensure_connected_graph(repaired)
        
        return repaired
    
    def _ensure_connected_graph(self, individual: Dict) -> Dict:
        """Ensure the architecture forms a connected computational graph."""
        # For now, basic check - all operations should be valid
        # More sophisticated connectivity checks can be added
        return individual

# Convenience functions
def crossover_genomes(parent1: Dict, parent2: Dict, 
                     method: str = 'uniform') -> Tuple[Dict, Dict]:
    """Convenience function for crossover."""
    from ..search_space.encoding import ArchitectureEncoder
    encoder = ArchitectureEncoder()
    operators = GeneticOperators(encoder)
    return operators.crossover(parent1, parent2, method)

def mutate_genome(individual: Dict, mutation_rate: float = 0.3) -> Dict:
    """Convenience function for mutation."""
    from ..search_space.encoding import ArchitectureEncoder
    encoder = ArchitectureEncoder()
    operators = GeneticOperators(encoder)
    return operators.mutate(individual, mutation_rate)