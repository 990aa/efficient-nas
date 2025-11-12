import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
import copy
from ..search_space.encoding import create_model_from_genome
from ..search_space.operations import *  # noqa: F403


class NetworkMorphism:
    """Weight inheritance through network morphism for faster evaluation."""

    def __init__(self):
        self.weight_cache = {}  # Cache parent weights
        self.morphism_operations = {}

    def register_parent(self, genome: Dict, model: nn.Module):
        """Register a parent architecture and its trained weights."""
        genome_key = self._genome_to_key(genome)
        self.weight_cache[genome_key] = {
            "state_dict": copy.deepcopy(model.state_dict()),
            "genome": copy.deepcopy(genome),
        }

    def can_morph(self, parent_genome: Dict, child_genome: Dict) -> bool:
        """
        Check if child can inherit weights from parent through morphism.

        Conditions for morphism:
        1. Same basic structure (number of cells, channels)
        2. Differences only in specific operations
        3. Compatible tensor shapes
        """
        # Check hyperparameter compatibility
        parent_hyper = parent_genome["hyperparams"]
        child_hyper = child_genome["hyperparams"]

        if (
            parent_hyper["num_cells"] != child_hyper["num_cells"]
            or parent_hyper["init_channels"] != child_hyper["init_channels"]
        ):
            return False

        # Check if differences are morphism-compatible
        differences = self._find_genome_differences(parent_genome, child_genome)

        # Allow morphism if only one operation changed
        return len(differences) == 1

    def morph_weights(
        self,
        parent_genome: Dict,
        child_genome: Dict,
        parent_model: nn.Module,
        child_model: nn.Module,
    ) -> nn.Module:
        """
        Transfer weights from parent to child through network morphism.

        Args:
            parent_genome: Parent architecture
            child_genome: Child architecture
            parent_model: Parent model with trained weights
            child_model: Child model to initialize

        Returns:
            Child model with morphed weights
        """
        parent_state_dict = parent_model.state_dict()
        child_state_dict = child_model.state_dict()

        differences = self._find_genome_differences(parent_genome, child_genome)

        if not differences:
            # No differences, direct weight transfer
            child_model.load_state_dict(parent_state_dict)
            return child_model

        # Handle each type of difference
        for diff in differences:
            cell_type, node_idx, connection_idx = diff

            if cell_type == "normal":
                parent_op = parent_genome["normal_cell"][connection_idx][1]
                child_op = child_genome["normal_cell"][connection_idx][1]
            else:
                parent_op = parent_genome["reduction_cell"][connection_idx][1]
                child_op = child_genome["reduction_cell"][connection_idx][1]

            # Perform operation-specific morphism
            child_state_dict = self._morph_operation(
                parent_state_dict,
                child_state_dict,
                cell_type,
                node_idx,
                connection_idx,
                parent_op,
                child_op,
            )

        child_model.load_state_dict(child_state_dict)
        return child_model

    def _find_genome_differences(self, parent: Dict, child: Dict) -> List[Tuple]:
        """Find differences between two genomes."""
        differences = []

        # Compare normal cells
        for i, (parent_gene, child_gene) in enumerate(
            zip(parent["normal_cell"], child["normal_cell"])
        ):
            if parent_gene[1] != child_gene[1]:  # Operation difference
                node_idx = i // 2
                connection_idx = i
                differences.append(("normal", node_idx, connection_idx))

        # Compare reduction cells
        for i, (parent_gene, child_gene) in enumerate(
            zip(parent["reduction_cell"], child["reduction_cell"])
        ):
            if parent_gene[1] != child_gene[1]:  # Operation difference
                node_idx = i // 2
                connection_idx = i
                differences.append(("reduction", node_idx, connection_idx))

        return differences

    def _morph_operation(
        self,
        parent_state_dict: Dict,
        child_state_dict: Dict,
        cell_type: str,
        node_idx: int,
        connection_idx: int,
        parent_op: int,
        child_op: int,
    ) -> Dict:
        """
        Morph weights for a specific operation change.

        Implements various morphism strategies based on operation types.
        """
        # Operation names mapping (from operations.py)
        op_names = [
            "skip_connect",
            "avg_pool_3x3",
            "max_pool_3x3",
            "sep_conv_3x3",
            "sep_conv_5x5",
            "sep_conv_7x7",
            "dil_sep_conv_3x3",
            "dil_sep_conv_5x5",
            "group_conv_2x2",
            "group_conv_4x4",
            "inv_bottleneck_2",
            "inv_bottleneck_4",
            "inv_bottleneck_6",
            "se_inv_bottleneck",
        ]

        parent_op_name = op_names[parent_op]
        child_op_name = op_names[child_op]

        # Construct parameter name patterns
        base_pattern = f"cells.{node_idx}.ops.{connection_idx % 2}"

        # Different morphism strategies based on operation pairs
        if self._is_identity_morphism(parent_op_name, child_op_name):
            # Operations with similar structure, direct transfer
            child_state_dict = self._transfer_identical_ops(
                parent_state_dict, child_state_dict, base_pattern
            )

        elif self._is_conv_morphism(parent_op_name, child_op_name):
            # Convolution operation morphism
            child_state_dict = self._morph_conv_operations(
                parent_state_dict,
                child_state_dict,
                base_pattern,
                parent_op_name,
                child_op_name,
            )

        elif "pool" in parent_op_name and "pool" in child_op_name:
            # Pooling operations - no trainable weights
            pass  # No weights to transfer

        else:
            # Default: initialize new operation randomly, transfer compatible parts
            child_state_dict = self._partial_weight_transfer(
                parent_state_dict, child_state_dict, base_pattern
            )

        return child_state_dict

    def _is_identity_morphism(self, parent_op: str, child_op: str) -> bool:
        """Check if operations are identical for weight transfer."""
        identical_pairs = [
            ("sep_conv_3x3", "sep_conv_5x3"),  # Same structure, different kernel
            (
                "inv_bottleneck_2",
                "inv_bottleneck_4",
            ),  # Same structure, different expansion
        ]

        return (parent_op, child_op) in identical_pairs or parent_op == child_op

    def _is_conv_morphism(self, parent_op: str, child_op: str) -> bool:
        """Check if both operations are convolution variants."""
        conv_ops = ["sep_conv", "dil_sep_conv", "group_conv", "inv_bottleneck"]
        return any(op in parent_op for op in conv_ops) and any(
            op in child_op for op in conv_ops
        )

    def _transfer_identical_ops(
        self, parent_state: Dict, child_state: Dict, pattern: str
    ) -> Dict:
        """Transfer weights for identical operation structures."""
        for key in parent_state:
            if pattern in key:
                if (
                    key in child_state
                    and parent_state[key].shape == child_state[key].shape
                ):
                    child_state[key] = parent_state[key].clone()

        return child_state

    def _morph_conv_operations(
        self,
        parent_state: Dict,
        child_state: Dict,
        pattern: str,
        parent_op: str,
        child_op: str,
    ) -> Dict:
        """Morph weights between different convolution operations."""
        # Handle different kernel sizes
        if "3x3" in parent_op and "5x5" in child_op:
            # 3x3 to 5x5 kernel morphism
            child_state = self._morph_kernel_size(
                parent_state, child_state, pattern, 3, 5
            )

        elif "5x5" in parent_op and "3x3" in child_op:
            # 5x5 to 3x3 kernel morphism
            child_state = self._morph_kernel_size(
                parent_state, child_state, pattern, 5, 3
            )

        # Handle depthwise to grouped convolution
        if "sep_conv" in parent_op and "group_conv" in child_op:
            child_state = self._morph_depthwise_to_grouped(
                parent_state, child_state, pattern
            )

        return child_state

    def _morph_kernel_size(
        self,
        parent_state: Dict,
        child_state: Dict,
        pattern: str,
        from_size: int,
        to_size: int,
    ) -> Dict:
        """Morph weights for different kernel sizes."""
        for key in parent_state:
            if pattern in key and "weight" in key:
                parent_weight = parent_state[key]

                if len(parent_weight.shape) == 4:  # Conv weight
                    if from_size < to_size:
                        # Zero-pad smaller kernel to larger size
                        pad = (to_size - from_size) // 2
                        new_weight = F.pad(
                            parent_weight,
                            (pad, pad, pad, pad),
                            mode="constant",
                            value=0,
                        )
                    else:
                        # Crop larger kernel to smaller size
                        start = (from_size - to_size) // 2
                        new_weight = parent_weight[
                            :, :, start : start + to_size, start : start + to_size
                        ]

                    if (
                        key in child_state
                        and new_weight.shape == child_state[key].shape
                    ):
                        child_state[key] = new_weight.clone()

        return child_state

    def _morph_depthwise_to_grouped(
        self, parent_state: Dict, child_state: Dict, pattern: str
    ) -> Dict:
        """Morph depthwise separable conv to grouped conv."""
        for key in parent_state:
            if pattern in key and "weight" in key:
                parent_weight = parent_state[key]

                if len(parent_weight.shape) == 4:
                    # Depthwise conv has groups=in_channels
                    # Grouped conv has groups=2,4,etc.
                    # For simplicity, repeat depthwise filters for groups
                    if parent_weight.size(1) == 1:  # Depthwise
                        groups = 2  # Default group size
                        repeated_weight = parent_weight.repeat(1, groups, 1, 1)

                        if (
                            key in child_state
                            and repeated_weight.shape == child_state[key].shape
                        ):
                            child_state[key] = repeated_weight.clone()

        return child_state

    def _partial_weight_transfer(
        self, parent_state: Dict, child_state: Dict, pattern: str
    ) -> Dict:
        """Transfer compatible weights and initialize rest randomly."""
        for key in parent_state:
            if pattern in key:
                if key in child_state:
                    parent_tensor = parent_state[key]
                    child_tensor = child_state[key]

                    if parent_tensor.shape == child_tensor.shape:
                        # Direct transfer
                        child_state[key] = parent_tensor.clone()
                    else:
                        # Partial transfer for compatible dimensions
                        min_shape = [
                            min(p, c)
                            for p, c in zip(parent_tensor.shape, child_tensor.shape)
                        ]

                        slices = tuple(slice(0, dim) for dim in min_shape)
                        child_state[key][slices] = parent_tensor[slices].clone()

        return child_state

    def _genome_to_key(self, genome: Dict) -> str:
        """Convert genome to cache key."""
        return str(sorted(genome.items()))


class MorphismEnhancedEvolution:
    """Evolutionary search enhanced with network morphism."""

    def __init__(self, base_searcher, morphism: NetworkMorphism):
        self.base_searcher = base_searcher
        self.morphism = morphism
        self.parent_child_map = {}  # Track genealogy

    def evaluate_with_morphism(
        self, population: List[Dict], generation: int
    ) -> np.ndarray:
        """Evaluate population with weight inheritance."""
        fitness = np.zeros((len(population), 3))

        for i, individual in enumerate(population):
            # Try to find morphism parent
            parent_genome, parent_model = self._find_morphism_parent(individual)

            if parent_genome and parent_model:
                # Use morphism for faster evaluation
                fitness[i] = self._evaluate_with_inheritance(
                    individual, parent_genome, parent_model
                )
            else:
                # Standard evaluation
                fitness[i] = self.base_searcher.evaluate_individual(individual)

        return fitness

    def _find_morphism_parent(
        self, child_genome: Dict
    ) -> Tuple[Optional[Dict], Optional[nn.Module]]:
        """Find suitable parent for network morphism."""
        best_parent = None
        best_parent_model = None
        best_similarity = 0

        for parent_key, parent_data in self.morphism.weight_cache.items():
            parent_genome = parent_data["genome"]

            if self.morphism.can_morph(parent_genome, child_genome):
                similarity = self._calculate_genome_similarity(
                    parent_genome, child_genome
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_parent = parent_genome
                    best_parent_model = self._reconstruct_parent_model(parent_data)

        return best_parent, best_parent_model

    def _calculate_genome_similarity(self, genome1: Dict, genome2: Dict) -> float:
        """Calculate similarity between two genomes."""
        similarity = 0.0
        total_genes = 0

        # Compare normal cells
        for gene1, gene2 in zip(genome1["normal_cell"], genome2["normal_cell"]):
            if gene1[1] == gene2[1]:  # Same operation
                similarity += 1
            total_genes += 1

        # Compare reduction cells
        for gene1, gene2 in zip(genome1["reduction_cell"], genome2["reduction_cell"]):
            if gene1[1] == gene2[1]:  # Same operation
                similarity += 1
            total_genes += 1

        # Compare hyperparameters
        hyper1 = genome1["hyperparams"]
        hyper2 = genome2["hyperparams"]

        if hyper1["num_cells"] == hyper2["num_cells"]:
            similarity += 1
        if hyper1["init_channels"] == hyper2["init_channels"]:
            similarity += 1

        total_genes += 2

        return similarity / total_genes

    def _reconstruct_parent_model(self, parent_data: Dict) -> nn.Module:
        """Reconstruct parent model from cached data."""
        genome = parent_data["genome"]
        state_dict = parent_data["state_dict"]

        model = create_model_from_genome(genome)
        model.load_state_dict(state_dict)

        return model

    def _evaluate_with_inheritance(
        self, child_genome: Dict, parent_genome: Dict, parent_model: nn.Module
    ) -> np.ndarray:
        """Evaluate child with weight inheritance from parent."""

        # Create child model
        child_model = create_model_from_genome(child_genome)

        # Apply network morphism
        child_model = self.morphism.morph_weights(
            parent_genome, child_genome, parent_model, child_model
        )

        # Quick evaluation with warm start
        accuracy = self._quick_evaluate(child_model)

        # Estimate latency and parameters
        from ..evaluation.zero_cost_proxies import ZeroCostProxies

        zc = ZeroCostProxies()

        # Mock input for estimation
        x = torch.randn(1, 3, 32, 32)
        latency = zc.estimate_latency(child_model, x)
        params = zc.parameter_count(child_model)

        # Cache successful morphism
        self.morphism.register_parent(child_genome, child_model)

        return np.array([accuracy, latency, params])

    def _quick_evaluate(self, model: nn.Module, epochs: int = 5) -> float:
        """Quick evaluation with warm start."""
        # This would use your existing training infrastructure
        # Simplified implementation
        return 80.0 + np.random.uniform(0, 10)  # Mock accuracy
