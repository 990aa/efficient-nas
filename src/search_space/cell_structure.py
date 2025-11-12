import torch
import torch.nn as nn
from typing import List, Tuple
from .operations import OPS, FactorizedReduce, ReLUConvBN

class Cell(nn.Module):
    """A single cell in the neural architecture."""
    
    def __init__(self, cell_type: str, steps: int, multiplier: int, 
                 C_prev_prev: int, C_prev: int, C: int, reduction: bool, 
                 reduction_prev: bool, genome: List[Tuple[int, int, int]]):
        """
        Args:
            cell_type: 'normal' or 'reduction'
            steps: Number of intermediate nodes in the cell
            multiplier: Output channel multiplier
            C_prev_prev: Channels in previous previous cell
            C_prev: Channels in previous cell
            C: Base channels for this cell
            reduction: Whether this is a reduction cell
            reduction_prev: Whether previous cell was reduction
            genome: Architecture encoding for this cell
        """
        super().__init__()
        self.cell_type = cell_type
        self.steps = steps
        self.multiplier = multiplier
        
        # Preprocessing for previous outputs
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        
        # Operation edges
        self.ops = nn.ModuleList()
        self.connections = []
        
        for i in range(self.steps):
            # Each step connects two previous nodes
            node_ops = nn.ModuleList()
            for j in range(2):  # Two inputs per node
                # Parse genome for this connection
                input_idx, op_idx, _ = genome[i * 2 + j]
                stride = 2 if reduction and input_idx < 2 else 1
                
                # Create operation
                op_name = list(OPS.keys())[op_idx]
                op = OPS[op_name](C, stride)
                node_ops.append(op)
                
            self.ops.append(node_ops)
            self.connections.append([genome[i * 2][0], genome[i * 2 + 1][0]])
            
    def forward(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        """Forward pass through the cell."""
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        states = [s0, s1]
        
        for i in range(self.steps):
            # Get inputs for this node
            input1_idx, input2_idx = self.connections[i]
            op1, op2 = self.ops[i]
            
            # Apply operations
            input1 = states[input1_idx]
            input2 = states[input2_idx]
            
            out1 = op1(input1)
            out2 = op2(input2)
            
            # Combine outputs (addition)
            node_output = out1 + out2
            states.append(node_output)
            
        # Concatenate all intermediate node outputs
        return torch.cat(states[-self.multiplier:], dim=1)

class Network(nn.Module):
    """Complete neural network composed of multiple cells."""
    
    def __init__(self, C: int, num_classes: int, layers: int, 
                 steps: int, multiplier: int, stem_multiplier: int,
                 normal_genome: List[Tuple[int, int, int]], 
                 reduce_genome: List[Tuple[int, int, int]]):
        super().__init__()
        self.C = C
        self.num_classes = num_classes
        self.layers = layers
        self.steps = steps
        self.multiplier = multiplier
        
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        reduction_prev = False
        
        self.cells = nn.ModuleList()
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                # Reduction cell
                C_curr *= 2
                cell = Cell('reduce', steps, multiplier, C_prev_prev, C_prev, 
                           C_curr, True, reduction_prev, reduce_genome)
                reduction_prev = True
            else:
                # Normal cell
                cell = Cell('normal', steps, multiplier, C_prev_prev, C_prev,
                           C_curr, False, reduction_prev, normal_genome)
                reduction_prev = False
                
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = s1 = self.stem(x)
        
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            
        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits