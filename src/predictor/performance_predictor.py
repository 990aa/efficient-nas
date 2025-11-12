import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import numpy as np
from typing import List, Dict, Any, Tuple
from ..search_space.encoding import ArchitectureEncoder

class ArchitectureGraph:
    """Convert architecture genome to graph representation for GNN."""
    
    def __init__(self, steps: int = 5, num_ops: int = 13):
        self.steps = steps
        self.num_ops = num_ops
        self.encoder = ArchitectureEncoder(steps, num_ops)
    
    def genome_to_graph(self, genome: Dict[str, Any]) -> Data:
        """
        Convert architecture genome to PyTorch Geometric graph.
        
        Node features: [operation_type, node_type, input_channel, output_channel]
        Edge features: connection_strength
        """
        # Extract cell information
        normal_cell = genome['normal_cell']
        hyperparams = genome['hyperparams']
        
        # Node features: [operation_type, node_type, channels]
        node_features = []
        
        # Input nodes (special type)
        node_features.append([0, 0, hyperparams['init_channels'], hyperparams['init_channels']])  # Input 1
        node_features.append([0, 0, hyperparams['init_channels'], hyperparams['init_channels']])  # Input 2
        
        # Intermediate nodes
        for i in range(self.steps):
            # Each intermediate node has two operations
            op1_idx = normal_cell[i * 2][1]
            op2_idx = normal_cell[i * 2 + 1][1]
            # Use average operation type for the node
            avg_op = (op1_idx + op2_idx) / 2.0
            channels = hyperparams['init_channels']
            node_features.append([avg_op, 1, channels, channels])  # Intermediate node
        
        # Output node (concatenation of intermediate nodes)
        node_features.append([0, 2, 
                            hyperparams['init_channels'] * self.steps,
                            hyperparams['init_channels'] * self.steps])  # Output node
        
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        # Build edge index and edge features
        edge_indices = []
        edge_features = []
        
        # Connect inputs to intermediate nodes based on genome
        for i in range(self.steps):
            # First connection for this node
            input1_idx = normal_cell[i * 2][0]
            edge_indices.append([input1_idx, 2 + i])  # From input to node
            edge_features.append([1.0])  # Connection strength
            
            # Second connection for this node  
            input2_idx = normal_cell[i * 2 + 1][0]
            edge_indices.append([input2_idx, 2 + i])
            edge_features.append([1.0])
        
        # Connect intermediate nodes to output node
        for i in range(self.steps):
            edge_indices.append([2 + i, 2 + self.steps])  # From intermediate to output
            edge_features.append([1.0])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    
    def batch_genomes_to_graphs(self, genomes: List[Dict]) -> List[Data]:
        """Convert batch of genomes to graph data."""
        return [self.genome_to_graph(genome) for genome in genomes]

class ArchitectureGNN(nn.Module):
    """Graph Neural Network for architecture performance prediction."""
    
    def __init__(self, 
                 node_dim: int = 4,
                 edge_dim: int = 1,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 output_dim: int = 2,  # [accuracy, latency]
                 dropout: float = 0.1):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Node embedding
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        # Edge embedding
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = GCNConv(hidden_dim, hidden_dim)
            self.convs.append(conv)
        
        # Attention layers for better representation
        self.attention = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        
        # Batch normalization
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Prediction heads
        self.accuracy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Accuracy prediction
        )
        
        self.latency_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Latency prediction
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)  # Uncertainty (variance)
        )
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through GNN."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Node embedding
        x = self.node_embedding(x)
        
        # Graph convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global attention pooling
        x = self.attention(x, edge_index)
        
        # Global mean pooling
        x = global_mean_pool(x, batch)
        
        # Predictions with uncertainty
        accuracy_pred = self.accuracy_head(x)
        latency_pred = self.latency_head(x)
        uncertainty = F.softplus(self.uncertainty_head(x))  # Ensure positive variance
        
        # Combine predictions and uncertainty
        predictions = torch.cat([accuracy_pred, latency_pred], dim=1)
        
        return predictions, uncertainty
    
    def predict(self, data: Data, num_samples: int = 10) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimation using Monte Carlo dropout.
        """
        self.train()  # Keep dropout active for uncertainty estimation
        
        predictions = []
        uncertainties = []
        
        for _ in range(num_samples):
            pred, unc = self.forward(data)
            predictions.append(pred)
            uncertainties.append(unc)
        
        predictions = torch.stack(predictions)
        uncertainties = torch.stack(uncertainties)
        
        # Compute mean and variance
        mean_pred = predictions.mean(dim=0)
        epistemic_unc = predictions.var(dim=0)  # Model uncertainty
        aleatoric_unc = uncertainties.mean(dim=0)  # Data uncertainty
        
        total_unc = epistemic_unc + aleatoric_unc
        
        return {
            'mean': mean_pred,
            'epistemic_uncertainty': epistemic_unc,
            'aleatoric_uncertainty': aleatoric_unc,
            'total_uncertainty': total_unc,
            'samples': predictions
        }

class PerformancePredictor:
    """Surrogate model for architecture performance prediction."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.graph_converter = ArchitectureGraph()
        self.model = None
        self.is_trained = False
        
        # Training dataset
        self.training_data = {
            'genomes': [],
            'accuracy': [],
            'latency': [],
            'graphs': []
        }
    
    def prepare_training_data(self, genomes: List[Dict], 
                            accuracies: List[float],
                            latencies: List[float]) -> DataLoader:
        """Prepare training data for GNN."""
        print("Converting architectures to graphs...")
        
        graphs = []
        
        for genome, acc, lat in zip(genomes, accuracies, latencies):
            graph = self.graph_converter.genome_to_graph(genome)
            target = torch.tensor([acc, lat], dtype=torch.float)
            graph.y = target
            graphs.append(graph)
        
        # Store for later use
        self.training_data['genomes'] = genomes
        self.training_data['accuracy'] = accuracies
        self.training_data['latency'] = latencies
        self.training_data['graphs'] = graphs
        
        return DataLoader(graphs, batch_size=32, shuffle=True)
    
    def train_predictor(self, 
                       genomes: List[Dict],
                       accuracies: List[float], 
                       latencies: List[float],
                       epochs: int = 100,
                       validation_split: float = 0.2) -> Dict[str, List[float]]:
        """
        Train the GNN performance predictor.
        
        Args:
            genomes: List of architecture genomes
            accuracies: Corresponding accuracy values
            latencies: Corresponding latency values
            epochs: Number of training epochs
            validation_split: Fraction of data for validation
            
        Returns:
            Training history
        """
        # Prepare data
        dataloader = self.prepare_training_data(genomes, accuracies, latencies)
        
        # Split data
        dataset_size = len(dataloader.dataset)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataloader.dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        self.model = ArchitectureGNN().to(self.device)
        
        # Loss function (Huber loss for robustness)
        criterion = nn.HuberLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy_mae': [],
            'val_accuracy_mae': []
        }
        
        print("Training performance predictor...")
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_acc_mae = 0.0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                predictions, _ = self.model(batch)
                loss = criterion(predictions, batch.y)
                
                # Accuracy MAE
                acc_mae = F.l1_loss(predictions[:, 0], batch.y[:, 0])
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_acc_mae += acc_mae.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_acc_mae = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    predictions, _ = self.model(batch)
                    loss = criterion(predictions, batch.y)
                    acc_mae = F.l1_loss(predictions[:, 0], batch.y[:, 0])
                    
                    val_loss += loss.item()
                    val_acc_mae += acc_mae.item()
            
            # Average losses
            train_loss /= len(train_loader)
            train_acc_mae /= len(train_loader)
            val_loss /= len(val_loader)
            val_acc_mae /= len(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_accuracy_mae'].append(train_acc_mae)
            history['val_accuracy_mae'].append(val_acc_mae)
            
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Val Acc MAE: {val_acc_mae:.4f}")
        
        self.is_trained = True
        print("Performance predictor training completed!")
        
        return history
    
    def predict_performance(self, genomes: List[Dict], 
                          with_uncertainty: bool = True) -> Dict[str, np.ndarray]:
        """
        Predict performance for multiple architectures.
        
        Returns:
            Dictionary with mean predictions and uncertainties
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Predictor not trained. Call train_predictor first.")
        
        self.model.eval()
        
        graphs = [self.graph_converter.genome_to_graph(genome) for genome in genomes]
        dataloader = DataLoader(graphs, batch_size=32, shuffle=False)
        
        all_predictions = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                
                if with_uncertainty:
                    results = self.model.predict(batch)
                    all_predictions.append(results['mean'].cpu().numpy())
                    all_uncertainties.append(results['total_uncertainty'].cpu().numpy())
                else:
                    predictions, _ = self.model(batch)
                    all_predictions.append(predictions.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        
        result = {
            'accuracy_mean': predictions[:, 0],
            'latency_mean': predictions[:, 1],
        }
        
        if with_uncertainty:
            uncertainties = np.vstack(all_uncertainties)
            result.update({
                'accuracy_uncertainty': uncertainties[:, 0],
                'latency_uncertainty': uncertainties[:, 1],
            })
        
        return result
    
    def save_model(self, filepath: str):
        """Save trained model and training data."""
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'training_data': self.training_data,
            'graph_converter_config': {
                'steps': self.graph_converter.steps,
                'num_ops': self.graph_converter.num_ops
            }
        }
        
        torch.save(save_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model."""
        save_data = torch.load(filepath, map_location=self.device)
        
        # Reconstruct graph converter
        config = save_data['graph_converter_config']
        self.graph_converter = ArchitectureGraph(
            steps=config['steps'], 
            num_ops=config['num_ops']
        )
        
        # Reconstruct model
        self.model = ArchitectureGNN().to(self.device)
        self.model.load_state_dict(save_data['model_state_dict'])
        
        # Restore training data
        self.training_data = save_data['training_data']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")

class PredictorEnhancedSearch:
    """NSGA-II search enhanced with performance predictor."""
    
    def __init__(self, predictor: PerformancePredictor, 
                 base_searcher, 
                 predictor_weight: float = 0.7):
        self.predictor = predictor
        self.base_searcher = base_searcher
        self.predictor_weight = predictor_weight
        self.evaluation_cache = {}
    
    def evaluate_with_predictor(self, population: List[Dict], 
                              generation: int) -> np.ndarray:
        """
        Hybrid evaluation using predictor and actual evaluation.
        
        Early generations: Mostly predictor
        Late generations: Mostly actual evaluation
        """
        total_generations = self.base_searcher.generations
        
        # Dynamic weighting based on generation
        if generation < total_generations // 3:
            predictor_weight = 0.8
        elif generation < 2 * total_generations // 3:
            predictor_weight = 0.5
        else:
            predictor_weight = 0.2
        
        # Get predictor estimates
        predictor_results = self.predictor.predict_performance(population)
        pred_accuracy = predictor_results['accuracy_mean']
        pred_latency = predictor_results['latency_mean']
        
        # Get actual evaluations for a subset
        actual_indices = np.random.choice(
            len(population), 
            size=max(1, int(len(population) * (1 - predictor_weight))),
            replace=False
        )
        
        actual_fitness = np.zeros((len(population), 3))
        
        for i in range(len(population)):
            if i in actual_indices:
                # Actual evaluation
                cache_key = str(population[i])
                if cache_key in self.evaluation_cache:
                    fitness = self.evaluation_cache[cache_key]
                else:
                    # This would call your actual evaluation function
                    fitness = self._evaluate_architecture(population[i])
                    self.evaluation_cache[cache_key] = fitness
                actual_fitness[i] = fitness
            else:
                # Use predictor with uncertainty adjustment
                accuracy = pred_accuracy[i]
                latency = pred_latency[i]
                # Simple parameter estimation
                params = self._estimate_parameters(population[i])
                actual_fitness[i] = [accuracy, latency, params]
        
        return actual_fitness
    
    def _evaluate_architecture(self, genome: Dict) -> np.ndarray:
        """Placeholder for actual architecture evaluation."""
        # This would integrate with your existing evaluation system
        # Mock implementation
        return np.array([85.0, 10.0, 1.5])  # [accuracy, latency, params]
    
    def _estimate_parameters(self, genome: Dict) -> float:
        """Estimate model parameters from genome."""
        from ..search_space.encoding import ConstraintAwareInitializer
        encoder = ArchitectureEncoder()
        initializer = ConstraintAwareInitializer(encoder)
        return initializer.estimate_parameters(genome) / 1e6  # Convert to millions