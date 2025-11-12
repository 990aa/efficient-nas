import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import time
from typing import Dict, Any, List
import numpy as np
from ..search_space.encoding import create_model_from_genome


class TrainingProtocol:
    """Standardized training protocols for search and evaluation phases."""

    def __init__(self, device: str = "cpu"):
        self.device = device

        # Search phase hyperparameters
        self.search_hparams = {
            "epochs": 20,
            "optimizer": "SGD",
            "lr": 0.025,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "scheduler": "cosine",
            "batch_size": 96,
        }

        # Final evaluation hyperparameters
        self.eval_hparams = {
            "epochs": 200,
            "optimizer": "SGD",
            "lr": 0.025,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "scheduler": "cosine",
            "batch_size": 96,
            "cutout": True,
        }

    def search_phase_training(
        self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader
    ) -> float:
        """
        20-epoch training for architecture search phase.
        Returns validation accuracy.
        """
        return self._train_model(
            model, train_loader, val_loader, self.search_hparams, phase="search"
        )

    def final_evaluation(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        seeds: List[int] = [42, 123, 456],
    ) -> Dict[str, Any]:
        """
        Final 200-epoch training with multiple seeds.
        Returns mean ± std statistics.
        """
        accuracies = []
        training_times = []

        for i, seed in enumerate(seeds):
            print(f"Final evaluation run {i + 1}/{len(seeds)} with seed {seed}")

            # Set random seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Create fresh model instance
            model_copy = type(model)(**self._get_model_config(model))
            model_copy.load_state_dict(model.state_dict())

            start_time = time.time()
            accuracy = self._train_model(
                model_copy,
                train_loader,
                test_loader,
                self.eval_hparams,
                phase="evaluation",
            )
            end_time = time.time()

            accuracies.append(accuracy)
            training_times.append(end_time - start_time)

        return {
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "accuracy_values": accuracies,
            "time_mean": np.mean(training_times),
            "time_std": np.std(training_times),
            "time_values": training_times,
        }

    def _train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        hparams: Dict[str, Any],
        phase: str = "search",
    ) -> float:
        """Core training loop."""
        model.to(self.device)

        # Setup optimizer
        if hparams["optimizer"] == "SGD":
            optimizer = optim.SGD(
                model.parameters(),
                lr=hparams["lr"],
                momentum=hparams["momentum"],
                weight_decay=hparams["weight_decay"],
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=hparams["lr"],
                weight_decay=hparams["weight_decay"],
            )

        # Setup scheduler
        if hparams["scheduler"] == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=hparams["epochs"])
        else:
            scheduler = None

        criterion = nn.CrossEntropyLoss()

        best_accuracy = 0.0

        for epoch in range(hparams["epochs"]):
            # Training phase
            model.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation phase
            accuracy = self._evaluate_model(model, val_loader)
            best_accuracy = max(best_accuracy, accuracy)

            if scheduler:
                scheduler.step()

            # Progress reporting
            if phase == "search" and epoch % 5 == 0:
                print(
                    f"Search Epoch {epoch}/{hparams['epochs']}, "
                    f"Accuracy: {accuracy:.2f}%"
                )
            elif phase == "evaluation" and epoch % 50 == 0:
                print(
                    f"Evaluation Epoch {epoch}/{hparams['epochs']}, "
                    f"Accuracy: {accuracy:.2f}%"
                )

        return best_accuracy

    def _evaluate_model(self, model: nn.Module, data_loader: DataLoader) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        return 100.0 * correct / total

    def _get_model_config(self, model: nn.Module) -> Dict[str, Any]:
        """Extract model configuration for reinitialization."""
        # This is a simplified version - would need to be adapted for specific model types
        return {
            "C": getattr(model, "C", 32),
            "num_classes": getattr(model, "num_classes", 10),
            "layers": getattr(model, "layers", 3),
            "steps": getattr(model, "steps", 4),
        }


class BenchmarkTrainer:
    """Orchestrates training across multiple datasets and architectures."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.training_protocol = TrainingProtocol(device)
        self.results = {}

    def evaluate_architecture(
        self, genome: Dict, dataset: str = "cifar10", phase: str = "final"
    ) -> Dict[str, Any]:
        """
        Evaluate architecture on specified dataset.

        Args:
            genome: Architecture encoding
            dataset: Dataset name ('cifar10', 'cifar100', etc.)
            phase: 'search' or 'final'

        Returns:
            Evaluation results
        """
        from ..utils.data_loader import DatasetManager

        dataset_manager = DatasetManager()

        # Get appropriate data loaders
        if dataset == "cifar10":
            if phase == "search":
                train_loader, val_loader, _ = dataset_manager.get_cifar10_loaders()
                test_loader = val_loader
            else:
                train_loader, _, test_loader = dataset_manager.get_cifar10_loaders(
                    cutout=True
                )
        elif dataset == "cifar100":
            train_loader, test_loader = dataset_manager.get_cifar100_loaders(
                cutout=True
            )
            val_loader = test_loader  # For search phase
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # Create model
        num_classes = dataset_manager.dataset_info[dataset]["num_classes"]
        model = create_model_from_genome(genome, num_classes)

        # Train and evaluate
        if phase == "search":
            accuracy = self.training_protocol.search_phase_training(
                model, train_loader, val_loader
            )

            # Estimate latency and parameters
            from .zero_cost_proxies import ZeroCostProxies

            zc = ZeroCostProxies(self.device)
            x, y = next(iter(test_loader))
            x, y = x.to(self.device), y.to(self.device)

            latency = zc.estimate_latency(model, x)
            params = zc.parameter_count(model)

            return {"accuracy": accuracy, "latency": latency, "params": params}

        else:  # final evaluation
            results = self.training_protocol.final_evaluation(
                model, train_loader, test_loader
            )

            # Add architecture metadata
            results["genome"] = genome
            results["dataset"] = dataset

            return results
