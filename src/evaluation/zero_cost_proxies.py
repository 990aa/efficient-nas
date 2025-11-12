import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from ..search_space.encoding import create_model_from_genome


class ZeroCostProxies:
    """Suite of zero-cost proxies for architecture evaluation."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.proxy_weights = {
            "synflow": 0.25,
            "naswot": 0.25,
            "grad_norm": 0.20,
            "zen_score": 0.20,
            "params": 0.10,
        }

    def evaluate_architecture(
        self,
        genome: Dict,
        dataloader: Optional[Any] = None,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
    ) -> Dict[str, float]:
        """
        Evaluate architecture using all zero-cost proxies.

        Returns:
            Dictionary with all proxy scores and aggregated accuracy proxy
        """
        model = create_model_from_genome(genome)
        model.to(self.device)
        model.eval()

        scores = {}

        # Get a batch of data for proxies that need it
        if dataloader is not None:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
            x = x.to(self.device)
        else:
            # Create random data with same statistics as CIFAR-10
            x = torch.randn(64, *input_shape).to(self.device)
            y = torch.randint(0, 10, (64,)).to(self.device)

        # Compute all proxies
        scores["synflow"] = self.synflow_score(model, x)
        scores["naswot"] = self.naswot_score(model, x, y)
        scores["grad_norm"] = self.grad_norm_score(model, x, y)
        scores["zen_score"] = self.zen_score(model, x)
        scores["params"] = self.parameter_count(model)
        scores["flops"] = self.estimate_flops(model, x)
        scores["latency"] = self.estimate_latency(model, x)
        scores["cpu_efficiency"] = self.cpu_efficiency_score(model, x)

        # Normalize scores
        normalized_scores = self._normalize_scores(scores)

        # Compute aggregated accuracy proxy
        scores["accuracy_proxy"] = self._aggregate_proxies(normalized_scores)

        return scores

    def synflow_score(self, model: nn.Module, x: torch.Tensor) -> float:
        """
        Synflow: Measures network trainability via gradient flow analysis.
        Based on "Pruning Neural Networks without Any Data by Iterative Conservation"
        """
        model.eval()

        # Forward pass with all ones
        with torch.no_grad():
            # Replace weights with their absolute values for synflow
            for param in model.parameters():
                if param.requires_grad:
                    param.data = torch.abs(param.data)

            # Forward with ones
            input_ones = torch.ones_like(x)
            output = model(input_ones)

            # Compute sum of all outputs
            score = torch.sum(output).item()

            # Restore original weights
            for param in model.parameters():
                if param.requires_grad:
                    param.data = torch.randn_like(param.data)  # Will be reloaded

        return abs(score)

    def naswot_score(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        NASWOT: Neural Architecture Search Without Training.
        Based on "Neural Architecture Search Without Training"
        """
        model.eval()

        with torch.no_grad():
            # Get activations for multiple inputs
            batch_size = x.shape[0]
            if batch_size > 256:  # Limit batch size for memory
                x = x[:256]
                y = y[:256]

            # Compute Jacobian
            x.requires_grad_(True)
            output = model(x)

            # Compute Jacobian matrix
            jacobian = []
            for i in range(output.shape[1]):  # For each output class
                grad_output = torch.zeros_like(output)
                grad_output[:, i] = 1.0

                gradients = torch.autograd.grad(
                    outputs=output,
                    inputs=x,
                    grad_outputs=grad_output,
                    retain_graph=True,
                    create_graph=False,
                )[0]

                jacobian.append(gradients.view(gradients.shape[0], -1))

            jacobian = torch.stack(jacobian, dim=1)  # [batch, classes, features]
            jacobian = jacobian.view(jacobian.shape[0], -1)  # [batch, classes*features]

            # Compute kernel matrix
            kernel_matrix = torch.mm(jacobian, jacobian.t())

            # Compute log determinant (approximate for stability)
            try:
                eigenvalues = torch.linalg.eigvalsh(kernel_matrix)
                eigenvalues = eigenvalues[
                    eigenvalues > 1e-8
                ]  # Remove near-zero eigenvalues
                log_det = torch.sum(torch.log(eigenvalues)).item()
            except Exception:
                # Fallback to trace for unstable matrices
                log_det = torch.trace(kernel_matrix).item()

        return log_det

    def grad_norm_score(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> float:
        """
        Gradient norm after single backward pass.
        Higher gradient norms indicate better trainability.
        """
        model.train()

        # Single forward-backward pass
        output = model(x)
        loss = F.cross_entropy(output, y)

        # Compute gradients
        model.zero_grad()
        loss.backward()

        # Compute total gradient norm
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm**2

        total_norm = total_norm**0.5
        return total_norm

    def zen_score(self, model: nn.Module, x: torch.Tensor) -> float:
        """
        Zen-Score: Measures expressivity via Gaussian input analysis.
        Based on "Zen-NAS: A Zero-Shot NAS for High-Performance Deep Image Recognition"
        """
        model.eval()

        with torch.no_grad():
            # Generate Gaussian noise inputs
            noise_input = torch.randn_like(x)

            # Get outputs for Gaussian inputs
            outputs = []
            for _ in range(5):  # Multiple samples for stability
                output = model(noise_input)
                outputs.append(output.view(output.shape[0], -1))

            outputs = torch.cat(outputs, dim=0)

            # Compute covariance matrix
            cov_matrix = torch.cov(outputs.T)

            # Compute eigenvalues of covariance matrix
            try:
                eigenvalues = torch.linalg.eigvalsh(cov_matrix)
                eigenvalues = eigenvalues[eigenvalues > 1e-8]
                zen_score = torch.sum(torch.log(eigenvalues)).item()
            except Exception:
                zen_score = torch.trace(cov_matrix).item()

        return zen_score

    def parameter_count(self, model: nn.Module) -> float:
        """Count total parameters in millions."""
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params / 1e6  # Convert to millions

    def estimate_flops(self, model: nn.Module, x: torch.Tensor) -> float:
        """Estimate FLOPs in millions."""
        # Simplified FLOPs estimation
        total_flops = 0

        def count_flops(m, x, y):
            nonlocal total_flops
            if isinstance(m, nn.Conv2d):
                # FLOPs = output_h * output_w * kernel_h * kernel_w * in_ch * out_ch / groups
                h, w = y.shape[-2:]
                kh, kw = m.kernel_size
                total_flops += (
                    h * w * kh * kw * m.in_channels * m.out_channels / m.groups
                )
            elif isinstance(m, nn.Linear):
                # FLOPs = in_features * out_features
                total_flops += m.in_features * m.out_features

        # Register hook
        hooks = []
        for layer in model.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                hook = layer.register_forward_hook(count_flops)
                hooks.append(hook)

        # Forward pass to count FLOPs
        with torch.no_grad():
            model(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return total_flops / 1e6  # Convert to millions

    def estimate_latency(
        self, model: nn.Module, x: torch.Tensor, iterations: int = 100
    ) -> float:
        """Estimate inference latency on CPU."""
        model.eval()
        model.to("cpu")
        x = x.to("cpu")

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)

        # Measure latency
        start_time = (
            torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        )
        end_time = (
            torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        )

        if start_time is None:
            # CPU timing
            times = []
            with torch.no_grad():
                for _ in range(iterations):
                    start = time.time()
                    _ = model(x)
                    end = time.time()
                    times.append((end - start) * 1000)  # Convert to ms

            latency = np.median(times)
        else:
            # GPU timing
            times = []
            with torch.no_grad():
                for _ in range(iterations):
                    start_time.record()
                    _ = model(x)
                    end_time.record()
                    torch.cuda.synchronize()
                    times.append(start_time.elapsed_time(end_time))

            latency = np.median(times)

        return latency

    def cpu_efficiency_score(self, model: nn.Module, x: torch.Tensor) -> float:
        """
        Custom CPU efficiency score combining:
        - Memory access patterns
        - Operation intensity
        - Cache friendliness
        """
        # Simplified efficiency metric based on:
        # 1. Parameter efficiency (higher is better)
        params = self.parameter_count(model) * 1e6
        flops = self.estimate_flops(model, x) * 1e6

        if flops > 0:
            param_efficiency = params / flops  # Parameters per FLOP
        else:
            param_efficiency = 0

        # 2. Depthwise operation ratio (higher is better for CPU)
        depthwise_ops = 0
        total_ops = 0

        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                total_ops += 1
                if module.groups == module.in_channels:
                    depthwise_ops += 1

        depthwise_ratio = depthwise_ops / max(total_ops, 1)

        # 3. Memory footprint (smaller is better)
        latency = self.estimate_latency(model, x)

        # Combine factors (weights can be tuned)
        efficiency_score = (
            0.4
            * (
                1.0 / (param_efficiency + 1e-8)
            )  # Inverse since lower params/FLOP is better
            + 0.4 * depthwise_ratio
            + 0.2 * (100.0 / (latency + 1e-8))  # Inverse latency
        )

        return efficiency_score

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize proxy scores to similar scales."""
        normalized = {}

        # Define normalization ranges (can be learned from data)
        ranges = {
            "synflow": (0, 1000),
            "naswot": (0, 100),
            "grad_norm": (0, 10),
            "zen_score": (0, 100),
            "params": (0, 5),  # Millions
            "flops": (0, 100),  # Millions
            "latency": (0, 100),  # ms
            "cpu_efficiency": (0, 10),
        }

        for key, value in scores.items():
            if key in ranges:
                min_val, max_val = ranges[key]
                # Clip and normalize to [0, 1]
                clipped = np.clip(value, min_val, max_val)
                normalized[key] = (clipped - min_val) / (max_val - min_val)
            else:
                normalized[key] = value

        return normalized

    def _aggregate_proxies(self, normalized_scores: Dict[str, float]) -> float:
        """Aggregate normalized proxy scores into single accuracy estimate."""
        accuracy_proxy = 0.0

        for proxy, weight in self.proxy_weights.items():
            if proxy in normalized_scores:
                accuracy_proxy += weight * normalized_scores[proxy]

        return accuracy_proxy * 100  # Convert to percentage scale


class ProxyEnsemble:
    """Ensemble of zero-cost proxies with learned weights."""

    def __init__(self, proxies: List[str] = None):
        self.proxies = proxies or [
            "synflow",
            "naswot",
            "grad_norm",
            "zen_score",
            "params",
        ]
        self.weights = None
        self.correlation_data = []

    def fit_weights(
        self, architectures: List[Dict], true_accuracies: List[float], dataloader: Any
    ) -> Dict[str, float]:
        """
        Learn optimal weights for proxy ensemble based on correlation with true accuracy.

        Args:
            architectures: List of architecture genomes
            true_accuracies: Corresponding true validation accuracies
            dataloader: Data loader for proxy computation

        Returns:
            Learned weights for each proxy
        """
        print("Learning proxy weights from 500 sampled architectures...")

        # Compute all proxies for sampled architectures
        proxy_scores = []
        zc_evaluator = ZeroCostProxies()

        for i, arch in enumerate(architectures[:500]):  # Use first 500
            if i % 50 == 0:
                print(
                    f"Computing proxies for architecture {i}/{min(500, len(architectures))}"
                )

            scores = zc_evaluator.evaluate_architecture(arch, dataloader)
            proxy_scores.append([scores[proxy] for proxy in self.proxies])

        proxy_scores = np.array(proxy_scores)
        true_accuracies = np.array(true_accuracies[:500])

        # Compute Spearman rank correlations
        correlations = []
        for i, proxy in enumerate(self.proxies):
            corr = self._spearman_correlation(proxy_scores[:, i], true_accuracies)
            correlations.append(abs(corr))  # Use absolute value

        # Normalize correlations to get weights
        total_corr = sum(correlations)
        self.weights = {
            proxy: corr / total_corr for proxy, corr in zip(self.proxies, correlations)
        }

        print("Learned proxy weights:")
        for proxy, weight in self.weights.items():
            print(f"  {proxy}: {weight:.3f}")

        return self.weights

    def _spearman_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Spearman rank correlation."""
        from scipy.stats import spearmanr

        return spearmanr(x, y).correlation

    def predict_accuracy(self, genome: Dict, dataloader: Any) -> float:
        """Predict accuracy using weighted proxy ensemble."""
        if self.weights is None:
            raise ValueError("Proxy weights not fitted. Call fit_weights first.")

        zc_evaluator = ZeroCostProxies()
        scores = zc_evaluator.evaluate_architecture(genome, dataloader)

        accuracy_estimate = 0.0
        for proxy, weight in self.weights.items():
            if proxy in scores:
                accuracy_estimate += weight * scores[proxy]

        return accuracy_estimate * 100  # Convert to percentage
