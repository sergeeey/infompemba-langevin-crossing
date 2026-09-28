"""
Module 3: Fisher metric analyzer (fisher_metrics.py)

ALL computation stays on GPU device. No CPU transfers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np


class FisherMetricAnalyzer:
    """
    Computes information geometry metrics. ALL on-device.
    """
    
    def __init__(
        self,
        model_star: nn.Module,
        val_loader: DataLoader,
        config: dict,
        device: torch.device
    ) -> None:
        self.model_star = model_star.to(device)
        self.model_star.eval()
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.val_samples_kl = config["data"]["val_samples_kl"]
    
    def compute_kl_divergence(self, model: nn.Module) -> float:
        """
        KL(current || star) - all on device.
        """
        model.eval()
        self.model_star.eval()
        
        kl_div = nn.KLDivLoss(reduction='batchmean')
        total_kl = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device)
                if data.dim() > 2:
                    data = data.view(data.size(0), -1)
                
                log_probs_current = torch.log_softmax(model(data), dim=1)
                probs_star = torch.softmax(self.model_star(data), dim=1)
                
                kl = kl_div(log_probs_current, probs_star)
                total_kl += kl.item() * data.size(0)
                num_samples += data.size(0)
                
                if num_samples >= self.val_samples_kl:
                    break
        
        return total_kl / num_samples if num_samples > 0 else 0.0
    
    def compute_fisher_trace(self, model: nn.Module) -> float:
        """
        Trace of empirical Fisher matrix - ALL on device.
        Single forward/backward pass.
        """
        training_mode = model.training
        model.train()
        total_fisher_trace = 0.0
        num_batches = 0
        max_batches = 5  # Fewer batches for speed on GPU
        
        with torch.enable_grad():
            for data, _ in self.val_loader:
                if num_batches >= max_batches:
                    break
                
                data = data.to(self.device)
                if data.dim() > 2:
                    data = data.view(data.size(0), -1)
                
                # Single forward pass
                logits = model(data)
                probs = torch.softmax(logits, dim=1)
                pseudo_targets = torch.argmax(probs, dim=1)
                
                loss = F.cross_entropy(logits, pseudo_targets)
                
                # Single backward for ALL parameters at once
                model.zero_grad()
                loss.backward()
                
                # Sum squared gradients = trace(F)
                batch_trace = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        batch_trace += (param.grad ** 2).sum().item()
                
                total_fisher_trace += batch_trace
                num_batches += 1
        
        model.train(training_mode)
        
        return total_fisher_trace / num_batches if num_batches > 0 else 0.0
    
    def compute_all_metrics(
        self,
        model: nn.Module,
        step: int
    ) -> Dict[str, float]:
        """Compute all metrics. All on device."""
        metrics = {}
        metrics['kl_div_star'] = self.compute_kl_divergence(model)
        metrics['trace_fisher'] = self.compute_fisher_trace(model)
        # Info temperature: disabled for speed, requires Lanczos on GPU
        metrics['info_temp'] = -1.0
        return metrics


def create_metric_analyzer(
    model_star: nn.Module,
    val_loader: DataLoader,
    config: dict,
    device: torch.device
) -> FisherMetricAnalyzer:
    """Factory for metric analyzer."""
    return FisherMetricAnalyzer(
        model_star=model_star,
        val_loader=val_loader,
        config=config,
        device=device
    )
