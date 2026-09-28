"""
Модуль 1: Генератор начальных состояний (initializers.py)

Создает три набора весов для архитектуры нейронной сети:
- theta_star: эталонная модель (обученная до сходимости)
- theta_cold: "переохлажденная" - малые возмущения theta_star
- theta_hot: "перегретая" - большая инициализация с тяжелыми хвостами
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Dict, Optional
import copy
import os


class MLP(nn.Module):
    """
    Многослойный перцептрон с активацией Tanh (для гладкости Фишера).
    """
    
    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "tanh",
        bias: bool = True
    ) -> None:
        super(MLP, self).__init__()
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=bias))
            if i < len(layer_sizes) - 2:
                if activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "gelu":
                    layers.append(nn.GELU())
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
        
        self.network = nn.Sequential(*layers)
        self.layer_sizes = layer_sizes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


def create_model(config: dict) -> MLP:
    """Создает модель MLP согласно конфигурации."""
    return MLP(
        layer_sizes=config["model"]["layers"],
        activation=config["model"]["activation"],
        bias=config["model"]["bias"]
    )


def train_reference_model(
    config: dict,
    device: torch.device,
    save_path: str
) -> MLP:
    """Обучает эталонную модель theta_star до сходимости."""
    print("=> Training reference model theta_star...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    dataset_name = config["data"]["dataset"]
    data_dir = config["data"]["data_dir"]
    
    if dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(
            data_dir, train=True, download=True, transform=transform
        )
    elif dataset_name == "MNIST":
        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True,
        num_workers=config["hardware"]["num_workers"], pin_memory=True
    )
    
    model = create_model(config).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["initialization"]["ref_lr"],
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["initialization"]["ref_epochs"]
    )
    
    criterion = nn.CrossEntropyLoss()
    target_loss = config["initialization"]["ref_target_loss"]
    ref_epochs = config["initialization"]["ref_epochs"]
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(ref_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / num_batches
        
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{ref_epochs}, Loss: {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())
        
        if avg_loss < target_loss:
            print(f"   => Reached target loss {target_loss} at epoch {epoch+1}")
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"   => Reference model saved at {save_path} (final loss: {best_loss:.6f})")
    
    return model


def compute_kl_divergence(
    model_current: nn.Module,
    model_star: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> float:
    """
    Computes KL(current || star) on validation data.
    All computation stays on device.
    """
    model_current.eval()
    model_star.eval()
    
    kl_div = nn.KLDivLoss(reduction='batchmean')
    total_kl = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            if data.dim() > 2:
                data = data.view(data.size(0), -1)
            
            log_probs_current = torch.log_softmax(model_current(data), dim=1)
            probs_star = torch.softmax(model_star(data), dim=1)
            
            kl = kl_div(log_probs_current, probs_star)
            total_kl += kl.item() * data.size(0)
            num_samples += data.size(0)
    
    return total_kl / num_samples if num_samples > 0 else 0.0


def create_cold_initialization(
    model_star: nn.Module,
    noise_std: float,
    seed: int,
    device: torch.device
) -> nn.Module:
    """Creates cold init: theta_star + small Gaussian noise."""
    torch.manual_seed(seed)
    
    model_cold = copy.deepcopy(model_star)
    
    with torch.no_grad():
        for param in model_cold.parameters():
            noise = torch.randn_like(param) * noise_std
            param.add_(noise)
    
    return model_cold


def create_hot_initialization(
    model_class: type,
    config: dict,
    seed: int,
    device: torch.device
) -> nn.Module:
    """Creates hot init: Kaiming Uniform * gain + Laplace noise."""
    torch.manual_seed(seed)
    
    model = model_class(
        layer_sizes=config["model"]["layers"],
        activation=config["model"]["activation"],
        bias=config["model"]["bias"]
    ).to(device)
    
    gain_multiplier = config["initialization"]["hot_gain_multiplier"]
    laplace_scale = config["initialization"]["hot_laplace_scale"]
    
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.0, mode='fan_in')
                module.weight.mul_(gain_multiplier)
                
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / torch.sqrt(torch.tensor(fan_in, dtype=torch.float))
                    nn.init.uniform_(module.bias, -bound, bound)
                    laplace_dist = torch.distributions.Laplace(
                        loc=torch.zeros_like(module.bias), scale=laplace_scale
                    )
                    module.bias.add_(laplace_dist.sample())
        
        for param in model.parameters():
            laplace_dist = torch.distributions.Laplace(
                loc=torch.zeros_like(param), scale=laplace_scale
            )
            param.add_(laplace_dist.sample())
    
    return model


def auto_tune_cold_noise(
    model_star: nn.Module,
    model_class: type,
    config: dict,
    val_loader: DataLoader,
    device: torch.device,
    target_kl: float = 0.3,
    max_kl: float = 0.5,
    seed: int = 42
) -> Tuple[nn.Module, float]:
    """
    Auto-tunes cold_noise_std so that KL(cold || star) < max_kl.
    Uses binary search.
    """
    low = 0.001
    high = 0.1
    best_noise = 0.005
    best_model = None
    
    print("=> Auto-tuning cold noise sigma...")
    
    for attempt in range(10):
        mid = (low + high) / 2
        model_cold = create_cold_initialization(model_star, mid, seed + 100 + attempt, device)
        kl = compute_kl_divergence(model_cold, model_star, val_loader, device)
        
        print(f"   sigma={mid:.4f} -> KL={kl:.4f}")
        
        if kl < max_kl:
            best_noise = mid
            best_model = copy.deepcopy(model_cold)
            # Try slightly larger to get closer to target
            if kl < target_kl:
                low = mid
            else:
                break
        else:
            high = mid
    
    if best_model is None:
        # Fallback: use smallest noise
        best_noise = 0.001
        best_model = create_cold_initialization(model_star, best_noise, seed + 1, device)
    
    print(f"   => Selected sigma={best_noise:.4f}")
    return best_model, best_noise


def initialize_all(
    config: dict,
    device: torch.device,
    seed: int = 42
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """
    Creates all three initializations with validation.
    Auto-tunes cold_noise_std if needed.
    """
    results_dir = config["output"]["results_dir"]
    theta_star_path = os.path.join(results_dir, config["output"]["theta_star_file"])
    
    # 1. Load or train theta_star
    if os.path.exists(theta_star_path):
        print(f"=> Loading reference model from {theta_star_path}")
        model_star = create_model(config).to(device)
        model_star.load_state_dict(torch.load(theta_star_path, weights_only=True))
    else:
        model_star = train_reference_model(config, device, theta_star_path)
    
    model_star.eval()
    
    # 2. Create validation loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    dataset_name = config["data"]["dataset"]
    data_dir = config["data"]["data_dir"]
    
    if dataset_name == "FashionMNIST":
        val_dataset = datasets.FashionMNIST(
            data_dir, train=False, download=True, transform=transform
        )
    else:
        val_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=transform
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["val_samples_kl"],
        shuffle=False,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=True
    )
    
    # 3. Create cold with auto-tuning
    cold_noise_std = config["initialization"]["cold_noise_std"]
    model_cold = create_cold_initialization(model_star, cold_noise_std, seed + 1, device)
    kl_cold = compute_kl_divergence(model_cold, model_star, val_loader, device)
    
    if kl_cold >= 0.5:
        print(f"   KL(cold)={kl_cold:.4f} >= 0.5, auto-tuning...")
        model_cold, cold_noise_std = auto_tune_cold_noise(
            model_star, MLP, config, val_loader, device, seed=seed
        )
        kl_cold = compute_kl_divergence(model_cold, model_star, val_loader, device)
    
    # 4. Create hot
    model_hot = create_hot_initialization(MLP, config, seed + 2, device)
    kl_hot = compute_kl_divergence(model_hot, model_star, val_loader, device)
    
    # 5. Validation
    print("\n=> Initialization validation:")
    print(f"   KL(cold || star) = {kl_cold:.4f} (required < 0.5)")
    print(f"   KL(hot || star)  = {kl_hot:.4f} (required > 5.0)")
    
    assert kl_cold < 0.5, f"COLD init failed: KL={kl_cold:.4f} >= 0.5"
    assert kl_hot > 5.0, f"HOT init failed: KL={kl_hot:.4f} <= 5.0"
    
    print("=> All initializations validated OK\n")
    
    return model_star, model_cold, model_hot
