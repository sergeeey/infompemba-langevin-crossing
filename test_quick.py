"""
Быстрый тест для проверки работоспособности всех модулей.

Запуск:
    python test_quick.py
"""

import torch
import sys
import os

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.initializers import create_model, MLP
from src.langevin_sgd import LangevinSGD
from src.fisher_metrics import FisherMetricAnalyzer
from torch.utils.data import DataLoader, TensorDataset


def test_model_creation():
    """Тест создания модели"""
    print("Тест 1: Создание модели...")
    
    config = {
        "model": {
            "layers": [784, 256, 128, 10],
            "activation": "tanh",
            "bias": True
        }
    }
    
    model = create_model(config)
    assert isinstance(model, MLP)
    
    # Проверяем архитектуру
    total_params = sum(p.numel() for p in model.parameters())
    expected = 784*256 + 256 + 256*128 + 128 + 128*10 + 10
    assert total_params == expected, f"Ожидалось {expected}, получено {total_params}"
    
    print("  ✓ Модель создана корректно")
    return model


def test_forward_pass(model):
    """Тест прямого прохода"""
    print("Тест 2: Прямой проход...")
    
    x = torch.randn(32, 784)
    output = model(x)
    
    assert output.shape == (32, 10), f"Неверная форма выхода: {output.shape}"
    print("  ✓ Прямой проход работает")


def test_langevin_step(model):
    """Тест шага Ланжевена"""
    print("Тест 3: Шаг оптимизатора...")
    
    # Создаем фейковые данные
    x = torch.randn(16, 784)
    y = torch.randint(0, 10, (16,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=16)
    
    config = {
        "optimizer": {
            "type": "VanillaSGD",
            "lr": 0.01,
            "noise_scale": 1.0
        },
        "data": {
            "batch_size": 16
        }
    }
    
    optimizer = LangevinSGD(model, loader, config, torch.device("cpu"))
    loss = optimizer.step()
    
    assert isinstance(loss, float)
    assert loss > 0
    print(f"  ✓ Шаг выполнен, loss = {loss:.4f}")


def test_fisher_metrics(model):
    """Тест вычисления метрик Фишера"""
    print("Тест 4: Метрики Фишера...")
    
    # Фейковые данные для тестов
    x = torch.randn(64, 784)
    y = torch.randint(0, 10, (64,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=64)
    
    config = {
        "data": {
            "val_samples_kl": 64
        },
        "model": {
            "layers": [784, 256, 128, 10],
            "activation": "tanh",
            "bias": True
        }
    }
    
    # Создаем "эталонную" модель
    model_star = create_model({"model": config["model"]})
    
    analyzer = FisherMetricAnalyzer(model_star, loader, config, torch.device("cpu"))
    
    # KL-дивергенция
    kl = analyzer.compute_kl_divergence(model)
    assert isinstance(kl, float)
    assert kl >= 0
    print(f"  ✓ KL-дивергенция: {kl:.4f}")
    
    # След Фишера
    trace = analyzer.compute_fisher_trace(model)
    assert isinstance(trace, float)
    assert trace >= 0
    print(f"  ✓ След Фишера: {trace:.4f}")


def main():
    """Запуск всех тестов"""
    print("="*60)
    print("Быстрый тест модулей InfoMpemba")
    print("="*60 + "\n")
    
    try:
        model = test_model_creation()
        test_forward_pass(model)
        test_langevin_step(model)
        test_fisher_metrics(model)
        
        print("\n" + "="*60)
        print("✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
