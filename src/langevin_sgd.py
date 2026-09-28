"""
Модуль 2: Симулятор SGD-Ланжевена (langevin_sgd.py)

Реализует цикл обучения с фиксацией траектории.
Поддерживает два режима:
- VanillaSGD: Стандартный стохастический градиентный спуск
- LangevinFisher: Ланжевеновский SGD с метрикой Фишера
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import copy
import math


class LangevinSGD:
    """
    Симулятор Ланжевеновского SGD с опциональной метрикой Фишера.

    Args:
        model: Модель для обучения
        train_loader: Загрузчик обучающих данных
        config: Словарь конфигурации эксперимента
        device: Устройство для вычислений
    """

    def __init__(
        self, model: nn.Module, train_loader: DataLoader, config: dict, device: torch.device
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.config = config
        self.device = device

        self.optimizer_type = config["optimizer"]["type"]
        self.lr = config["optimizer"]["lr"]
        self.batch_size = config["data"]["batch_size"]
        self.noise_scale = config["optimizer"].get("noise_scale", 1.0)

        # Mixed Precision (AMP)
        self.use_amp = config.get("hardware", {}).get("mixed_precision", False)
        self.scaler = (
            torch.cuda.amp.GradScaler() if (self.use_amp and self.device.type == "cuda") else None
        )

        # Итератор для бесконечной выборки батчей
        self.data_iter = iter(train_loader)

        # История обучения
        self.loss_history: List[float] = []
        self.step_count = 0

    def _get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Получает следующий батч данных (циклически).

        Returns:
            Кортеж (data, target)
        """
        try:
            data, target = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            data, target = next(self.data_iter)

        return data.to(self.device), target.to(self.device)

    def _compute_diagonal_fisher(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Вычисляет диагональ информационной матрицы Фишера.

        F = E[grad log p(y|x,theta) * grad log p(y|x,theta)^T]

        Для классификации используем аппроксимацию через градиенты logits.

        Args:
            data: Входные данные
            target: Целевые метки

        Returns:
            Список тензоров - диагональных элементов Фишера для каждого параметра
        """
        # Вычисляем логиты
        if data.dim() > 2:
            data = data.view(data.size(0), -1)

        logits = self.model(data)

        # Для классификации: F ≈ E[grad_theta log p(y|x) * grad_theta log p(y|x)^T]
        # Используем градиенты cross-entropy loss как аппроксимацию
        loss = F.cross_entropy(logits, target)

        # Вычисляем градиенты
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)

        # Диагональ Фишера ≈ квадрат градиента
        fisher_diag = [g.detach() ** 2 for g in grads]

        return fisher_diag

    def step_vanilla_sgd(self) -> float:
        """
        Выполает один шаг Vanilla SGD с опциональным AMP.

        Returns:
            Значение функции потерь на текущем батче
        """
        data, target = self._get_batch()

        if data.dim() > 2:
            data = data.view(data.size(0), -1)

        # Forward pass с опциональным AMP
        if self.use_amp and self.scaler is not None:
            with torch.cuda.amp.autocast():
                logits = self.model(data)
                loss = F.cross_entropy(logits, target)

            # Backward pass через scaler
            self.model.zero_grad()
            self.scaler.scale(loss).backward()

            # SGD update
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        # Дескейлим градиент
                        self.scaler.unscale_(self.model)
                        noise_std = math.sqrt(2.0 * self.lr / self.batch_size) * self.noise_scale
                        noise = torch.randn_like(param) * noise_std
                        param.data.sub_(self.lr * param.grad + noise)

            self.scaler.step(self.model.parameters())
            self.scaler.update()
        else:
            # Обычный forward/backward
            logits = self.model(data)
            loss = F.cross_entropy(logits, target)

            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        noise_std = math.sqrt(2.0 * self.lr / self.batch_size) * self.noise_scale
                        noise = torch.randn_like(param) * noise_std
                        param.data.sub_(self.lr * param.grad + noise)

        self.loss_history.append(loss.item())
        self.step_count += 1

        return loss.item()

    def step_langevin_fisher(self) -> float:
        """
        Выполняет один шаг Ланжевеновского SGD с метрикой Фишера.

        Обновление:
        theta_{t+1} = theta_t - lr * F^{-1} * grad + sqrt(2 * lr / B) * F^{-1/2} * noise

        Оптимизация: переиспользуем градиенты из первого backward,
        без второго forward/backward.

        Returns:
            Значение функции потерь на текущем батче
        """
        data, target = self._get_batch()

        if data.dim() > 2:
            data = data.view(data.size(0), -1)

        # Forward pass
        logits = self.model(data)
        loss = F.cross_entropy(logits, target)

        # Backward + вычисляем Fisher из тех же градиентов
        self.model.zero_grad()
        loss.backward()

        # Диагональ Фишера ≈ квадрат градиента (переиспользуем!)
        fisher_diag = []
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad)
                fisher_diag.append(param.grad.detach() ** 2)
            else:
                grads.append(None)
                fisher_diag.append(torch.zeros_like(param))

        # Langevin-Fisher update
        with torch.no_grad():
            for param, grad, fisher in zip(self.model.parameters(), grads, fisher_diag):
                if grad is None:
                    continue

                # Damping для стабильности natural gradient
                # WHY: fisher = grad², поэтому F^{-1}*grad = 1/grad — взрывается при малых градиентах
                # Damping 1.0 ограничивает усиление до разумных значений
                fisher_safe = fisher + 1.0

                # Естественный градиент: F^{-1} * grad
                natural_grad = grad / fisher_safe

                # Масштабированный шум: sqrt(2 * lr / B) * F^{-1/2} * noise
                noise_std = math.sqrt(2.0 * self.lr / self.batch_size) * self.noise_scale
                noise = torch.randn_like(param) * noise_std / torch.sqrt(fisher_safe)

                # Обновление
                param.data.sub_(self.lr * natural_grad + noise)

        self.loss_history.append(loss.item())
        self.step_count += 1

        return loss.item()

    def step(self) -> float:
        """
        Выполняет один шаг обучения (выбирает метод на основе конфигурации).

        Returns:
            Значение функции потерь на текущем батче
        """
        if self.optimizer_type == "VanillaSGD":
            return self.step_vanilla_sgd()
        elif self.optimizer_type == "LangevinFisher":
            return self.step_langevin_fisher()
        else:
            raise ValueError(f"Неподдерживаемый тип оптимизатора: {self.optimizer_type}")

    def run_trajectory(
        self, max_iterations: int, eval_interval: int = 10, eval_callback: Optional[callable] = None
    ) -> Dict[str, List]:
        """
        Запускает полную траекторию обучения.

        Args:
            max_iterations: Максимальное количество итераций
            eval_interval: Интервал вызова callback (в шагах)
            eval_callback: Функция для вычисления метрик (вызывается каждые eval_interval шагов)

        Returns:
            Словарь с историей метрик
        """
        from tqdm import tqdm

        metrics_history = {"step": [], "loss_train": []}
        metrics_initialized = False

        print(f"   => Запуск траектории: {self.optimizer_type}, {max_iterations} итераций")

        for step in tqdm(range(max_iterations), desc="   Обучение", leave=False):
            loss = self.step()

            # Вызываем callback для вычисления тяжелых метрик
            if eval_callback is not None and (step + 1) % eval_interval == 0:
                eval_metrics = eval_callback(self.model, step + 1)

                if not metrics_initialized:
                    # Инициализируем историю дополнительными метриками
                    for key in eval_metrics.keys():
                        metrics_history[key] = []
                    metrics_initialized = True

                metrics_history["step"].append(step + 1)
                metrics_history["loss_train"].append(loss)

                for key, value in eval_metrics.items():
                    metrics_history[key].append(value)
            else:
                metrics_history["step"].append(step + 1)
                metrics_history["loss_train"].append(loss)

        print(f"   => Траектория завершена (final loss: {metrics_history['loss_train'][-1]:.6f})")

        return metrics_history


def create_trajectory_runner(
    model: nn.Module, train_loader: DataLoader, config: dict, device: torch.device
) -> LangevinSGD:
    """
    Фабрика для создания симулятора траектории.

    Args:
        model: Модель для обучения
        train_loader: Загрузчик обучающих данных
        config: Конфигурация эксперимента
        device: Устройство

    Returns:
        Экземпляр LangevinSGD
    """
    return LangevinSGD(model=model, train_loader=train_loader, config=config, device=device)


def reset_model_weights(
    model: nn.Module, weights: Dict[str, torch.Tensor], device: torch.device
) -> nn.Module:
    """
    Сбрасывает веса модели к заданным значениям.

    Args:
        model: Модель для сброса
        weights: Словарь с весами (state_dict)
        device: Устройство

    Returns:
        Модель с загруженными весами
    """
    model.load_state_dict(weights)
    model.to(device)
    return model


def clone_model_weights(source: nn.Module, target: nn.Module, device: torch.device) -> nn.Module:
    """
    Копирует веса из одной модели в другую.

    Args:
        source: Исходная модель
        target: Целевая модель (будет перезаписана)
        device: Устройство

    Returns:
        Целевая модель с скопированными весами
    """
    target.load_state_dict(copy.deepcopy(source.state_dict()))
    target.to(device)
    return target
