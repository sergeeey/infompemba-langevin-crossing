"""
Модуль 4: Статистический контроллер (mpemba_runner.py)

Запускает N независимых траекторий для "холодного" и "горячего" старта.
Собирает и сохраняет результаты в формате Parquet.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict, List, Optional
import yaml
import argparse
import os
import sys
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# Добавляем родительскую директорию в путь для импортов
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.initializers import create_model, initialize_all, MLP
from src.langevin_sgd import LangevinSGD, clone_model_weights
from src.fisher_metrics import FisherMetricAnalyzer, create_metric_analyzer


def setup_deterministic(seed: int, deterministic: bool = True) -> None:
    """
    Устанавливает детерминированное поведение для воспроизводимости.

    Args:
        seed: Сид для генератора случайных чисел
        deterministic: Включить строгую детерминированность
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """
    Создает загрузчики для обучающих и валидационных данных.

    Args:
        config: Конфигурация эксперимента

    Returns:
        Кортеж (train_loader, val_loader)
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),  # FashionMNIST
        ]
    )

    dataset_name = config["data"]["dataset"]
    data_dir = config["data"]["data_dir"]
    batch_size = config["data"]["batch_size"]
    num_workers = config["hardware"]["num_workers"]

    # Обучающий набор
    if dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(
            data_dir, train=True, download=True, transform=transform
        )
        val_dataset = datasets.FashionMNIST(
            data_dir, train=False, download=True, transform=transform
        )
    elif dataset_name == "MNIST":
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Неподдерживаемый датасет: {dataset_name}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["val_samples_kl"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def run_single_trajectory(
    init_weights: Dict[str, torch.Tensor],
    train_loader: DataLoader,
    config: dict,
    analyzer: FisherMetricAnalyzer,
    device: torch.device,
    run_id: int,
    init_type: str,
) -> List[Dict]:
    """
    Запускает одну траекторию обучения.

    Args:
        init_weights: Начальные веса модели
        train_loader: Загрузчик обучающих данных
        config: Конфигурация
        analyzer: Анализатор метрик
        device: Устройство
        run_id: ID запуска
        init_type: Тип инициализации ("hot" или "cold")

    Returns:
        Список словарей с метриками для каждого шага
    """
    # Создаем модель с начальными весами
    model = create_model(config).to(device)
    model.load_state_dict(copy.deepcopy(init_weights))
    model.train()

    # Создаем симулятор
    runner = LangevinSGD(model=model, train_loader=train_loader, config=config, device=device)

    # Callback для вычисления метрик
    def eval_callback(model, step):
        return analyzer.compute_all_metrics(model, step)

    # Запускаем траекторию
    metrics_history = runner.run_trajectory(
        max_iterations=config["experiment"]["max_iterations"],
        eval_interval=config["experiment"]["eval_interval"],
        eval_callback=eval_callback,
    )

    # Форматируем результаты
    # Метрики вычисляются только каждые eval_interval шагов
    results = []

    # Проверяем, есть ли вычисленные метрики
    has_metrics = "kl_div_star" in metrics_history and len(metrics_history["kl_div_star"]) > 0

    if has_metrics:
        num_evals = len(metrics_history["kl_div_star"])
        for i in range(num_evals):
            record = {
                "run_id": run_id,
                "init_type": init_type,
                "step": metrics_history["step"][i],
                "loss_train": metrics_history["loss_train"][i],
                "kl_div_star": metrics_history["kl_div_star"][i],
                "trace_fisher": metrics_history["trace_fisher"][i],
                "info_temp": metrics_history["info_temp"][i],
            }
            results.append(record)
    else:
        # Только loss, без метрик
        for i in range(len(metrics_history["step"])):
            record = {
                "run_id": run_id,
                "init_type": init_type,
                "step": metrics_history["step"][i],
                "loss_train": metrics_history["loss_train"][i],
                "kl_div_star": float("nan"),
                "trace_fisher": float("nan"),
                "info_temp": float("nan"),
            }
            results.append(record)

    return results


def run_experiment(config: dict) -> pd.DataFrame:
    """
    Запускает полный эксперимент с множеством траекторий.

    Args:
        config: Конфигурация эксперимента

    Returns:
        DataFrame с результатами всех траекторий
    """
    # Настройка устройства
    device = torch.device(config["hardware"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Создаем датасеты
    print("\n=> Загрузка данных...")
    train_loader, val_loader = create_dataloaders(config)

    # Инициализация моделей
    print("\n=> Инициализация моделей...")
    base_seed = config["experiment"]["base_seed"]
    setup_deterministic(base_seed, config["hardware"]["deterministic"])

    model_star, model_cold, model_hot = initialize_all(config, device, seed=base_seed)

    # Создаем анализатор метрик
    analyzer = create_metric_analyzer(
        model_star=model_star, val_loader=val_loader, config=config, device=device
    )

    # Параметры эксперимента
    num_runs = config["experiment"]["num_runs"]
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    parquet_path = os.path.join(results_dir, config["output"]["parquet_file"])
    cooldown = config["hardware"].get("cooldown_seconds", 0)

    # Проверяем, есть ли уже сохранённые результаты (resume после сбоя)
    completed_runs = set()
    if os.path.exists(parquet_path):
        df_existing = pd.read_parquet(parquet_path)
        for _, grp in df_existing.groupby(["init_type", "run_id"]):
            completed_runs.add((grp["init_type"].iloc[0], int(grp["run_id"].iloc[0])))
        all_results = df_existing.to_dict("records")
        print(f"   => Найдено {len(completed_runs)} завершённых траекторий, продолжаем...")

    # Запускаем траектории
    print(f"\n=> Запуск {num_runs} траекторий для каждого типа инициализации...")
    start_time = time.time()

    for init_type, init_model in [("cold", model_cold), ("hot", model_hot)]:
        print(f"\n{'=' * 60}")
        print(f"Инициализация: {init_type.upper()}")
        print(f"{'=' * 60}")

        init_weights = init_model.state_dict()

        for run_idx in tqdm(range(num_runs), desc=f"Траектории {init_type}"):
            # Пропускаем уже завершённые траектории (resume)
            if (init_type, run_idx) in completed_runs:
                continue

            run_seed = base_seed + run_idx + (0 if init_type == "cold" else 10000)
            setup_deterministic(run_seed, config["hardware"]["deterministic"])

            # Запускаем одну траекторию
            trajectory_results = run_single_trajectory(
                init_weights=init_weights,
                train_loader=train_loader,
                config=config,
                analyzer=analyzer,
                device=device,
                run_id=run_idx,
                init_type=init_type,
            )

            all_results.extend(trajectory_results)

            # Инкрементальное сохранение после каждой траектории
            pd.DataFrame(all_results).to_parquet(parquet_path, index=False)

            # Освобождаем GPU-память
            torch.cuda.empty_cache()

            # Пауза для снижения нагрузки на GPU
            if cooldown > 0:
                time.sleep(cooldown)

            # Логируем прогресс
            if (run_idx + 1) % config["output"]["log_interval"] == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (run_idx + 1) * (num_runs - run_idx - 1)
                print(
                    f"   [{init_type}] Запущено {run_idx + 1}/{num_runs}, ETA: {eta / 60:.1f} мин"
                )

    total_time = time.time() - start_time
    print(f"\n=> Эксперимент завершен за {total_time / 60:.1f} минут")

    # Финальное сохранение
    df_results = pd.DataFrame(all_results)
    df_results.to_parquet(parquet_path, index=False)
    print(f"   => Результаты сохранены в {parquet_path}")
    print(f"   => Всего записей: {len(df_results)}")

    return df_results


def main():
    """
    Главная функция для запуска из командной строки.
    """
    parser = argparse.ArgumentParser(
        description="InfoMpemba: Верификация эффекта Мпемба в нейронных сетях"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Путь к файлу конфигурации"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=None,
        help="Переопределить количество запусков из конфигурации",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Переопределить максимальное количество итераций",
    )

    args = parser.parse_args()

    # Загружаем конфигурацию
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Переопределения из командной строки
    if args.num_runs is not None:
        config["experiment"]["num_runs"] = args.num_runs
    if args.max_iterations is not None:
        config["experiment"]["max_iterations"] = args.max_iterations

    # Валидация конфигурации
    assert config["optimizer"]["type"] in ["VanillaSGD", "LangevinFisher"], (
        f"Неподдерживаемый тип оптимизатора: {config['optimizer']['type']}"
    )
    assert config["model"]["activation"] == "tanh", (
        "Для гладкости Фишера требуется активация 'tanh'"
    )

    print("=" * 60)
    print("InfoMpemba: Верификация эффекта Мпемба")
    print("=" * 60)
    print(f"Эксперимент: {config['experiment']['name']}")
    print(f"Оптимизатор: {config['optimizer']['type']}")
    print(f"Количество запусков: {config['experiment']['num_runs']}")
    print(f"Итераций на запуск: {config['experiment']['max_iterations']}")
    print("=" * 60)

    # Запускаем эксперимент
    df_results = run_experiment(config)

    # Краткая статистика
    print("\n" + "=" * 60)
    print("Краткая статистика:")
    print("=" * 60)

    for init_type in ["cold", "hot"]:
        mask = df_results["init_type"] == init_type
        df_subset = df_results[mask]

        final_steps = df_subset.groupby("run_id").last().reset_index()

        print(f"\n{init_type.upper()}:")
        print(f"  Средний KL: {final_steps['kl_div_star'].mean():.6f}")
        print(f"  Медианный KL: {final_steps['kl_div_star'].median():.6f}")
        print(f"  След Фишера (средний): {final_steps['trace_fisher'].mean():.6f}")

    print("\n=> Готово!")


if __name__ == "__main__":
    main()
