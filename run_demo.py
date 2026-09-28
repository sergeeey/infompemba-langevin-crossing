"""
Скрипт для запуска небольшого эксперимента (демо-режим).

Использует уменьшенные параметры для быстрой проверки:
- 10 траекторий вместо 1000
- 200 итераций вместо 2000

Запуск:
    python run_demo.py
"""

import sys
import os

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.mpemba_runner import run_experiment
import yaml


def main():
    """Запуск демо-эксперимента"""
    print("="*70)
    print("InfoMpemba: Демо-режим (10 траекторий, 200 итераций)")
    print("="*70)
    
    # Загружаем конфигурацию
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Уменьшаем параметры для демо
    config["experiment"]["num_runs"] = 2
    config["experiment"]["max_iterations"] = 50
    config["experiment"]["eval_interval"] = 10
    config["experiment"]["name"] = "mpemba_demo"
    
    # Для демо используем VanillaSGD (быстрее)
    config["optimizer"]["type"] = "VanillaSGD"
    
    print(f"\nКонфигурация:")
    print(f"  Оптимизатор: {config['optimizer']['type']}")
    print(f"  Траекторий: {config['experiment']['num_runs']}")
    print(f"  Итераций: {config['experiment']['max_iterations']}")
    print(f"  Learning rate: {config['optimizer']['lr']}")
    print(f"  Batch size: {config['data']['batch_size']}")
    print()
    
    # Запускаем эксперимент
    try:
        df_results = run_experiment(config)
        
        print("\n" + "="*70)
        print("✓ ДЕМО-ЭКСПЕРИМЕНТ ЗАВЕРШЕН УСПЕШНО")
        print("="*70)
        print(f"\nРезультаты сохранены в: results/trajectories.parquet")
        print(f"Всего записей: {len(df_results)}")
        print(f"\nДля анализа запустите:")
        print(f"  jupyter notebook analysis/Mpemba_Crossing_Analysis.ipynb")
        
    except Exception as e:
        print(f"\n✗ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
