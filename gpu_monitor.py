"""
Скрипт мониторинга GPU в реальном времени.

Отображает температуру, использование GPU, потребление энергии
и использование VRAM каждые 2 секунды.

Запуск:
    python gpu_monitor.py
    
Или в отдельном окне PowerShell:
    python gpu_monitor.py --watch
"""

import subprocess
import sys
import time
from datetime import datetime


def get_gpu_info():
    """
    Получает информацию о GPU через nvidia-smi.
    
    Returns:
        Словарь с параметрами GPU или None при ошибке
    """
    try:
        query = (
            "temperature.gpu,"
            "utilization.gpu,"
            "utilization.memory,"
            "memory.used,"
            "memory.total,"
            "power.draw,"
            "power.max_limit,"
            "clocks.gr"
        )
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode != 0:
            return None
        
        values = result.stdout.strip().split(", ")
        
        return {
            "temp": float(values[0]),
            "gpu_util": float(values[1]),
            "mem_util": float(values[2]),
            "mem_used": float(values[3]),
            "mem_total": float(values[4]),
            "power": float(values[5]),
            "power_max": float(values[6]),
            "clock": float(values[7]),
        }
    
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
        return None


def format_bar(value, max_value, width=30):
    """Форматирует прогресс-бар."""
    filled = int(width * value / max_value) if max_value > 0 else 0
    return "█" * filled + "░" * (width - filled)


def monitor_loop(refresh_interval=2):
    """
    Цикл непрерывного мониторинга.
    
    Args:
        refresh_interval: Интервал обновления (секунды)
    """
    print("=" * 70)
    print("GPU Monitor (Ctrl+C для выхода)")
    print("=" * 70)
    print()
    
    try:
        while True:
            info = get_gpu_info()
            
            if info is None:
                print("\r⚠ Не удалось получить данные GPU...", end="", flush=True)
                time.sleep(refresh_interval)
                continue
            
            now = datetime.now().strftime("%H:%M:%S")
            
            # Температура
            temp = info["temp"]
            if temp < 60:
                temp_icon = "❄️"
            elif temp < 70:
                temp_icon = "✅"
            elif temp < 80:
                temp_icon = "⚠️"
            else:
                temp_icon = "🔥"
            
            # VRAM
            vram_pct = info["mem_used"] / info["mem_total"] * 100
            vram_bar = format_bar(info["mem_used"], info["mem_total"])
            
            # GPU Util
            gpu_bar = format_bar(info["gpu_util"], 100)
            
            # Power
            power_pct = info["power"] / info["power_max"] * 100
            
            # Форматируем вывод
            output = (
                f"\r{now} | "
                f"{temp_icon} Temp: {temp:.0f}°C | "
                f"GPU: {info['gpu_util']:.0f}% [{gpu_bar}] | "
                f"VRAM: {info['mem_used']:.0f}/{info['mem_total']:.0f} MB ({vram_pct:.0f}%) [{vram_bar}] | "
                f"Power: {info['power']:.0f}/{info['power_max']:.0f} W ({power_pct:.0f}%) | "
                f"Clock: {info['clock']:.0f} MHz"
            )
            
            # Очищаем строку и выводим
            print(f"\r{' ' * 120}", end="")
            print(output, end="", flush=True)
            
            time.sleep(refresh_interval)
    
    except KeyboardInterrupt:
        print("\n\n✅ Мониторинг остановлен")


def check_gpu():
    """Проверяет доступность GPU и выводит информацию."""
    info = get_gpu_info()
    
    if info is None:
        print("❌ GPU не обнаружен или nvidia-smi недоступен")
        return False
    
    print("=" * 70)
    print("Информация о GPU")
    print("=" * 70)
    print(f"  Температура:     {info['temp']:.0f}°C")
    print(f"  Загрузка GPU:    {info['gpu_util']:.0f}%")
    print(f"  Загрузка памяти: {info['mem_util']:.0f}%")
    print(f"  VRAM:            {info['mem_used']:.0f} / {info['mem_total']:.0f} MB")
    print(f"  Энергопотребление: {info['power']:.0f} / {info['power_max']:.0f} W ({info['power']/info['power_max']*100:.0f}%)")
    print(f"  Частота GPU:     {info['clock']:.0f} MHz")
    print("=" * 70)
    
    return True


def main():
    """Главная функция."""
    if "--check" in sys.argv:
        success = check_gpu()
        sys.exit(0 if success else 1)
    else:
        monitor_loop()


if __name__ == "__main__":
    main()
