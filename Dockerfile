FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Установка Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создание виртуального окружения
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Копирование зависимостей
COPY requirements.txt .

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Установка BackPACK из исходников
RUN pip install --no-cache-dir git+https://github.com/f-dangel/backpack.git

# Копирование кода проекта
COPY . .

# Создание директории для результатов
RUN mkdir -p results

# Переменная окружения для детерминированности
ENV PYTHONHASHSEED=42
ENV OMP_NUM_THREADS=1

# Точка входа
CMD ["python", "src/mpemba_runner.py", "--config", "configs/default.yaml"]
