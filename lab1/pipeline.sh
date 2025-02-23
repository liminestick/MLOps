#!/bin/bash

# Убедитесь, что вы находитесь в правильной директории
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

# Активация виртуального окружения (используем абсолютный путь)
VENV_PATH="/mnt/c/Users/limin/PycharmProjects/MLOps/.venv"  # Замените на ваш абсолютный путь
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Шаг 1: Создание данных
echo "Running data_creation.py..."
python data_creation.py
if [ $? -ne 0 ]; then
    echo "Error: data_creation.py failed."
    exit 1
fi

# Шаг 2: Предобработка данных
echo "Running data_preprocessing.py..."
python data_preprocessing.py
if [ $? -ne 0 ]; then
    echo "Error: data_preprocessing.py failed."
    exit 1
fi

# Шаг 3: Обучение модели
echo "Running model_preparation.py..."
python model_preparation.py
if [ $? -ne 0 ]; then
    echo "Error: model_preparation.py failed."
    exit 1
fi

# Шаг 4: Тестирование модели
echo "Running model_testing.py..."
python model_testing.py
if [ $? -ne 0 ]; then
    echo "Error: model_testing.py failed."
    exit 1
fi

echo "Pipeline completed successfully!"