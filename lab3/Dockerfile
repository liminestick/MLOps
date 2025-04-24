# Базовый образ Python 3.9
FROM python:3.9-slim

# Установка рабочей директории внутри контейнера
WORKDIR /app

# Копирование зависимостей (requirements.txt) в контейнер
COPY requirements.txt .

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование остальных файлов проекта в контейнер
COPY . .

# Экспонирование порта 5000 (порт, на котором работает Flask)
EXPOSE 5000

# Команда для запуска приложения
CMD ["python", "app.py"]