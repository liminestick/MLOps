# Лабораторные работы
## Лабораторная работа №1
<details>

Цель: 
Создать автоматизированный пайплайн для обработки данных, обучения и тестирования модели машинного обучения.
Этапы:
1. Подготовить датасет Iris и сохранить в CSV.
2. Предобработать данные: стандартизация и разделение на train/test.
3. Обучить модель логистической регрессии.
4. Проверить точность модели на тестовых данных.
5. Автоматизировать процесс с помощью pipeline.sh.

```shell
(.venv) ayrat@AyratPC:/mnt/c/Users/limin/PycharmProjects/MLOps/lab1$ ./pipeline.sh
Running data_creation.py...
Data successfully saved to data/iris_data.csv
Running data_preprocessing.py...
Data preprocessing completed successfully.
Running model_preparation.py...
Model successfully saved to model/logistic_regression_model.pkl
Running model_testing.py...
Model test accuracy is: 1.000
Pipeline completed successfully!
```
</details>

## Лабораторная работа №2
<details>

Цель: 
Целью проекта было создание автоматизированного конвейера машинного обучения для решения задачи классификации изображений с использованием датасета CIFAR-10. Конвейер был разработан для демонстрации полного цикла работы с данными и моделями, включая загрузку данных, обучение модели и оценку её качества.
Этапы проекта
1. **Автоматизация процессов**:
   * Разработка скриптов Python для выполнения всех этапов (загрузка данных, обучение модели, оценка качества).
   * Настройка Jenkins для автоматизации выполнения этих этапов.

2. **Использование готовых инструментов**:
   * Применение предобученной модели ResNet18 для ускорения разработки.
   * Использование популярного датасета CIFAR-10 для тестирования.

3. **Оценка производительности**:
   * Проверка точности модели на тестовой выборке для подтверждения её эффективности.

4. **Воспроизводимость**:
   * Создание файла зависимостей (`requirements.txt`) для управления библиотеками.
   * Хранение всех скриптов и конфигураций в единой структуре, что позволяет легко воспроизвести проект на другом сервере.

```shell
Датасет CIFAR-10 успешно загружен.
Epoch 1, Loss: 0.5696378521945166
Epoch 2, Loss: 0.326714326734738
Epoch 3, Loss: 0.227498026783852
Epoch 4, Loss: 0.16253975077587016
Epoch 5, Loss: 0.12570661185976223
Обучение завершено.
Модель сохранена.
Accuracy: 89.74%
```
</details>

## Лабораторная работа №3
<details>

Цель:
Использовать полученные ранее знания по созданию микросервисов. Развернуть микросервис в контейнере докер. Модель машинного обучения для классификации текста, принимающая запрос по API и возвращающая ответ.
1. **Разработка Python-скриптов для обработки данных и обучения модели**:
   * Создан скрипт train_model.py для загрузки данных, обучения модели классификации текста на основе LogisticRegression и сохранения обученной модели..
   * Модель использует пайплайн из TfidfVectorizer для преобразования текста в числовые признаки и LogisticRegression для классификации.

2. **Создание микросервиса**:
   * Разработан Flask-микросервис (app.py) для приема POST-запросов с текстом и возвращения предсказания (позитивный/негативный сентимент)..
   * Микросервис интегрирует обученную модель через joblib.

3. **Контейнеризация с использованием Docker**:
   * Составлен Dockerfile для создания образа, содержащего все зависимости (Flask, scikit-learn, joblib) и код приложения.
   * Образ собран и запущен в контейнере, обеспечивающем изоляцию и переносимость приложения.

4. **Использование готовых инструментов**:
   * Применение библиотеки scikit-learn для создания и тренировки модели классификации текста.
   * Использование Docker для контейнеризации приложения, что обеспечивает воспроизводимость и удобство развертывания.

```shell
Тестирование API:

Текст: "I love this product" -> Сентимент: positive
Текст: "This is the worst experience ever" -> Сентимент: positive
Текст: "The service was okay" -> Сентимент: negative
Текст: "I hate this" -> Сентимент: negative
Текст: "Best movie I've ever seen" -> Сентимент: positive

Process finished with exit code 0
```
</details>
