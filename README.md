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
