# Лабораторные работы
## Лабораторная работа №1
<details>

###Цель:
*Создать автоматизированный пайплайн для обработки данных, обучения и тестирования модели машинного обучения.
###Этапы:
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