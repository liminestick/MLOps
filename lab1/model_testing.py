import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import os


def test_model():
    # Проверка наличия тестовых данных
    test_csv = 'data/test_data.csv'
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"File {test_csv} not found. Run data_preprocessing.py first.")

    # Загрузка тестовых данных
    test_data = pd.read_csv(test_csv)

    # Разделение на признаки (X) и целевую переменную (y)
    X_test = test_data.drop(columns=['target'])
    y_test = test_data['target']

    # Проверка наличия модели
    model_filename = 'model/logistic_regression_model.pkl'
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"File {model_filename} not found. Run model_preparation.py first.")

    # Загрузка модели
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)

    # Выполнение предсказаний
    y_pred = loaded_model.predict(X_test)

    # Вычисление точности модели
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model test accuracy is: {accuracy:.3f}")


if __name__ == "__main__":
    test_model()