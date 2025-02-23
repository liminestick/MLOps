import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle  # Используем pickle для сохранения модели
import os


def train_model():
    # Загрузка обучающих данных
    train_csv = 'data/train_data.csv'
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"File {train_csv} not found. Run data_preprocessing.py first.")

    train_data = pd.read_csv(train_csv)

    # Проверка наличия пропущенных значений
    if train_data.isnull().values.any():
        print("Error: Missing values found in the dataset!")
        print(train_data[train_data.isnull().any(axis=1)])  # Показать строки с NaN
        return

    # Разделение на признаки (X) и целевую переменную (y)
    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']

    # Проверка целевой переменной на наличие NaN
    if y_train.isnull().any():
        print("Error: Target variable contains NaN values!")
        return

    # Инициализация модели
    model = LogisticRegression(max_iter=200)

    # Обучение модели
    model.fit(X_train, y_train)

    # Сохранение модели с помощью pickle
    model_filename = 'model/logistic_regression_model.pkl'
    os.makedirs('model', exist_ok=True)  # Создаем папку model, если её нет

    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    print(f"Model successfully saved to {model_filename}")


if __name__ == "__main__":
    train_model()