import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def preprocess_data():
    # Загрузка данных из CSV
    csv_filename = 'data/iris_data.csv'
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"File {csv_filename} not found. Run data_creation.py first.")

    data = pd.read_csv(csv_filename)

    # Удаление строк с пропущенными значениями
    data = data.dropna()

    # Разделение на признаки (X) и целевую переменную (y)
    X = data.drop(columns=['target'])  # Все столбцы, кроме 'target'
    y = data['target']  # Целевой столбец

    # Проверка целевой переменной на наличие NaN
    if y.isnull().any():
        print("Error: Target variable contains NaN values after preprocessing!")
        return

    # Стандартизация признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Создание DataFrame для обучающих данных
    train_data = pd.DataFrame(X_train, columns=data.columns[:-1])
    train_data['target'] = y_train.values  # Используем .values для корректного соответствия индексов

    # Создание DataFrame для тестовых данных
    test_data = pd.DataFrame(X_test, columns=data.columns[:-1])
    test_data['target'] = y_test.values  # Используем .values для корректного соответствия индексов

    # Сохранение предобработанных данных
    train_data.to_csv('data/train_data.csv', index=False)
    test_data.to_csv('data/test_data.csv', index=False)

    print("Data preprocessing completed successfully.")


if __name__ == "__main__":
    preprocess_data()