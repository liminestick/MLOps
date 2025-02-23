import pandas as pd
from sklearn.datasets import load_iris


def create_data_csv():
    # Загрузка датасета Iris
    iris = load_iris()

    # Создание DataFrame
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['target'] = iris.target  # Добавляем целевой столбец

    # Проверка наличия пропущенных значений
    if data.isnull().values.any():
        print("Error: Missing values found in the original dataset!")
        return

    # Сохранение данных в CSV файл
    csv_filename = 'data/iris_data.csv'
    data.to_csv(csv_filename, index=False)
    print(f"Data successfully saved to {csv_filename}")


if __name__ == "__main__":
    create_data_csv()