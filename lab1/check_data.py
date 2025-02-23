import pandas as pd

# Проверка train_data.csv
train_data = pd.read_csv('data/train_data.csv')
print(train_data.isnull().sum())

# Проверка test_data.csv
test_data = pd.read_csv('data/test_data.csv')
print(test_data.isnull().sum())