import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Простые данные для обучения
texts = [
    "I love this product", "Best experience ever",
    "I hate this", "Terrible service"
]
labels = [1, 1, 0, 0]  # 1 - положительный, 0 - отрицательный

# Создание пайплайна: векторизация + логистическая регрессия
model = make_pipeline(TfidfVectorizer(), LogisticRegression())

# Обучение модели
model.fit(texts, labels)

# Сохранение модели
joblib.dump(model, "text_classifier.joblib")