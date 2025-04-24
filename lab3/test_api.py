import requests

def test_api(url, text):
    """
    Отправляет запрос к API и возвращает результат.

    :param url: URL вашего микросервиса (например, http://localhost:5000/predict)
    :param text: Текст для анализа
    :return: Ответ от API
    """
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Проверка статуса ответа
        result = response.json()
        return result
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе: {e}")
        return None


if __name__ == "__main__":
    # URL вашего микросервиса
    api_url = "http://localhost:5000/predict"

    # Примеры текстов для тестирования
    test_texts = [
        "I love this product",
        "This is the worst experience ever",
        "The service was okay",
        "I hate this",
        "Best movie I've ever seen"
    ]

    print("Тестирование API:\n")
    for text in test_texts:
        result = test_api(api_url, text)
        if result:
            sentiment = result.get("sentiment", "N/A")
            print(f"Текст: \"{text}\" -> Сентимент: {sentiment}")