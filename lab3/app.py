from flask import Flask, request, jsonify
import joblib

# Загрузка модели
model = joblib.load("text_classifier.joblib")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    prediction = model.predict([text])[0]
    return jsonify({'sentiment': 'positive' if prediction == 1 else 'negative'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)