# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the trained model, encoders, and scaler
model_path = 'crop_price_model.pkl'
crop_encoder_path = 'crop_encoder.pkl'
month_encoder_path = 'month_encoder.pkl'
city_encoder_path = 'city_encoder.pkl'
state_encoder_path = 'state_encoder.pkl'
scaler_path = 'scaler.pkl'


with open(model_path, 'rb') as file:
    model = pickle.load(file)
with open(crop_encoder_path, 'rb') as file:
    crop_encoder = pickle.load(file)
with open(month_encoder_path, 'rb') as file:
    month_encoder = pickle.load(file)
with open(city_encoder_path, 'rb') as file:
    city_encoder = pickle.load(file)
with open(state_encoder_path, 'rb') as file:
    state_encoder = pickle.load(file)
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received Data:", data)

        # Extract input values
        year = data.get('year')
        crop = data.get('crop')
        month = data.get('month')
        city = data.get('city')
        state = data.get('state')
        
        print("Extracted Data:", year, crop, month, city, state)

        # Encode input values
        year_encoded = int(year)
        crop_encoded = encode_input(crop, crop_encoder, 'crop')
        month_encoded = encode_input(month, month_encoder, 'month')
        city_encoded = encode_input(city, city_encoder, 'city')
        state_encoded = encode_input(state, state_encoder, 'state')

        print("Encoded Values:", year_encoded, crop_encoded, month_encoded, city_encoded, state_encoded)

        # Prepare input for prediction
        input_data = np.array([[year_encoded, crop_encoded, month_encoded, city_encoded, state_encoded]])

        # Apply scaling
        input_data_scaled = scaler.transform(input_data)

        # Make prediction using the scaled input
        predicted_price = model.predict(input_data_scaled)[0]
        print("Predicted Price:", predicted_price)

        # Return the predicted price
        return jsonify({'price': float(predicted_price)})

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)})


def encode_input(value, encoder, name):
    """Encodes the input value using the provided encoder, returns -1 if value is invalid."""
    if value not in encoder.classes_:
        logging.warning(f"Invalid {name}: {value}")
        return -1
    return encoder.transform([value])[0]

if __name__ == '__main__':
    app.run(debug=True)