from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load saved model and scalers
model = pickle.load(open('model.pkl', 'rb'))
minmax = pickle.load(open('minmaxscaler.pkl', 'rb'))
stand = pickle.load(open('standscaler.pkl', 'rb'))

# Reverse mapping dictionary
df_dict = {
    "rice": 1, "maize": 2, "jute": 3, "cotton": 4, "coconut": 5, "papaya": 6,
    "orange": 7, "apple": 8, "muskmelon": 9, "watermelon": 10, "grapes": 11,
    "mango": 12, "banana": 13, "pomegranate": 14, "lentil": 15, "blackgram": 16,
    "mungbean": 17, "mothbeans": 18, "pigeonpeas": 19, "kidneybeans": 20,
    "chickpea": 21, "coffee": 22
}
reverse_dict = {v: k for k, v in df_dict.items()}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Prepare data for prediction
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        scaled = minmax.transform(features)
        scaled = stand.transform(scaled)

        prediction = model.predict(scaled)[0]
        crop_name = reverse_dict.get(prediction, "Unknown")

        return render_template('result.html', crop=crop_name)
    except Exception as e:
        return render_template('result.html', crop="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
