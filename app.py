import pickle

from flask import Flask, request, jsonify, render_template

import numpy as np


app = Flask(__name__)


# Load the trained model
model = pickle.load(open('regmodel.pkl', 'rb'))

# Load the scaler
scalar = pickle.load(open('scaling.pkl', 'rb'))


# Home page
@app.route('/')
def home():
    return render_template('home.html')


# API prediction
@app.route('/predict_api', methods=['POST'])
def predict_api():

    data = request.json['data']

    print(data)

    # Keep the features in the correct order
    features = [
        data['MedInc'],
        data['HouseAge'],
        data['AveRooms'],
        data['AveBedrms'],
        data['Population'],
        data['AveOccup'],
        data['Latitude'],
        data['Longitude']
    ]

    final_input = np.array(features).reshape(1, -1)

    print(final_input)

    # Scale the input
    new_data = scalar.transform(final_input)

    # Make prediction
    output = model.predict(new_data)

    print(output[0])

    return jsonify(output[0])


# HTML form prediction
@app.route('/predict', methods=['POST'])
def predict():

    # Get values from HTML form
    data = [
        float(request.form['MedInc']),
        float(request.form['HouseAge']),
        float(request.form['AveRooms']),
        float(request.form['AveBedrms']),
        float(request.form['Population']),
        float(request.form['AveOccup']),
        float(request.form['Latitude']),
        float(request.form['Longitude'])
    ]

    # Convert to NumPy array
    final_input = np.array(data).reshape(1, -1)

    print("Input:", final_input)

    # Scale input
    input_data = scalar.transform(final_input)

    print("Scaled input:", input_data)

    # Prediction
    output = model.predict(input_data)[0]

    print("Prediction:", output)

    return render_template(
        'home.html',
        prediction_text=f"The House Price Prediction is {output:.2f}"
    )


if __name__ == '__main__':
    app.run(debug=True)
