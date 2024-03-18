from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Load the Model and the Encoder
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the values from the form
    species = request.form['Species']
    length1 = float(request.form['length1'])
    length2 = float(request.form['length2'])
    length3 = float(request.form['length3'])
    height = float(request.form['height'])
    width = float(request.form['width'])

    # Apply the encoder to the 'Species' column of the input data
    species_encoded = encoder.transform(np.array([species]).reshape(-1, 1)).toarray()[0]

    # Combine all the features
    features = [length1, length2, length3, height, width] + list(species_encoded)

    final_features = np.array([features])
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Estimated Weight: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
