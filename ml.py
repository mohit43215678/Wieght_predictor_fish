import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle

# Load the dataset
data = pd.read_csv('Fish.csv')

# Create a OneHotEncoder
encoder = OneHotEncoder(drop='first')

# Fit the encoder and transform the 'Species' column in the training data
species_encoded = encoder.fit_transform(data['Species'].values.reshape(-1, 1)).toarray()

# Drop the 'Species' column from the original data
data = data.drop('Species', axis=1)

# Concatenate the original data with the encoded 'Species' data
data = np.concatenate([data, species_encoded], axis=1)

# Split the data into features and target
X = data[:, 1:]  # All columns except 'Weight'
y = data[:, 0]  # 'Weight' column

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model and the encoder
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(encoder, open('encoder.pkl', 'wb'))
