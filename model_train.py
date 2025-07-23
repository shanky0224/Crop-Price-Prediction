# model_train.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('crops_muj.csv')

# Clean column names
data.columns = data.columns.str.strip()

# Define features and target
X = data[['Year', 'Crop', 'Month', 'City', 'State']]
y = data['Price']

# Check for NaN values in y and drop corresponding rows in X and y
if y.isna().sum() > 0:
    print(f"Found {y.isna().sum()} missing values in target variable 'Price'. Removing these rows.")
    X = X[~y.isna()]
    y = y.dropna()

# Initialize Label Encoders
crop_encoder = LabelEncoder()
month_encoder = LabelEncoder()
city_encoder = LabelEncoder()
state_encoder = LabelEncoder()

# Fit the Label Encoders on the entire dataset
X['Crop'] = crop_encoder.fit_transform(X['Crop'])
X['Month'] = month_encoder.fit_transform(X['Month'])
X['City'] = city_encoder.fit_transform(X['City'])
X['State'] = state_encoder.fit_transform(X['State'])

# Initialize the scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Save the model, encoders, and scaler
with open('crop_price_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('crop_encoder.pkl', 'wb') as file:
    pickle.dump(crop_encoder, file)
with open('month_encoder.pkl', 'wb') as file:
    pickle.dump(month_encoder, file)
with open('city_encoder.pkl', 'wb') as file:
    pickle.dump(city_encoder, file)
with open('state_encoder.pkl', 'wb') as file:
    pickle.dump(state_encoder, file)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("Model, encoders, and scaler saved.")

