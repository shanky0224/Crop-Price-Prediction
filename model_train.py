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



'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load the CSV file
data = pd.read_csv('crops_muj.csv')

# Clean column names (remove leading/trailing whitespace)
data.columns = data.columns.str.strip()

# Initialize Label Encoders for categorical variables
crop_encoder = LabelEncoder()
month_encoder = LabelEncoder()
city_encoder = LabelEncoder()
state_encoder = LabelEncoder()

# Fit the Label Encoders on the entire dataset
data['Crop'] = crop_encoder.fit_transform(data['Crop'])
data['Month'] = month_encoder.fit_transform(data['Month'])
data['City'] = city_encoder.fit_transform(data['City'])
data['State'] = state_encoder.fit_transform(data['State'])

# Split the data into features (X) and target variable (y)
X = data.drop('Price', axis=1)  # Features
y = data['Price']               # Target variable

# Check for NaN values in y and drop corresponding rows in X and y
if y.isna().sum() > 0:
    print(f"Found {y.isna().sum()} missing values in target variable 'Price'. Removing these rows.")
    X = X[~y.isna()]
    y = y.dropna()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Standard Scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform the training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Initialize the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the scaled training data
model.fit(X_train_scaled, y_train)

# Evaluate the model (Optional: You can print the R^2 score or other metrics)
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f'Training Score: {train_score}')
print(f'Testing Score: {test_score}')

# Save the trained model and encoders to the models directory
model_path = 'crop_price_model.pkl'
crop_encoder_path = 'crop_encoder.pkl'
month_encoder_path = 'month_encoder.pkl'
city_encoder_path = 'city_encoder.pkl'
state_encoder_path = 'state_encoder.pkl'
scaler_path = 'scaler.pkl'

with open(model_path, 'wb') as file:
    pickle.dump(model, file)
with open(crop_encoder_path, 'wb') as file:
    pickle.dump(crop_encoder, file)
with open(month_encoder_path, 'wb') as file:
    pickle.dump(month_encoder, file)
with open(city_encoder_path, 'wb') as file:
    pickle.dump(city_encoder, file)
with open(state_encoder_path, 'wb') as file:
    pickle.dump(state_encoder, file)
with open(scaler_path, 'wb') as file:
    pickle.dump(scaler, file)

print("Model and encoders saved successfully!")'''


'''import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Load the dataset
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

# Define the OneHotEncoder for the categorical columns
categorical_features = ['Crop', 'Month', 'City', 'State']
categorical_transformer = OneHotEncoder()

# Create a ColumnTransformer to apply the OneHotEncoder to categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')  # Pass 'Year' through without transformation

# Build the pipeline: preprocessing followed by linear regression
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),  # Scale all features after encoding
    ('model', LinearRegression())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Save the model pipeline
with open('model_pipeline.pkl', 'wb') as model_file:
    pickle.dump(model_pipeline, model_file)

print("Model pipeline saved successfully.")'''
