import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# Load the dataset
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['MedHouseVal'] = california.target

# Handling missing values (if any)
data = data.dropna()

# Split the data into feature and target
X = data.drop(columns=['MedHouseVal'])
Y = data['MedHouseVal']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = Sequential()
model.add(Dense(1, input_dim=X_train.shape[1], activation='linear'))  # Adjust input_dim to match number of features

# Compile the model
model.compile(optimizer=SGD(0.01), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, Y_train, epochs=10, validation_split=0.2, verbose=1)

# Evaluate the model's performance
y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Plot the predicted vs. actual values
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, y_pred, color='blue')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.show()

# Plot residuals
residuals = Y_test - y_pred.flatten()
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='blue')
plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), color='red')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curves')
plt.legend()
plt.show()

# Perform cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=0)

def create_model():
    model = Sequential()
    model.add(Dense(1, input_dim=X.shape[1], activation='linear'))
    model.compile(optimizer=SGD(0.01), loss='mean_squared_error')
    return model

cv_scores = cross_val_score(create_model(), X.values, Y.values, cv=kf, scoring='neg_mean_squared_error')
print(f'Cross-Validation MSE: {-cv_scores.mean()} ± {cv_scores.std()}')
