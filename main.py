import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

# Load the Excel file
excel_file_path = './data/medicine_datasets.xlsx'

# Read the datasets from different sheets
consumption_data = pd.read_excel(excel_file_path, sheet_name='consumption_data')
inventory_data = pd.read_excel(excel_file_path, sheet_name='inventory_data')
ordering_data = pd.read_excel(excel_file_path, sheet_name='ordering_data')

# Rename columns if necessary
consumption_data = consumption_data.rename(columns={'Drug Name': 'medicine_name'})
inventory_data = inventory_data.rename(columns={'Drug Name': 'medicine_name'})

# Display the first few rows of each dataset to verify
print("Consumption Data:\n", consumption_data.head())
print("\nInventory Data:\n", inventory_data.head())
print("\nOrdering Data:\n", ordering_data.head())

# Check column names and data types
print("Consumption Data Columns:", consumption_data.columns)
print("Inventory Data Columns:", inventory_data.columns)
print("Ordering Data Columns:", ordering_data.columns)

print("Data types in consumption data:", consumption_data.dtypes)
print("Data types in inventory data:", inventory_data.dtypes)
print("Data types in ordering data:", ordering_data.dtypes)

# Perform the merge step by step
merged_data_1 = pd.merge(consumption_data, inventory_data, on='medicine_name', how='left')
print("Intermediate Merged Data 1:\n", merged_data_1.head())

# Check if 'district' is in the columns
if 'district' in merged_data_1.columns:
    final_data = pd.merge(merged_data_1, ordering_data, on=['medicine_name', 'district'], how='left')
else:
    final_data = pd.merge(merged_data_1, ordering_data, on='medicine_name', how='left')

# Display the final merged dataset
print("\nFinal Merged Data:\n", final_data.head())

# Handle missing values
final_data = final_data.dropna()

# Convert date columns if needed
if 'date' in final_data.columns:
    final_data['date'] = pd.to_datetime(final_data['date'])

# Convert categorical features into numeric using Label Encoding
if 'medicine_name' in final_data.columns and 'district' in final_data.columns:
    le = LabelEncoder()
    final_data['medicine_encoded'] = le.fit_transform(final_data['medicine_name'])
    final_data['district_encoded'] = le.fit_transform(final_data['district'])
else:
    print("Error: Required columns for Label Encoding are missing.")

# Feature selection
required_columns = ['year', 'month', 'district_encoded', 'medicine_encoded', 'inventory_level', 'order_quantity']
if all(col in final_data.columns for col in required_columns):
    X = final_data[required_columns]  # Features
    y = final_data['quantity']  # Target variable
else:
    print("Error: Required columns for feature selection are missing.")
    exit()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on test data
y_pred_lr = lr_model.predict(X_test)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred_rf = rf_model.predict(X_test)

# ANN Model
ann_model = Sequential()
ann_model.add(Dense(300, input_dim=X_train.shape[1], activation='relu'))
ann_model.add(Dense(150, activation='relu'))
ann_model.add(Dense(75, activation='relu'))
ann_model.add(Dense(1, activation='linear'))

# Compile the model
ann_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
ann_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Predict on test data
y_pred_ann = ann_model.predict(X_test)

# Evaluate Linear Regression
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression - RMSE: {rmse_lr}, R²: {r2_lr}")

# Evaluate Random Forest
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest - RMSE: {rmse_rf}, R²: {r2_rf}")

# Evaluate ANN
rmse_ann = mean_squared_error(y_test, y_pred_ann, squared=False)
r2_ann = r2_score(y_test, y_pred_ann)
print(f"ANN - RMSE: {rmse_ann}, R²: {r2_ann}")

# Plot results for Random Forest
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred_rf, label='Predicted', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Actual')
plt.title("Random Forest Predictions vs Actual")
plt.xlabel("Actual Quantity")
plt.ylabel("Predicted Quantity")
plt.legend()
plt.show()
