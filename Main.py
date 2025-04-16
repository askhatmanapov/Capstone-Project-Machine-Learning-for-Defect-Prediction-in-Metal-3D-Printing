import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load Excel file
file_path = 'ANSYS Data.xlsx'
excel_data = pd.ExcelFile(file_path)

# Extract and clean data
def extract_clean_data(sheet_names, excel_data):
    temperatures, deformations, von_mises = [], [], []

    for sheet_name in sheet_names:
        df = excel_data.parse(sheet_name)
        try:
            temp_data = pd.to_numeric(df['Unnamed: 6'], errors='coerce')  # Temperature
            deform_data = pd.to_numeric(df['Unnamed: 11'], errors='coerce')  # Total Deformation
            von_mises_data = pd.to_numeric(df['Unnamed: 16'], errors='coerce')  # Von-Mises Strain

            for temp, deform, vm in zip(temp_data, deform_data, von_mises_data):
                if not pd.isna(temp) and not pd.isna(deform) and not pd.isna(vm):
                    temperatures.append(temp)
                    deformations.append(deform)
                    von_mises.append(vm)
        except KeyError:
            continue

    return pd.DataFrame({
        "Temperature": temperatures,
        "Total Deformation": deformations,
        "Von-Mises Strain": von_mises
    })

# Extract data
sheet_names = excel_data.sheet_names
data = extract_clean_data(sheet_names, excel_data)

# Apply temperature threshold
threshold = 5000
filtered_data = data[data["Temperature"] <= threshold]

# Prepare features (X) and target (y)
X = filtered_data[["Temperature", "Von-Mises Strain"]]
y = filtered_data["Total Deformation"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Ridge model
ridge = Ridge(alpha=10.0, max_iter=10000)
ridge.fit(X_train, y_train)

# Predictions
ridge_pred = ridge.predict(X_test)

# Evaluate model
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

mse, mae, r2 = evaluate_model(y_test, ridge_pred)

# Visualization: True vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, ridge_pred, color='green', label='Ridge Predictions', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', label='Ideal Fit', linestyle='--')
plt.xlabel('True Total Deformation')
plt.ylabel('Predicted Total Deformation')
plt.title('True vs Predicted Total Deformation (Linear Model)')
plt.legend()

# Add metrics to the figure with 5 decimal places
metrics_text = (
    f"R-squared (R²): {r2:.5f}\n"
    f"Mean Squared Error (MSE): {mse:.5f}\n"
    f"Mean Absolute Error (MAE): {mae:.5f}"
)
plt.text(0.7 * y_test.max(), 0.2 * y_test.max(), metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

plt.show()

