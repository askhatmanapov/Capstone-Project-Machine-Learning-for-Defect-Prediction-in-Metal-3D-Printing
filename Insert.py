import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from colorama import Fore, Style, init  # For colored console output

# Initialize colorama
init(autoreset=True)

# Print current working directory
print(Fore.CYAN + "Current Directory:", os.getcwd())

# Load dataset from Excel file using a relative path
script_directory = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
data_path = os.path.join(script_directory, "crack_prediction_data.xlsx")  # Construct the full path to the file

try:
    data = pd.read_excel(data_path)
    print(Fore.GREEN + "File loaded successfully!")
except FileNotFoundError:
    print(Fore.RED + f"File not found at {data_path}. Please check the path and try again.")
    exit()

# Train Linear Regression Model
X = data[['RPM', 'Laser Power', 'Feed Rate']]
y = data['Crack Presence']

model = LinearRegression()
model.fit(X, y)

# Function to predict crack probability
def predict_crack_probability():
    print(Fore.YELLOW + "\n=== Crack Probability Prediction ===")
    try:
        rpm = float(input(Fore.WHITE + "Enter RPM: "))
        laser_power = float(input(Fore.WHITE + "Enter Laser Power (W): "))
        feed_rate = float(input(Fore.WHITE + "Enter Feed Rate (mm/s): "))
        
        # Create input data with feature names to avoid warnings
        input_data = pd.DataFrame([[rpm, laser_power, feed_rate]], columns=['RPM', 'Laser Power', 'Feed Rate'])
        
        # Predict crack probability
        probability = model.predict(input_data)[0]
        
        # Convert probability to percentage
        probability_percentage = probability * 100
        
        # Display the result in a table format
        print(Fore.GREEN + "\n=== Prediction Result ===")
        print(Fore.WHITE + "+-----------------------------+--------+")
        print(Fore.WHITE + "|          Parameter          | Value  |")
        print(Fore.WHITE + "+-----------------------------+--------+")
        print(Fore.WHITE + f"|             RPM             | {rpm:<7.1f}|")
        print(Fore.WHITE + f"|       Laser Power (W)       | {laser_power:<7.1f}|")
        print(Fore.WHITE + f"|      Feed Rate (mm/s)       | {feed_rate:<7.1f}|")
        print(Fore.WHITE + f"| Predicted Crack Probability | {probability_percentage:<7.2f}%|")
        print(Fore.WHITE + "+-----------------------------+--------+")
        
        # Display the probability in a larger font
        print(Fore.BLUE + f"\nGiven the above parameters, the probability of crack is: {Style.BRIGHT}{probability_percentage:.2f}%")
    except ValueError:
        print(Fore.RED + "Invalid input. Please enter numeric values.")

# Run prediction function
predict_crack_probability()

# Plot relationships
print(Fore.YELLOW + "\n=== Plotting Relationships ===")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, feature in enumerate(['RPM', 'Laser Power', 'Feed Rate']):
    X_feature = data[[feature]]
    y = data['Crack Presence']
    
    model_feature = LinearRegression()
    model_feature.fit(X_feature, y)
    y_pred = model_feature.predict(X_feature)
    
    axes[i].scatter(X_feature, y, alpha=0.3, label='Actual')
    axes[i].plot(X_feature, y_pred, color='red', label='Linear Fit')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Crack Presence')
    axes[i].set_title(f'{feature} vs Crack Presence')
    axes[i].legend()

plt.tight_layout()
plt.show()