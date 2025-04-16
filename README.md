# 🛠️ Capstone Project – Machine Learning for Defect Prediction in Metal 3D Printing

This project explores the application of supervised machine learning to predict deformation and crack probability in metal additive manufacturing, specifically in Directed Energy Deposition (DED) 3D printing. The code is based on real-world simulation data extracted from ANSYS and experimental datasets.

---

## 📂 Project Structure

### `Main.py`
- Loads and processes ANSYS simulation data from multiple Excel sheets.
- Extracts features like Temperature and Von-Mises strain.
- Trains a Ridge Regression model to predict Total Deformation.
- Applies standard scaling, model evaluation (R², MSE, MAE), and visualizes results.

### `Insert.py`
- Loads a separate experimental dataset (`crack_prediction_data.xlsx`).
- Trains a Linear Regression model to predict the probability of crack formation based on:
  - RPM
  - Laser Power
  - Feed Rate
- Offers an interactive command-line interface for testing predictions.
- Includes colored console outputs and inline plotting of feature relationships.

---

## 🧠 ML Techniques Used
- Ridge Regression
- Linear Regression
- Data Standardization
- Feature Engineering from domain-specific parameters
- Evaluation Metrics: R², MSE, MAE
- Visualization of model performance and feature relationships

---

## 📊 Sample Outputs
- R² up to 0.81 for deformation prediction (based on filtered ANSYS and historical open source data).
- Crack probability shown interactively in CLI based on input parameters.

---

## 🛠️ Tools & Libraries
- Python, pandas, numpy
- scikit-learn
- matplotlib
- colorama (for CLI output)

---

## 👨‍💻 Author
**Askhat Manapov**  
BSc Mechanical and Aerospace Engineering | Nazarbayev University  

---

## 📌 Note
All data used in this project has been anonymized and reflects simulation or synthetic sources.
