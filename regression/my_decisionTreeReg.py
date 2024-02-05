import tkinter as tk
from tkinter import filedialog
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from pandas import read_csv

def load_data():
    file_path = filedialog.askopenfilename()
    data = read_csv(file_path)
    return data.values[:, :-1], data.values[:, -1]

def decision_tree_regressor(input_data, output_data):
    # Scale the input data
    scaler = StandardScaler() 
    scaled_data = scaler.fit_transform(input_data)

    # Initialize Decision Tree Regressor
    decision_tree = DecisionTreeRegressor(random_state=42)

    # Cross-validation to avoid ucderfitting and overfitting.
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_predictions = cross_val_predict(decision_tree, scaled_data, output_data, cv=kf)

    # Evaluate Decision Tree Regressor
    mae = mean_absolute_error(output_data, cv_predictions)
    mse = mean_squared_error(output_data, cv_predictions)
    r2 = r2_score(output_data, cv_predictions)

    return mae, mse, r2

def start_processing():
    global input_data, output_data
    input_data, output_data = load_data()

    # Perform Decision Tree Regression
    mae, mse, r2 = decision_tree_regressor(input_data, output_data)

    # Print and display evaluation metrics
    result_text = "\nDecision Tree Regression Metrics:\n"
    result_text += f"Mean Absolute Error: {mae:.4f}\n"
    result_text += f"Mean Squared Error: {mse:.4f}\n"
    result_text += f"R-squared: {r2:.4f}\n"

    result_label.config(text=result_text)

# GUI Setup
root = tk.Tk()
root.title("Decision Tree Regressor GUI")

# Styling
root.geometry("500x400")
root.configure(bg="#f0f0f0")

load_button = tk.Button(root, text="Load Data", command=load_data, padx=10, pady=5, bg="#4CAF50", fg="white")
load_button.pack(pady=20)

start_button = tk.Button(root, text="Start Processing", command=start_processing, padx=10, pady=5, bg="#008CBA", fg="white")
start_button.pack(pady=20)

result_label = tk.Label(root, text="", font=("Helvetica", 10), justify="left", bg="#f0f0f0")
result_label.pack(pady=20)

root.mainloop()
