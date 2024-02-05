import tkinter as tk
from tkinter import filedialog
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from pandas import read_csv
import numpy as np

def load_data():
    file_path = filedialog.askopenfilename()
    data = read_csv(file_path)
    return data.values[:, :-1], data.values[:, -1]

def start_processing():
    global input_data, output_data
    input_data, output_data = load_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

    # Create and train the Linear Regression model
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = linear_reg.predict(X_test)

    # Print evaluation metrics
    result_text = "\nLinear Regression Evaluation Metrics:\n"
    result_text += f"Mean Squared Error: {mean_squared_error(y_test, predictions):.4f}\n"
    result_text += f"R-squared: {r2_score(y_test, predictions):.4f}\n"

    result_label.config(text=result_text)

# GUI Setup
root = tk.Tk()
root.title("Linear Regression GUI")

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
