import tkinter as tk
from tkinter import filedialog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pandas import read_csv

def load_data():
    file_path = filedialog.askopenfilename()
    data = read_csv(file_path)
    return data.values[:, :-1], data.values[:, -1]

def start_processing():
    global input_data, output_data
    input_data, output_data = load_data()

    # You can adjust Random Forest parameters based on your needs
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation predictions
    cv_predictions = cross_val_predict(random_forest, input_data, output_data, cv=skf)

    # Print overall evaluation metrics
    result_text = "\nRandom Forest Evaluation Metrics:\n"
    result_text += print_evaluation_metrics(cv_predictions, output_data)

    result_label.config(text=result_text)

def print_evaluation_metrics(predictions, output_data):
    result = ""
    result += "Accuracy: " + str(accuracy_score(predictions, output_data)) + "\n"
    result += "Precision: " + str(precision_score(predictions, output_data, average='weighted')) + "\n"
    result += "Recall: " + str(recall_score(predictions, output_data, average='weighted')) + "\n"
    result += "F1 Score: " + str(f1_score(predictions, output_data, average='weighted')) + "\n"
    result += "Confusion Matrix:\n" + str(confusion_matrix(predictions, output_data)) + "\n"
    return result

# GUI Setup
root = tk.Tk()
root.title("Random Forest Classifier GUI")

# Styling
root.geometry("500x400")
root.configure(bg="#f0f0f0")

# Using ttk for themed buttons
load_button = tk.Button(root, text="Load Data", command=load_data, padx=10, pady=5, bg="#4CAF50", fg="white")
load_button.pack(pady=20)

start_button = tk.Button(root, text="Start Processing", command=start_processing, padx=10, pady=5, bg="#008CBA", fg="white")
start_button.pack(pady=20)

# Added a text widget to display the result
result_label = tk.Label(root, text="", font=("Helvetica", 10), justify="left", bg="#f0f0f0")
result_label.pack(pady=20)

root.mainloop()
