import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pandas import read_csv

def load_data():
    file_path = filedialog.askopenfilename()
    data = read_csv(file_path)
    return data.values[:, :-1], data.values[:, -1]

def start_processing():
    global input_data, output_data
    input_data, output_data = load_data()
    
    num_neighbors = 3  

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)

    # Cross-validation
    cv_predictions = cross_val_predict(knn, input_data, output_data, cv=skf)

    # Print overall evaluation metrics
    result_text = "\nOverall Evaluation Metrics:\n"
    result_text += print_evaluation_metrics(cv_predictions, output_data)

    result_text += "\nCross-Validation Evaluation Metrics:\n"
    result_text += print_evaluation_metrics(cv_predictions, output_data)

    result_label.config(text=result_text)  # Corrected line

def print_evaluation_metrics(predictions, output_data):
    result = ""
    result += "Accuracy: {:.2f}\n".format(accuracy_score(predictions, output_data))
    result += "Precision: {:.2f}\n".format(precision_score(predictions, output_data, average='weighted'))
    result += "Recall: {:.2f}\n".format(recall_score(predictions, output_data, average='weighted'))
    result += "F1 Score: {:.2f}\n".format(f1_score(predictions, output_data, average='weighted'))
    result += "Confusion Matrix:\n" + str(confusion_matrix(predictions, output_data)) + "\n"
    return result

# GUI Setup
root = tk.Tk()
root.title("K-Nearest Neighbors Classifier GUI")

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
