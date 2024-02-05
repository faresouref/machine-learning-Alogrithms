import tkinter as tk
from tkinter import filedialog
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pandas import read_csv
import numpy as np

def load_data():
    file_path = filedialog.askopenfilename()
    data = read_csv(file_path)
    return data.values[:, :-1], data.values[:, -1]

def start_processing():
    global input_data, output_data
    input_data, output_data = load_data()

    # You can adjust the number of clusters based on your needs
    num_clusters = 2
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    # Fit k-Means to the data
    kmeans.fit(input_data)

    # Get cluster labels for each data point
    cluster_labels = kmeans.labels_

    # Evaluate the clustering using silhouette score
    silhouette_avg = silhouette_score(input_data, cluster_labels)

    # Print clustering results
    result_text = "\nClustering Results:\n"
    result_text += f"Silhouette Score: {silhouette_avg:.4f}\n"
    result_text += "Cluster Labels:\n" + str(cluster_labels) + "\n"

    result_label.config(text=result_text)

# GUI Setup
root = tk.Tk()
root.title("k-Means Clustering GUI")

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
