import numpy as np
import pandas as pd
import random
from typing import Tuple, Literal, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.signal import decimate
import matplotlib.pyplot as plt

from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, List

def plot_avg_sensor_similarity(
    sim_matrix: np.ndarray,
    sensor_order: Optional[np.ndarray] = None,
    cluster_labels: Optional[np.ndarray] = None,
    figsize: tuple = (10, 8),
    title: str = "Average Sensor Similarity Across Tasks",
    cmap: str = "viridis"
):
    """
        sim_matrix: 2D numpy array [sensors x sensors], average cosine similarity matrix.
        sensor_order: array of reordered sensor indices.
        cluster_labels: array of cluster labels (used to mark cluster boundaries).

    """
    # Apply sensor reordering if provided
    if sensor_order is not None:
        sim_matrix = sim_matrix[np.ix_(sensor_order, sensor_order)]
        if cluster_labels is not None:
            cluster_labels = np.array(cluster_labels)[sensor_order]

    plt.figure(figsize=figsize)
    sns.heatmap(sim_matrix, cmap=cmap, square=True, cbar=True)
    plt.title(title)
    plt.xlabel("Sensors")
    plt.ylabel("Sensors")

    # Optional: draw lines between clusters
    if cluster_labels is not None:
        boundaries = []
        last_label = cluster_labels[0]
        for i, lbl in enumerate(cluster_labels):
            if lbl != last_label:
                boundaries.append(i)
                last_label = lbl
        for b in boundaries:
            plt.axhline(b, color='white', lw=1)
            plt.axvline(b, color='white', lw=1)

    plt.tight_layout()
    plt.show()


def plot_training_history(history, metrics=['loss', 'accuracy']):
    """
    Plot training and validation metrics from Keras history.

    Parameters:
        history: Keras History object (e.g., returned by model.fit)
        metrics: List of metrics to plot (default: ['loss', 'accuracy'])
    """
    if not hasattr(history, 'history'):
        raise ValueError("Expected a Keras History object")

    history_dict = history.history
    epochs = range(1, len(history_dict[metrics[0]]) + 1)

    for metric in metrics:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, history_dict[metric], 'b-', label=f'Train {metric}')
        val_metric = f'val_{metric}'
        if val_metric in history_dict:
            plt.plot(epochs, history_dict[val_metric], 'r--', label=f'Val {metric}')
        plt.title(f'{metric.capitalize()} over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
