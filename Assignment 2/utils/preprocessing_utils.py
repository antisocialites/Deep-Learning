import os
import glob
import re
import h5py
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
import itertools
from collections import Counter


def _get_dataset_name(path: str) -> str:
    """Return the internal dataset name inside the HDF5 file."""
    return "_".join(os.path.basename(path).split('_')[:-1])

def load_participant_arrays(participant_id: int, base_dir: str = "\train"):
    """
    Concatenate all chunks for every task of one participant and return
    four NumPy arrays in the order:
        rest, task_motor, task_story_math, task_working_memory

    Each array has shape (n_nodes, total_timepoints) or is None.
    """
    # buckets: task → list[(chunk_number, matrix)

    buckets = {
        "rest":               [],
        "task_motor":         [],
        "task_story_math":    [],
        "task_working_memory": []
    }

    # find all relevant files, e.g. rest_105923_1.h5
    _CHUNK_RE = re.compile(r'_(\d+)\.h5$') # capture the trailing “…_<chunk>.h5”
    pattern = os.path.join(base_dir, f"*_{participant_id}_*.h5")
    for path in glob.glob(pattern):
        ds_name = _get_dataset_name(path)

        # identify task & chunk number
        task = next((t for t in buckets if ds_name.startswith(t)), None)
        if task is None:
            continue  # skip unrecognised file

        chunk_match = _CHUNK_RE.search(path)
        chunk_num = int(chunk_match.group(1)) if chunk_match else 0

        # load matrix
        with h5py.File(path, "r") as f:
            matrix = f[ds_name][()]        # (nodes, timepoints)

        buckets[task].append((chunk_num, matrix))

    # concatenate chunks for each task
    out = []
    for task, lst in buckets.items():
        if not lst:
            out.append(None)
            continue

        # sort by chunk number to keep temporal order
        lst.sort(key=lambda item: item[0])
        matrices = [m for _, m in lst]

        # sanity‑check dimensionality: all chunks must share the node axis size
        first_rows = matrices[0].shape[0]
        if not all(mat.shape[0] == first_rows for mat in matrices):
            raise ValueError(f"Inconsistent node counts in {task} chunks for participant {participant_id}")

        # concat along time axis (axis=1)
        out.append(np.concatenate(matrices, axis=1))

    return tuple(out)  # (rest, motor, story_math, working_memory)


def preprocess_meg_data(
    arr: np.ndarray,
    *, #specify by keyword
    scaling: Literal["zscore", "minmax"] = "zscore",
    axis: int = 1,  # time axis
    downsample_method: Literal["slice", "decimate"] = "decimate",
    downsample_factor: Optional[int] = None,
    target_rate: Optional[int] = None,
    orig_rate: int = 2034,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Scale and downsample MEG data.

    Parameters:
        arr: 2D MEG array [sensors x time]
        scaling: "zscore" or "minmax"
        axis: Axis along which to scale and downsample (default: time)
        downsample_method: "slice" or "decimate"
        downsample_factor: Optional integer downsampling factor
        target_rate: Optional new sample rate (alternative to factor)
        orig_rate: Original sample rate (used with target_rate)
        eps: Stability term

    Returns:
        Preprocessed MEG array [sensors x reduced_time]
    """
    # Scaling/Normalization
    if scaling == "zscore":
        means = arr.mean(axis=axis, keepdims=True)
        stds = arr.std(axis=axis, keepdims=True)
        arr = (arr - means) / (stds + eps)
    elif scaling == "minmax":
        mins = arr.min(axis=axis, keepdims=True)
        maxs = arr.max(axis=axis, keepdims=True)
        arr = (arr - mins) / (maxs - mins + eps)
    else:
        raise ValueError("scaling must be 'zscore' or 'minmax'")

    # Downsampling
    if downsample_factor is None:
        if target_rate is None:
            raise ValueError("Specify either downsample_factor or target_rate")
        downsample_factor = int(round(orig_rate / target_rate))

    if downsample_factor > 1:
        if downsample_method == "slice":
            slicer = [slice(None)] * arr.ndim
            slicer[axis] = slice(None, None, downsample_factor)
            arr = arr[tuple(slicer)]
        elif downsample_method == "decimate":
            arr = decimate(arr, downsample_factor, axis=axis, zero_phase=True)
        else:
            raise ValueError("downsample_method must be 'slice' or 'decimate'")

    return arr

def cossim_cluster_multi_task(task_list, n_clusters=5):
    """
    output:
        sorted_indices: Indices that reorder sensors by cluster.
        labels: Cluster labels for each sensor.
        avg_sim_matrix: Averaged cosine similarity matrix.
    """

    # cosine similarity matrix for each task
    sim_matrices = []
    for task in task_list:
        sim = cosine_similarity(task)
        sim_matrices.append(sim)

    # avg similarity matrices
    avg_sim_matrix = np.mean(sim_matrices, axis=0)
    avg_dist_matrix = 1 - avg_sim_matrix

    # clustering
    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        n_clusters=n_clusters
    )
    labels = clustering.fit_predict(avg_dist_matrix)

    # reorder indices
    sorted_indices = np.argsort(labels)

    return sorted_indices, labels, avg_sim_matrix

def prepare_ffnn_data(X):
    """
    X shape: (samples, sequence_length, height, width)
    Returns shape: (samples, sequence_length) after spatial average
    """
    return X.mean(axis=(2, 3))  # average over height and width



def make_sequences(
    data, 
    time_per_snapshot=32,       # e.g., 0.5s at 50 Hz
    sequence_length=10, 
    overlap=0.3                 # 0.5 means 50% overlap
):
    """
    Create sequences of MEG data snapshots with optional overlap.
    """
    sensors, total_time = data.shape
    step = int(time_per_snapshot * sequence_length * (1 - overlap))
    
    # Ensure at least one step forward
    step = max(1, step)

    sequences = []
    for start in range(0, total_time - time_per_snapshot * sequence_length + 1, step):
        seq = np.zeros((sequence_length, sensors, time_per_snapshot))
        for j in range(sequence_length):
            snap_start = start + j * time_per_snapshot
            snap_end = snap_start + time_per_snapshot
            seq[j] = data[:, snap_start:snap_end]
        sequences.append(seq)

    return np.array(sequences)


def make_sequences_with_ids(
    data_list,
    labels_list,
    time_per_snapshot=32, 
    sequence_length=10, 
    overlap=0.3  
):
    """
    Creates subchunk sequences for a list of MEG sequences with labels.
    
    Returns:
        X: np.array of shape (num_subchunks, sequence_length, sensors, time_per_snapshot)
        y: np.array of shape (num_subchunks,) — label for each subchunk
        ids: np.array of shape (num_subchunks,) — sequence ID for grouping later
    """
    X = []
    y = []
    ids = []
    seq_counter = 0

    for data, label in zip(data_list, labels_list):
        # data shape: (sensors, total_time)
        sensors, total_time = data.shape
        step = int(time_per_snapshot * sequence_length * (1 - overlap))
        step = max(1, step)

        for start in range(0, total_time - time_per_snapshot * sequence_length + 1, step):
            chunk = np.zeros((sequence_length, sensors, time_per_snapshot))
            for j in range(sequence_length):
                snap_start = start + j * time_per_snapshot
                snap_end = snap_start + time_per_snapshot
                chunk[j] = data[:, snap_start:snap_end]
            X.append(chunk)
            y.append(label)
            ids.append(seq_counter)  # group ID for later aggregation

        seq_counter += 1

    return np.array(X), np.array(y), np.array(ids)

