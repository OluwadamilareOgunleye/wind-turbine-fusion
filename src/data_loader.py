import sys
import os

from pathlib import Path
import numpy as np
import scipy.io as sio



def load_mat_file(file_path: str | Path):
    """
    Load a single .mat file and return its numeric data as a NumPy array.
    """

    mat_data = sio.loadmat(file_path)

    # Remove MATLAB metadata keys
    data_keys = [
        key for key in mat_data.keys()
        if not key.startswith("__")
    ]

    if not data_keys:
        raise ValueError(f"No data found in {file_path}")

    # If the file contains multiple variables,
    # select the first one
    data = mat_data[data_keys[0]]

    return np.asarray(data)


def load_nrel_healthy_data(data_dir="./data/raw/nrel/Healthy"):
    """
    Load all .mat files from the NREL Healthy dataset folder.
    """

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Directory not found: {data_path.resolve()}"
        )

    mat_files = sorted(data_path.glob("*.mat"))

    if not mat_files:
        raise FileNotFoundError(
            f"No .mat files found in {data_path.resolve()}"
        )

    dataset = {}

    for mat_file in mat_files:
        try:
            data = load_mat_file(mat_file)

            dataset[mat_file.stem] = data

            print(
                f"Loaded {mat_file.name} "
                f"| Shape: {data.shape} "
                f"| Type: {data.dtype}"
            )

        except Exception as e:
            print(f"Failed to load {mat_file.name}: {e}")

    return dataset


if __name__ == "__main__":

    healthy_data = load_nrel_healthy_data()

    print("\nDataset Summary")

    for filename, array in healthy_data.items():
        print(f"{filename}: {array.shape}")




if __name__ == "__main__":

    healthy_data = load_nrel_healthy_data()

    # Inspect H1
    signal = healthy_data["H1"]

    print("\n--- H1 Inspection ---")

    print("Shape:", signal.shape)
    print("Data type:", signal.dtype)

    print("\nFirst 10 values:")
    print(signal[:10])

    print("\nMinimum value:", signal.min())
    print("Maximum value:", signal.max())
    print("Mean:", signal.mean())
    print("Standard deviation:", signal.std())