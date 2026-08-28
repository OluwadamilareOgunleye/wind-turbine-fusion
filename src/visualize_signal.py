import matplotlib.pyplot as plt

from src.data_loader import load_nrel_healthy_data


# Load dataset
healthy_data = load_nrel_healthy_data()

# Get H1 and convert to 1D
H1 = healthy_data["H1"].flatten()


# Number of samples to display
n_samples = 5000

signal = H1[:n_samples]


# Plot
plt.figure(figsize=(14, 5))

plt.plot(signal)

plt.title("NREL Healthy Wind Turbine Signal - H1")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.grid(True)
plt.tight_layout()

plt.show()