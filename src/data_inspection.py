from src.data_loader import load_nrel_healthy_data


# Load all healthy signals
healthy_data = load_nrel_healthy_data()


# Access H1
H1 = healthy_data["H1"].flatten()


print("H1 shape:", H1.shape)

print("\nFirst 10 samples:")
print(H1[:10])

print("\nStatistics:")
print("Minimum:", H1.min())
print("Maximum:", H1.max())
print("Mean:", H1.mean())
print("Standard deviation:", H1.std())