from src.data_loader import load_nrel_healthy_data


healthy_data = load_nrel_healthy_data(
    "../data/raw/nrel/Healthy"
)

print("\nNumber of files loaded:", len(healthy_data))