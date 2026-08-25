from pathlib import Path

# Use the current folder as the project root
ROOT = Path.cwd()

# Directories to create
directories = [
    "data/raw",
    "data/processed",
    "data/features",
    "cad/solidworks",
    "cad/renders",
    "notebooks",
    "src",
    "dashboard",
    "models",
    "reports/figures",
]

# Files to create
files = [
    "README.md",

    # Jupyter notebooks
    "notebooks/01_data_exploration.ipynb",
    "notebooks/02_data_cleaning.ipynb",
    "notebooks/03_signal_processing.ipynb",
    "notebooks/04_feature_engineering.ipynb",
    "notebooks/05_model_training.ipynb",
    "notebooks/06_model_evaluation.ipynb",
    "notebooks/07_explainability.ipynb",

    # Source code
    "src/data_processing.py",
    "src/signal_processing.py",
    "src/feature_engineering.py",
    "src/train.py",
    "src/evaluate.py",
    "src/predict.py",

    # Dashboard
    "dashboard/app.py",

    # Model
    "models/gearbox_model.pkl",

    # Report
    "reports/project_report.pdf",

    # Dependencies
    "requirements.txt",
]


def create_project_structure():
    print(f"Project root: {ROOT.resolve()}\n")

    # Create directories
    for directory in directories:
        path = ROOT / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"[DIR]  {directory}")

    # Create files
    for file in files:
        path = ROOT / file

        # Make sure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            print(f"[SKIP] {file}")
            continue

        # Create valid empty Jupyter notebook
        if path.suffix == ".ipynb":
            path.write_text(
                """{
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}
""",
                encoding="utf-8"
            )

        # README
        elif path.name == "README.md":
            path.write_text(
                """# Wind Turbine Fusion

An industrial-style wind turbine digital engineering,
signal processing, predictive maintenance, and machine
learning project.
""",
                encoding="utf-8"
            )

        # Requirements
        elif path.name == "requirements.txt":
            path.write_text(
                """numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
plotly
jupyter
joblib
streamlit
openpyxl
""",
                encoding="utf-8"
            )

        # Python files
        elif path.suffix == ".py":
            path.write_text(
                f'"""Wind Turbine Fusion - {path.name}."""\n\n',
                encoding="utf-8"
            )

        # Other files
        else:
            path.touch()

        print(f"[FILE] {file}")

    print("\nProject structure created successfully.")


if __name__ == "__main__":
    create_project_structure()