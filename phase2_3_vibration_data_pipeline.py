# %% [markdown]
# # Wind Turbine Fusion Project — Phase 2 & 3
# Data Acquisition/Cleaning (Phase 2) + Signal Processing & Feature Engineering (Phase 3)
#
# Works against the NREL Wind Turbine Gearbox Condition Monitoring Vibration
# Analysis Benchmarking Dataset (Healthy.zip / Damaged.zip you already have).
#
# IMPORTANT — read this before running on your real files:
# The exact variable names inside NREL's .mat files aren't independently
# confirmed here (I don't have your actual files to inspect). Documentation
# consistently describes 8 accelerometer channels labeled AN3–AN10 plus a
# rotational-speed/tachometer signal, sampled at 40 kHz for 1-minute files.
# Run Section 1 (inspect_mat_file) on ONE real file first — it prints every
# variable name and shape in the file — and adjust CHANNEL_MAP in Section 2
# to match exactly what you see before trusting any output past that point.

# %%
import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import kurtosis, skew
from scipy.fft import rfft, rfftfreq

# ----------------------------------------------------------------------
# Section 1 — Inspect a raw .mat file (RUN THIS FIRST on a real file)
# ----------------------------------------------------------------------
def inspect_mat_file(path):
    """Print every variable in a .mat file with its shape and dtype.
    Use this on one real H*.mat / D*.mat file before assuming CHANNEL_MAP
    below is correct — NREL's internal naming isn't independently verified
    in this script."""
    mat = sio.loadmat(path)
    print(f"\nFile: {path}")
    print(f"{'Variable':<15}{'Shape':<20}{'Dtype'}")
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        shape = getattr(v, "shape", None)
        dtype = getattr(v, "dtype", type(v))
        print(f"{k:<15}{str(shape):<20}{dtype}")
    return mat

# Example (uncomment and point at a real file):
# inspect_mat_file("data/raw/Healthy/H1.mat")

# ----------------------------------------------------------------------
# Section 2 — Configuration
# ----------------------------------------------------------------------
SAMPLE_RATE_HZ = 40_000          # documented NREL sampling rate
VIBRATION_CHANNELS = [f"AN{i}" for i in range(3, 11)]   # AN3..AN10
SPEED_CHANNEL_CANDIDATES = ["Speed", "Tach", "RPM", "Speed1", "Tacho"]

# Physical mount location for each channel — this is what turns a column
# name into an engineering statement, and is what you'll reproduce next to
# your Phase 1 CAD sensor-mapping diagram.
CHANNEL_LOCATIONS = {
    "AN3":  "Ring gear radial, 6 o'clock",
    "AN4":  "Ring gear radial, 12 o'clock",
    "AN5":  "Low-speed shaft (LSS) radial",
    "AN6":  "Intermediate-speed shaft (ISS) radial",
    "AN7":  "High-speed shaft (HSS) radial",
    "AN8":  "HSS upwind bearing radial",
    "AN9":  "HSS downwind bearing radial",
    "AN10": "Planet carrier downwind radial",
}

RAW_DIR = {"Healthy": "data/raw/Healthy", "Damaged": "data/raw/Damaged"}
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Window length for feature extraction. NREL's own benchmarking work splits
# each 1-minute, 40 kHz file into short segments rather than treating it as
# one giant reading — 0.1s (4,000 samples) is a documented convention; 1.0s
# is a reasonable, less noisy alternative if you want fewer, more stable rows.
WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE_HZ)

# %% [markdown]
# ## Section 3 — Robust file loader
# Auto-detects vibration channels (AN3–AN10) and a speed channel from
# whatever keys are actually present, rather than assuming a rigid layout —
# real .mat exports sometimes wrap signals in nested structs or use slightly
# different capitalization.

# %%
def load_mat_signals(path):
    """Return (dict of 1D vibration arrays, speed array or None, samples)."""
    mat = sio.loadmat(path)
    signals = {}
    speed = None
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        arr = np.asarray(v).squeeze()
        if arr.ndim != 1:
            continue
        if k in VIBRATION_CHANNELS:
            signals[k] = arr.astype(float)
        elif k in SPEED_CHANNEL_CANDIDATES or "speed" in k.lower() or "tach" in k.lower():
            speed = arr.astype(float)
    return signals, speed

def label_from_folder(folder_key):
    return 0 if folder_key == "Healthy" else 1

# %% [markdown]
# ## Section 4 — Data cleaning
# Applied per-channel, per-file, before any feature is computed.

# %%
def clean_signal(x, name="signal"):
    """Physical-plausibility + gap handling for a raw accelerometer trace."""
    x = np.asarray(x, dtype=float)
    n_nan = np.isnan(x).sum()
    if n_nan > 0:
        # short linear interpolation for isolated dropouts; if the block of
        # missing samples is large, that's a sensor/DAQ fault, not noise —
        # flag it instead of silently patching a long stretch
        idx = np.arange(len(x))
        good = ~np.isnan(x)
        if good.sum() / len(x) > 0.99:
            x = np.interp(idx, idx[good], x[good])
        else:
            print(f"  [warn] {name}: {n_nan} NaNs ({100*n_nan/len(x):.2f}%) — large gap, inspect this file manually")
    # accelerometers reading exactly 0.0 for a long stretch usually means the
    # channel was disconnected/saturated, not a genuinely quiet drivetrain
    zero_run = np.mean(x == 0.0)
    if zero_run > 0.05:
        print(f"  [warn] {name}: {100*zero_run:.1f}% exact-zero samples — possible sensor dropout")
    return x

# %% [markdown]
# ## Section 5 — Time-domain feature extraction
# Standard vibration-analysis statistics. Each has a direct mechanical
# interpretation (noted inline) rather than being generic descriptive stats.

# %%
def time_domain_features(x, prefix):
    rms = np.sqrt(np.mean(x**2))                      # overall vibration energy
    peak = np.max(np.abs(x))                          # worst instantaneous event
    p2p = np.max(x) - np.min(x)                        # peak-to-peak amplitude
    crest = peak / rms if rms > 0 else np.nan           # impulsiveness (spiky vs smooth)
    kurt = kurtosis(x, fisher=True)                     # sensitive to sharp impacts (early gear/bearing damage)
    sk = skew(x)                                        # asymmetry of the vibration waveform
    return {
        f"{prefix}_rms": rms,
        f"{prefix}_peak": peak,
        f"{prefix}_p2p": p2p,
        f"{prefix}_crest_factor": crest,
        f"{prefix}_kurtosis": kurt,
        f"{prefix}_skew": sk,
        f"{prefix}_std": np.std(x),
    }

# %% [markdown]
# ## Section 6 — Frequency-domain features (FFT)
# Converts each window into its frequency spectrum, then summarizes it —
# dominant frequency and energy in mechanically meaningful bands — rather
# than keeping thousands of raw FFT bins as features.

# %%
def frequency_domain_features(x, fs, prefix, bands=((0, 500), (500, 2000), (2000, 8000), (8000, 20000))):
    n = len(x)
    windowed = x * np.hanning(n)                       # reduces spectral leakage at window edges
    spectrum = np.abs(rfft(windowed))
    freqs = rfftfreq(n, d=1.0 / fs)

    total_energy = np.sum(spectrum**2) + 1e-12
    dominant_freq = freqs[np.argmax(spectrum)]

    feats = {
        f"{prefix}_dominant_freq_hz": dominant_freq,
        f"{prefix}_spectral_energy": total_energy,
    }
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        band_energy = np.sum(spectrum[mask] ** 2)
        feats[f"{prefix}_band_{lo}_{hi}Hz_energy_ratio"] = band_energy / total_energy
    return feats

# %% [markdown]
# ## Section 7 — Build the full feature table
# Walks every Healthy/Damaged .mat file, windows each channel, extracts
# time + frequency features per window, and stacks everything into one
# tidy DataFrame — this is the table that feeds Phase 4 (ML).

# %%
def process_file(path, label, file_id):
    signals, speed = load_mat_signals(path)
    if not signals:
        print(f"  [warn] no recognized vibration channels in {path} — check CHANNEL_MAP / inspect_mat_file()")
        return []

    n_samples = len(next(iter(signals.values())))
    n_windows = n_samples // WINDOW_SAMPLES
    rows = []

    for w in range(n_windows):
        s, e = w * WINDOW_SAMPLES, (w + 1) * WINDOW_SAMPLES
        row = {"file_id": file_id, "window": w, "label": label}
        for ch, arr in signals.items():
            seg = clean_signal(arr[s:e], name=f"{file_id}:{ch}:win{w}")
            row.update(time_domain_features(seg, ch))
            row.update(frequency_domain_features(seg, SAMPLE_RATE_HZ, ch))
        if speed is not None and len(speed) >= e:
            row["speed_mean"] = np.mean(speed[s:e])
            row["speed_std"] = np.std(speed[s:e])
        rows.append(row)
    return rows

def build_feature_table():
    all_rows = []
    for folder_key, folder_path in RAW_DIR.items():
        label = label_from_folder(folder_key)
        files = sorted(glob.glob(os.path.join(folder_path, "*.mat")))
        if not files:
            print(f"[info] no .mat files found in {folder_path} — update RAW_DIR to your extracted Healthy.zip/Damaged.zip paths")
            continue
        for f in files:
            file_id = os.path.splitext(os.path.basename(f))[0]
            print(f"Processing {file_id} ({folder_key})...")
            all_rows.extend(process_file(f, label, file_id))
    return pd.DataFrame(all_rows)

# %% [markdown]
# ## Section 8 — Run the pipeline
# %%
if __name__ == "__main__":
    features_df = build_feature_table()
    if len(features_df):
        out_path = os.path.join(PROCESSED_DIR, "gearbox_vibration_features.csv")
        features_df.to_csv(out_path, index=False)
        print(f"\nSaved {len(features_df)} feature rows to {out_path}")
        print(features_df.groupby("label").size().rename({0: "Healthy", 1: "Damaged"}))
    else:
        print("\nNo data processed — set RAW_DIR to point at your extracted "
              "Healthy.zip / Damaged.zip contents (e.g. data/raw/Healthy/H1.mat "
              "... data/raw/Damaged/D1.mat) and re-run.")
