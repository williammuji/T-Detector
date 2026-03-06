# T-Detector (Refined for macOS)

This is a refined version of the original 《T-detector: A Trajectory based Pretrained Model for Game Bot Detection in MMORPGs》, adapted for modern macOS (ARM64) development environments and enhanced with interactive 2D behavioral visualization.

Base Repository: [aker218/T-Detector](https://github.com/aker218/T-Detector)

## macOS Compatibility Refinements

The original codebase (designed for Python 3.6/3.7) has been upgraded and patched to support **macOS ARM64 (M1/M2/M3)** and **Python 3.12+**:

1.  **Gensim 4.0+ Integration**:
    *   Updated `Word2Vec` parameters: `size` → `vector_size`.
    *   Updated `iter` → `epochs`.
    *   Fixed vocabulary access for modern Gensim API.
2.  **Transformer & PyTorch Upgrades**:
    *   Migrated `AdamW` and `BertLayerNorm` imports to standard `torch.optim` and `torch.nn` for compatibility with `transformers 4.x`.
3.  **Dependency Cleanup**:
    *   Removed `DGL` library dependencies to bypass C++ compilation errors on ARM64 macOS.
    *   Replaced deprecated NumPy aliases (e.g., `np.int`, `np.float`) with standard Python types to prevent runtime crashes.
4.  **Stability Fixes**:
    *   Resolved directory creation typos (`makedirs(exists_ok=True)`).
    *   Patched `trainer.py` to handle zero-loss batches during initial pre-training phases.
    *   Fixed sequence length calculation logic in `models.py`.

## New Features: Combat Radar 2D Visualization

A high-performance 2D behavioral radar has been added to visualize character trajectories in the latent space:

*   **Technology**: Uses **UMAP** for dimensionality reduction and **Plotly WebGL** for interactive rendering.
*   **Combat Radar v1.2**:
    *   **SVG Overlays**: Pulse-modulated markers and "TARGET" annotations guaranteed to render on top of all data points.
    *   **Overlap Handling**: Automatically aggregates perfectly overlapping behavioral segments (X-N indicators).
    *   **Partial/Fuzzy Search**: Find players by AccID or UserID with real-time radar locking.

## Environment Setup

Recommended Environment: **Python 3.12+**

```bash
# Core Dependencies
pip install numpy pandas scipy scikit-learn tqdm torch transformers gensim

# Visualization Dependencies
pip install umap-learn plotly
```

## Directory Structure

```shell
trajectory_detector
├── train_data_sampled/             # Behavioral samples (AccID-UserID)
├── visualize_2d.py                 # Interactive Radar Visualization Tool
├── extract_features.py             # Feature Extraction for Latent Space
├── dataset.py                      # Torch Dataset logic
├── models.py                       # T-Detector Model Architecture
├── trainer.py                      # Training & Evaluation logic
└── run.sh                          # Pipeline entry point
```

## Quick Start

1.  **Prepare Sample Data**: Place behavioral JSONs in `./train_data_sampled/`.
2.  **Run Pipeline**:
    ```bash
    bash run.sh
    ```
3.  **Launch Visualization**:
    ```bash
    python3 trajectory_detector/visualize_2d.py \
      --features ./train_data_sampled_processed/extracted_sampled_features.npy \
      --meta ./train_data_sampled_processed/extracted_sampled_meta.json \
      --output radar_vision.html
    ```

---
*Maintained by williammuji. Original research by the T-Detector authors.*
