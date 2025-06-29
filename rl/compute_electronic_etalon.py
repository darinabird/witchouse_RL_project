# compute_electronic_etalon.py
"""
Compute and expose the mean embedding vector for the electronic etalon.
"""
from pathlib import Path
import numpy as np

# Directory containing electronic embeddings
elec_dir = Path("embeddings/electronic")
# Load all .npy embeddings
vecs = [np.load(f) for f in elec_dir.glob("*.npy")]
# Compute mean vector
MEAN_ELEC_VECTOR = np.mean(vecs, axis=0)

# Optionally save the mean vector to disk
output_path = Path("embeddings") / "mean_electronic.npy"
np.save(output_path, MEAN_ELEC_VECTOR)
print(f"Electronic etalon saved to {output_path}")
