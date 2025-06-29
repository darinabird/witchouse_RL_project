from pathlib import Path
import numpy as np


def compute_etalon():
    """
    Compute and save the mean embedding vector from all target .npy files.
    """
    emb_dir = Path("embeddings/target")
    vectors = []
    for f in emb_dir.glob("*.npy"):
        vectors.append(np.load(f))
    if not vectors:
        raise RuntimeError(f"No .npy files found in {emb_dir}")

    mean_target = np.mean(vectors, axis=0)
    out_path = Path("embeddings/mean_witchouse.npy")
    np.save(out_path, mean_target)
    print(f"Saved mean etalon to {out_path}")


# Automatically compute etalon on import if missing
_etalon_path = Path("embeddings/mean_witchouse.npy")
if not _etalon_path.exists():
    compute_etalon()

# Load the computed etalon vector
ETALON_VECTOR = np.load(str(_etalon_path))
