"""Optional RadioML 2018.01A dataset loader."""
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class RadioMLDataset:
    """Loader for RadioML 2018.01A dataset (HDF5 format)."""

    def __init__(self, h5_path: str, fallback_to_synthetic: bool = True):
        self.h5_path = Path(h5_path)
        self.fallback = fallback_to_synthetic
        self.data = None
        self.labels = None
        self.snrs = None

        if self.h5_path.exists():
            self._load()
        else:
            logger.warning(f"RadioML HDF5 not found at {self.h5_path}. Will fall back to synthetic data.")
            self.data = None

    def _load(self) -> None:
        """Load HDF5 file into memory."""
        try:
            with h5py.File(self.h5_path, 'r') as f:
                # RadioML 2018.01A structure
                self.data = f['X'][:]  # (num_samples, 1024, 2)
                self.labels = f['Y'][:].squeeze().argmax(axis=1)  # one-hot to indices
                self.snrs = f['Z'][:].squeeze()
            logger.info(f"Loaded RadioML: {len(self.data)} samples, {len(np.unique(self.labels))} modulations")
        except Exception as e:
            logger.error(f"Failed to load RadioML: {e}")
            self.data = None

    def __len__(self) -> int:
        return len(self.data) if self.data is not None else 0

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, float]:
        """Return (iq, label, snr)."""
        if self.data is None:
            raise RuntimeError("RadioML data not loaded")
        iq = self.data[idx]  # (1024, 2) - I/Q channels
        # Convert to (2, 1024) I/Q channels first
        iq = iq.transpose(1, 0)  # Now (I,Q) per column -> want (2,1024) with I and Q separate?
        # Original RadioML: (1024, 2) where last dim is I/Q
        iq = iq.astype(np.float32)
        label = int(self.labels[idx])
        snr = float(self.snrs[idx])
        return iq, label, snr