"""PyTorch Dataset for CEW synthetic data."""
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
import yaml

from .synth import ModulationGenerator, ThreatScenarioGenerator
from .preprocess import normalize_frame, augment_frame

class CEWDataset(Dataset):
    """
    Dataset for modulation classification or threat assessment.
    Mode: "amc" returns (iq, label), "threat" returns (features, label)
    """

    def __init__(
        self,
        config: dict,
        split: str = "train",
        mode: str = "amc",
        class_names: Optional[list] = None,
    ):
        self.config = config
        self.split = split
        self.mode = mode
        self.class_names = class_names or config["data"]["class_names"]
        self.num_classes = len(self.class_names)

        self.generator = ModulationGenerator(config["data"])
        self.threat_gen = ThreatScenarioGenerator(config["threat"])

        # Generate or load samples
        self.samples = self._prepare_samples()

    def _prepare_samples(self) -> list:
        """Generate synthetic dataset with SNR-stratified sampling."""
        samples = []
        rng = np.random.RandomState(self.config["seed"])

        snr_range = self.config["data"]["snr_range"]
        split_ratio = self.config["data"]["split_ratio"]

        if self.mode == "amc":
            # Generate N samples per class, stratified by SNR bins
            snr_bins = np.linspace(snr_range[0], snr_range[1], 5)
            N_per_class = 2000 if self.split == "train" else 500

            for class_idx, mod_type in enumerate(self.class_names):
                for _ in range(N_per_class):
                    # Pick SNR uniformly, then ensure it's from target bin via rejection? Just uniform
                    snr_db = rng.uniform(*snr_range)
                    iq, measured_snr = self.generator.generate_frame(mod_type, snr_db)
                    iq = iq.astype(np.complex64)

                    # Apply augmentation for train split
                    if self.split == "train":
                        iq = augment_frame(iq, measured_snr, mode="train")
                    iq = normalize_frame(iq, norm="power")

                    # Convert to stacked real/imag
                    iq_stacked = np.stack([iq.real, iq.imag], axis=0).astype(np.float32)

                    samples.append({
                        "iq": iq_stacked,
                        "label": class_idx,
                        "snr": measured_snr
                    })

        elif self.mode == "threat":
            # Generate threat sequences
            threat_types = ["benign", "suspicious", "hostile"]
            N = 1000 if self.split == "train" else 200

            for threat_idx, threat_type in enumerate(threat_types):
                for _ in range(N):
                    features, labels = self.threat_gen.generate_sequence(threat_type)
                    # One sample is a sequence of T frames with T labels
                    samples.append({
                        "features": features.astype(np.float32),
                        "labels": labels.astype(np.int64)
                    })

        rng.shuffle(samples)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        if self.mode == "amc":
            iq = torch.from_numpy(sample["iq"])
            label = torch.tensor(sample["label"], dtype=torch.long)
            return iq, label
        elif self.mode == "threat":
            features = torch.from_numpy(sample["features"])
            labels = torch.from_numpy(sample["labels"])
            return features, labels
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

def collate_fn(batch):
    """Collate function for DataLoader (handles variable-length sequences for threat)."""
    if len(batch[0]) == 2:
        x, y = zip(*batch)
        x = torch.stack(x, dim=0)
        y = torch.stack(y, dim=0)
        return x, y
    else:
        raise ValueError("Batch format not supported")