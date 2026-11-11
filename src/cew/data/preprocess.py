"""Preprocessing utilities for I/Q frames."""
import numpy as np
from typing import Tuple

def normalize_frame(iq: np.ndarray, norm: str = "power") -> np.ndarray:
    """
    Normalize I/Q frame.
    norm: "power" (unit power), "std" (zero-mean, unit-var), or "none"
    """
    if norm == "power":
        power = np.mean(np.abs(iq) ** 2)
        if power > 0:
            iq = iq / np.sqrt(power)
        return iq
    elif norm == "std":
        mean = np.mean(iq)
        std = np.std(iq)
        if std > 0:
            iq = (iq - mean) / std
        return iq
    elif norm == "none":
        return iq
    else:
        raise ValueError(f"Unknown normalization: {norm}")

def augment_frame(iq: np.ndarray, snr_db: float, mode: str = "train") -> np.ndarray:
    """
    Apply data augmentation (SNR-aware).
    During training: add frequency shift, small rotation, noise.
    During eval: no augmentation.
    """
    if mode != "train":
        return iq

    # Random small frequency offset (as if from oscillator drift)
    freq_offset = np.random.uniform(-0.005, 0.005)
    t = np.arange(len(iq)) / 1024  # normalized time
    iq = iq * np.exp(2j * np.pi * freq_offset * t)

    # Small phase rotation
    phase_rot = np.random.uniform(-np.pi/32, np.pi/32)
    iq = iq * np.exp(1j * phase_rot)

    # Additive noise (keep SNR within reasonable range)
    current_power = np.mean(np.abs(iq) ** 2)
    if snr_db < 20:
        extra_noise_power = current_power / (10 ** (snr_db / 10)) * 0.1
        noise = np.sqrt(extra_noise_power / 2) * (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))
        iq = iq + noise

    return iq