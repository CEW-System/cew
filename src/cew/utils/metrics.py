"""Metrics computation utilities."""
import numpy as np
from typing import Tuple, Optional

def compute_snr(iq_signal: np.ndarray, signal_power: Optional[float] = None) -> float:
    """
    Compute SNR in dB from I/Q samples.
    Estimates signal power (if not provided) and noise power via variance.
    """
    if signal_power is None:
        # Assuming signal is centered; use median absolute deviation for noise
        power = np.mean(np.abs(iq_signal) ** 2)
        # Rough noise estimate from high-order moments or derivative
        noise_est = np.var(np.diff(np.abs(iq_signal)))
        snr = 10 * np.log10(power / (noise_est + 1e-10))
    else:
        noise_power = np.var(iq_signal)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    return float(snr)

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Smooth a 1D signal using moving average."""
    return np.convolve(x, np.ones(window)/window, mode='valid')

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    return cm

def accuracy(cm: np.ndarray) -> float:
    """Overall accuracy from confusion matrix."""
    return np.trace(cm) / np.sum(cm)

def accuracy_vs_snr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snrs: np.ndarray,
    snr_bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute accuracy per SNR bin."""
    accs = []
    counts = []
    for i in range(len(snr_bins) - 1):
        mask = (snrs >= snr_bins[i]) & (snrs < snr_bins[i+1])
        if np.any(mask):
            correct = np.sum(y_true[mask] == y_pred[mask])
            total = np.sum(mask)
            accs.append(correct / total)
            counts.append(total)
        else:
            accs.append(np.nan)
            counts.append(0)
    return np.array(accs), np.array(counts)