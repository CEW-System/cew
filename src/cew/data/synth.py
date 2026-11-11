"""Synthetic RF signal generator with realistic impairments."""
import numpy as np
from scipy import signal
from typing import Tuple, Dict, List, Optional
import yaml

def generate_rrc_filter(span: int, sps: int, alpha: float = 0.35) -> np.ndarray:
    """Generate Root Raised Cosine filter."""
    t = np.arange(-span * sps, span * sps + 1) / sps
    num = np.sin(np.pi * t * (1 - alpha)) + 4 * alpha * t * np.cos(np.pi * t * (1 + alpha))
    den = np.pi * t * (1 - (4 * alpha * t) ** 2)
    h = num / den
    h[np.abs(t) < 1e-8] = 1 - alpha + 4 * alpha / np.pi  # t=0 limit
    return h / np.sqrt(np.sum(h ** 2))

class ModulationGenerator:
    """Generate I/Q samples for various modulation types with impairments."""

    MODULATIONS = [
        "BPSK", "QPSK", "8PSK", "16QAM", "64QAM",
        "GFSK", "CPFSK", "4-FSK", "AM-DSB", "WBFM"
    ]

    def __init__(self, config: Dict):
        self.config = config
        self.frame_len = config["frame_length"]
        self.sps = 4  # samples per symbol
        self.rrc_filter = generate_rrc_filter(8, self.sps, config["rrc_rolloff"])
        self.constellations = self._init_constellations()

    def _init_constellations(self) -> Dict[str, np.ndarray]:
        """Define constellation diagrams."""
        return {
            "BPSK": np.array([1, -1]) / np.sqrt(1),
            "QPSK": np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2),
            "8PSK": np.exp(2j * np.pi * np.arange(8) / 8),
            "16QAM": self._make_square_qam(16),
            "64QAM": self._make_square_qam(64),
            "GFSK": None,  # generated differently
            "CPFSK": None,
            "4-FSK": None,
            "AM-DSB": None,
            "WBFM": None,
        }

    def _make_square_qam(self, M: int) -> np.ndarray:
        """Generate square M-QAM constellation."""
        k = int(np.sqrt(M))
        total_symbols = M
        real = 2 * (np.arange(total_symbols) % k) - (k - 1)
        imag = 2 * (np.arange(total_symbols) // k) - (k - 1)
        constellation = (real + 1j * imag) / np.sqrt(2 * (k**2 - 1) / 3)
        return constellation

    def _apply_rrc(self, symbols: np.ndarray) -> np.ndarray:
        """Pulse shape with RRC filter."""
        upsampled = np.zeros(len(symbols) * self.sps, dtype=complex)
        upsampled[::self.sps] = symbols
        filtered = np.convolve(upsampled, self.rrc_filter, mode='same')
        return filtered

    def generate_am_dsb(self, message_len: int, snr_db: float) -> np.ndarray:
        """Generate AM-DSB signal."""
        m = np.random.randn(message_len)
        m = m / np.max(np.abs(m)) * 0.9  # modulation index < 1
        fc = 0.1  # normalized carrier
        t = np.arange(message_len * self.sps) / self.sps
        carrier = np.cos(2 * np.pi * fc * t)
        baseband = self._apply_rrc(m)
        # Ensure same length
        n = len(baseband)
        tx = (1 + baseband[:n]) * carrier[:n]
        return self._add_awgn(tx, snr_db)

    def generate_wbfm(self, message_len: int, snr_db: float) -> np.ndarray:
        """Generate wideband FM (simplified)."""
        m = np.random.randn(message_len) * 2
        dev_ratio = 0.05
        integrated = np.cumsum(m) / self.sps
        fc = 0.1
        t = np.arange(message_len * self.sps) / self.sps
        phase = 2 * np.pi * fc * t + dev_ratio * integrated
        signal = np.exp(1j * phase)
        return self._add_awgn(signal, snr_db)

    def generate_fsk(self, fsk_type: str, M: int, snr_db: float) -> np.ndarray:
        """Generate FSK signal (GFSK/CPFSK/FSK)."""
        num_symbols = self.frame_len // self.sps
        symbols = np.random.randint(0, M, size=num_symbols)
        freq_sep = 1.0 / self.sps
        f0 = -freq_sep * (M / 2 - 0.5)

        if fsk_type == "GFSK":
            gaussian_taps = signal.firwin(11, 0.35)
            phase = np.zeros(num_symbols * self.sps)
            for i, s in enumerate(symbols):
                freq = f0 + s * freq_sep
                phase[i*self.sps:(i+1)*self.sps] += 2*np.pi*freq/self.sps
            phase = np.convolve(phase, gaussian_taps, mode='same')
        else:
            phase = np.zeros(num_symbols * self.sps)
            for i, s in enumerate(symbols):
                freq = f0 + s * freq_sep
                phase[i*self.sps:(i+1)*self.sps] += 2*np.pi*freq/self.sps

        signal = np.exp(1j * phase)
        return self._add_awgn(signal, snr_db)

    def generate_psk(self, M: int, snr_db: float) -> np.ndarray:
        """Generate PSK signal."""
        if M == 2:
            key = "BPSK"
        elif M == 4:
            key = "QPSK" if "QPSK" in self.constellations else "4PSK"
        else:
            key = f"{M}PSK"
        constellation = self.constellations[key]
        num_symbols = self.frame_len // self.sps
        symbols = np.random.choice(constellation, size=num_symbols)
        tx = self._apply_rrc(symbols)
        return self._add_awgn(tx, snr_db)

    def generate_qam(self, M: int, snr_db: float) -> np.ndarray:
        """Generate QAM signal."""
        constellation = self.constellations[f"{M}QAM"]
        num_symbols = self.frame_len // self.sps
        symbols = np.random.choice(constellation, size=num_symbols)
        tx = self._apply_rrc(symbols)
        return self._add_awgn(tx, snr_db)

    def _add_awgn(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Add complex AWGN to achieve target SNR."""
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
        return signal + noise

    def _apply_rtl_sdr_impairments(self, iq: np.ndarray) -> np.ndarray:
        """Apply RTL-SDR specific impairments: DC offset, IQ imbalance, quantization."""
        # DC spike
        dc_offset = np.random.randn() * self.config["dc_offset_std"]
        iq += dc_offset

        # IQ imbalance (amplitude and phase)
        gain_imbalance = 1 + np.random.randn() * self.config["iq_imbalance_std"]
        phase_imbalance = np.random.randn() * self.config["iq_imbalance_std"] * np.pi/180
        i = np.real(iq)
        q = np.imag(iq)
        i_corrected = i
        q_corrected = gain_imbalance * q * np.cos(phase_imbalance) - gain_imbalance * i * np.sin(phase_imbalance)
        iq = i_corrected + 1j * q_corrected

        # ADC quantization to 8 bits
        max_val = 2**7  # signed 8-bit
        iq_real = np.clip(np.real(iq), -1, 1)
        iq_imag = np.clip(np.imag(iq), -1, 1)
        iq_real = np.round(iq_real * max_val) / max_val
        iq_imag = np.round(iq_imag * max_val) / max_val
        iq = iq_real + 1j * iq_imag

        return iq

    def _apply_fading(self, iq: np.ndarray, snr_db: float) -> np.ndarray:
        """Apply Rayleigh fading if enabled."""
        if not self.config["fading_enabled"]:
            return iq
        h = (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq))) / np.sqrt(2)
        return h * iq

    def generate_frame(self, mod_type: str, snr_db: float) -> Tuple[np.ndarray, float]:
        """
        Generate a single I/Q frame of shape (2, frame_len)
        Returns: (iq_frame, actual_snr)
        """
        if mod_type not in self.MODULATIONS:
            raise ValueError(f"Unknown modulation: {mod_type}")

        num_symbols = self.frame_len // self.sps

        # Baseband generation
        if mod_type in ["BPSK", "QPSK", "8PSK"]:
            M = {"BPSK": 2, "QPSK": 4, "8PSK": 8}[mod_type]
            iq = self.generate_psk(M, snr_db)
        elif mod_type in ["16QAM", "64QAM"]:
            M = {"16QAM": 16, "64QAM": 64}[mod_type]
            iq = self.generate_qam(M, snr_db)
        elif mod_type == "GFSK":
            iq = self.generate_fsk("GFSK", 2, snr_db)
        elif mod_type == "CPFSK":
            iq = self.generate_fsk("CPFSK", 2, snr_db)
        elif mod_type == "4-FSK":
            iq = self.generate_fsk("FSK", 4, snr_db)
        elif mod_type == "AM-DSB":
            iq = self.generate_am_dsb(num_symbols, snr_db)
        elif mod_type == "WBFM":
            iq = self.generate_wbfm(num_symbols, snr_db)
        else:
            iq = np.zeros(self.frame_len * self.sps, dtype=complex)

        # Carrier frequency offset and phase noise
        freq_offset = np.random.uniform(-0.01, 0.01)  # normalized
        phase_noise = np.random.randn(len(iq)) * 0.01
        t = np.arange(len(iq)) / self.sps
        iq = iq * np.exp(2j * np.pi * freq_offset * t + 1j * phase_noise)

        # Timing offset (resample)
        timing_offset = np.random.uniform(-0.1, 0.1)
        if abs(timing_offset) > 1e-6:
            iq = signal.resample(iq, int(len(iq) * (1 + timing_offset)))
            iq = signal.resample(iq, self.frame_len * self.sps)

        # Fading
        iq = self._apply_fading(iq, snr_db)

        # Trim/pad to exact length
        if len(iq) > self.frame_len * self.sps:
            start = np.random.randint(0, len(iq) - self.frame_len * self.sps)
            iq = iq[start:start + self.frame_len * self.sps]
        elif len(iq) < self.frame_len * self.sps:
            pad_len = self.frame_len * self.sps - len(iq)
            iq = np.pad(iq, (0, pad_len), mode='constant')

        # Downsample to frame length
        iq = signal.decimate(iq, self.sps, zero_phase=True)

        # RTL-SDR impairments
        iq = self._apply_rtl_sdr_impairments(iq)

        # Normalize power
        power = np.mean(np.abs(iq) ** 2)
        if power > 0:
            iq = iq / np.sqrt(power)

        measured_snr = self._estimate_snr(iq)
        return iq, measured_snr

    def _estimate_snr(self, iq: np.ndarray) -> float:
        """Simple SNR estimator from signal statistics."""
        # Noise floor estimate from high-order statistics
        noise_est = np.median(np.abs(np.diff(iq))) ** 2
        signal_power = np.var(iq)
        snr = 10 * np.log10(signal_power / (noise_est + 1e-12))
        return float(snr)

class FHSSGenerator:
    """Generate frequency-hopping spread spectrum sequences."""

    def __init__(self, config: Dict):
        self.config = config
        self.num_bins = config["num_hop_bins"]
        self.pn_state = np.random.randint(0, 2, 10)

    def generate_lfsr_sequence(self, length: int, seed: int = 12345) -> np.ndarray:
        """PN sequence using LFSR."""
        state = seed
        seq = []
        for _ in range(length):
            bit = ((state >> 0) ^ (state >> 1) ^ (state >> 2)) & 1
            seq.append(bit)
            state = (state >> 1) | (bit << 15)
        bins = np.arange(self.num_bins)[np.array(seq[:length]) % self.num_bins]
        return bins

    def generate_semi_structured(self, length: int) -> np.ndarray:
        """Pattern: visit all bins in order but some with random jumps."""
        bins = np.arange(self.num_bins)
        np.random.shuffle(bins)
        repeats = length // self.num_bins + 1
        sequence = np.tile(bins, repeats)[:length]
        return sequence

    def generate_sequence(self, pattern: str = "pn", length: int = None) -> np.ndarray:
        """Generate hop frequency sequence."""
        length = length or self.config["hop_pattern_length"]
        if pattern == "pn":
            return self.generate_lfsr_sequence(length)
        elif pattern == "structured":
            return self.generate_semi_structured(length)
        else:
            raise ValueError(f"Unknown hop pattern: {pattern}")

class ThreatScenarioGenerator:
    """Generate labeled threat sequences for LSTM training."""

    def __init__(self, config: Dict):
        self.config = config
        self.seq_len = config["threat_sequence_length"]
        self.feature_dim = config["threat_feature_dim"]

    def generate_sequence(self, threat_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a temporal sequence of features.
        Returns: (features [T, feature_dim], labels [T])
        """
        T = self.seq_len
        features = np.zeros((T, self.feature_dim))
        labels = np.zeros(T, dtype=int)

        if threat_type == "benign":
            # Low probability unknown, low threat score
            for t in range(T):
                features[t, :10] = np.random.dirichlet(np.ones(10))  # class probs
                features[t, 10] = np.random.uniform(0, 0.2)  # unknown prob
                features[t, 11:14] = np.random.uniform(0.3, 0.8, 3)  # other feats
                labels[t] = 0

        elif threat_type == "suspicious":
            # Sporadic high unknown, medium threat
            for t in range(T):
                if t > T // 3 and t < 2 * T // 3:
                    features[t, :10] = np.random.dirichlet(np.ones(10))
                    features[t, 10] = np.random.uniform(0.4, 0.7)
                    features[t, 11:14] = np.random.uniform(0.5, 1.0, 3)
                    labels[t] = 1
                else:
                    features[t, :10] = np.random.dirichlet(np.ones(10))
                    features[t, 10] = np.random.uniform(0, 0.2)
                    features[t, 11:14] = np.random.uniform(0.3, 0.6, 3)
                    labels[t] = 0

        elif threat_type == "hostile":
            # Persistent high unknown, high threat
            for t in range(T):
                features[t, :10] = np.random.dirichlet(np.ones(10))
                features[t, 10] = np.random.uniform(0.6, 0.9)
                features[t, 11:14] = np.random.uniform(0.7, 1.0, 3)
                labels[t] = 2

        # Normalize features
        features[:, :10] /= features[:, :10].sum(axis=1, keepdims=True)
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

        return features, labels