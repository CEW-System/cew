"""Tests for synthetic data generation."""
import numpy as np
import pytest
from cew.data.synth import ModulationGenerator, FHSSGenerator, ThreatScenarioGenerator
from cew.utils.config import load_config

@pytest.fixture
def config():
    return load_config()

def test_generator_shapes(config):
    gen = ModulationGenerator(config["data"])
    iq, snr = gen.generate_frame("BPSK", 10.0)
    assert iq.shape == (config["data"]["frame_length"],)
    assert iq.dtype in [np.complex64, np.complex128]
    assert isinstance(snr, float)

def test_all_modulations(config):
    gen = ModulationGenerator(config["data"])
    for mod in gen.MODULATIONS:
        iq, snr = gen.generate_frame(mod, 5.0)
        assert iq.shape == (config["data"]["frame_length"],)
        assert not np.any(np.isnan(iq))

def test_snr_range(config):
    gen = ModulationGenerator(config["data"])
    snrs = []
    for _ in range(100):
        _, snr = gen.generate_frame("QPSK", np.random.uniform(*config["data"]["snr_range"]))
        snrs.append(snr)
    # SNR should roughly be within -10 to +30 dB after impairments
    assert np.percentile(snrs, 5) > -20
    assert np.percentile(snrs, 95) < 30

def test_fhss_generator(config):
    fh = FHSSGenerator(config["fh"])
    seq = fh.generate_sequence("pn", length=50)
    assert len(seq) == 50
    assert np.all(seq >= 0) and np.all(seq < config["fh"]["num_hop_bins"])

    seq2 = fh.generate_sequence("structured", length=30)
    assert len(seq2) == 30

def test_threat_scenario_generator(config):
    thr = ThreatScenarioGenerator(config["threat"])
    feats, labels = thr.generate_sequence("benign")
    T = config["threat"]["sequence_length"]
    assert feats.shape == (T, config["threat"]["feature_dim"])
    assert labels.shape == (T,)
    assert np.all(np.isin(labels, [0, 1, 2]))

def test_dataset_basic(config):
    from cew.data.dataset import CEWDataset
    ds = CEWDataset(config, split="train", mode="amc")
    iq, label = ds[0]
    assert iq.shape == (2, config["data"]["frame_length"])
    assert isinstance(label.item(), int)

def test_dataset_threat(config):
    from cew.data.dataset import CEWDataset
    ds = CEWDataset(config, split="train", mode="threat")
    feats, labels = ds[0]
    T = config["threat"]["sequence_length"]
    assert feats.shape == (T, config["threat"]["feature_dim"])
    assert labels.shape == (T,)