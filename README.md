# CEW Testbed – Receive-Only Cognitive EW Receiver

## Architecture

```
┌─────────────────┐
│  RTL-SDR / GNU  │   I/Q (complex64)
│    Radio        │──────────────┐
└─────────────────┘              │
                                 ▼
                        ┌─────────────────┐
                        │  SDR Stream /   │  File/ZMQ/Synthetic
                        │  IQSource       │─────────────┐
                        └─────────────────┘             │
                                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Pipeline                          │
│  ┌──────────┐  ┌────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │   CNN    │→│  OpenMax   │→│  FH Transf. │→│  LSTM    │ │
│  │ Mod Rec  │  │ Open Set   │  │ Hop Predict │  │ Threat   │ │
│  └──────────┘  └────────────┘  └─────────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
                                               ┌────────────────────┐
                                               │  Decision Engine   │
                                               │  (KNOWN/UNKNOWN/   │
                                               │   NON-THREATENING) │
                                               └────────────────────┘
                                                          │
                    ┌─────────────────────────────────────┴───────────────┐
                    ▼                     ▼                     ▼
            ┌─────────────┐      ┌─────────────┐    ┌─────────────────┐
            │  Counter    │      │   Alert     │    │   Monitor /     │
            │  (logged)   │      │  (UI)       │    │   Ignore        │
            └─────────────┘      └─────────────┘    └─────────────────┘
```

## Setup

```bash
make install
mkdir -p checkpoints logs runs
```

## Run Demo

```bash
# Generate synthetic dataset (optional, created on-the-fly too)
python scripts/generate_data.py

# Train models
make train

# Evaluate
python scripts/evaluate.py

# Launch dashboard
make demo
  # Sidebar: select "Synthetic Source", adjust thresholds, Start.
```

## GNU Radio Hook-up

1. In GNU Radio, add a **ZMQ PUB Sink** block.
   - Address: `tcp://*:5555` (or as configured)
   - Items: `complex`
   - Message size: `1024` (matching `frame_length`)
   - Set SO_SNDBUF to ~1MB for throughput.

2. Connect the ZMQ block to the FFT output or filtered baseband.

3. In Streamlit sidebar, select **ZMQ Source** and start.

## Performance Goal

End-to-end latency: **< 50 ms** on CPU (Intel i5+/Ryzen 5+).

## Limitations

- Synthetic data: does not fully capture real RTL-SDR impairments (e.g., spurious, nonlinearity).
- RTL-SDR ADC quantization: modeled as 8-bit; real dongles vary.
- Max sample rate: ~2.4 MS/s for RTL-SDR; we downsample to 1 kHz symbol rate equivalent.
- The system is **passive receive-only**. "Counter" action is only a logged recommendation; no transmission.

## Ethics Note

This testbed is for **defensive research and education only**. Use only on signals you own or have explicit permission to analyze.