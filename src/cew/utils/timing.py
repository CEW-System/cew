"""Timing and latency measurement utilities."""
import time
from collections import deque
from typing import Dict, List, Optional
import numpy as np

class Timer:
    """Simple context manager for timing code blocks."""
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time

class LatencyRecorder:
    """Record latencies and compute statistics."""
    def __init__(self, max_samples: int = 10000):
        self.latencies: Dict[str, deque] = {}
        self.max_samples = max_samples

    def record(self, stage: str, latency_ms: float) -> None:
        """Record latency for a stage in milliseconds."""
        if stage not in self.latencies:
            self.latencies[stage] = deque(maxlen=self.max_samples)
        self.latencies[stage].append(latency_ms)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return statistics for all stages."""
        stats = {}
        for stage, values in self.latencies.items():
            arr = np.array(values)
            stats[stage] = {
                "mean_ms": float(np.mean(arr)),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
                "min_ms": float(np.min(arr)),
                "max_ms": float(np.max(arr)),
            }
        return stats

    def reset(self) -> None:
        """Clear all recorded latencies."""
        for queue in self.latencies.values():
            queue.clear()