"""I/Q data sources: file replay, ZMQ socket, and synthetic generator."""
import threading
import queue
import time
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Iterator, Tuple
import numpy as np
import zmq
import logging

logger = logging.getLogger(__name__)

class IQSource(ABC):
    """Abstract base class for I/Q sources."""

    @abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

class FileReplaySource(IQSource):
    """Read pre-recorded I/Q frames from .npy or .cfile."""

    def __init__(self, file_path: str, loop: bool = True, frame_size: int = 1024):
        self.file_path = Path(file_path)
        self.loop = loop
        self.frame_size = frame_size
        self.frames = None
        self.index = 0

    def start(self) -> None:
        if self.file_path.suffix == ".npy":
            data = np.load(self.file_path)
        elif self.file_path.suffix == ".cfile":
            data = np.fromfile(self.file_path, dtype=np.complex64)
        else:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}")

        # Reshape into frames
        n_frames = len(data) // self.frame_size
        self.frames = data[:n_frames * self.frame_size].reshape(n_frames, self.frame_size)
        self.index = 0
        logger.info(f"Loaded {n_frames} frames from {self.file_path}")

    def stop(self) -> None:
        self.frames = None

    def __iter__(self) -> Iterator[np.ndarray]:
        if self.frames is None:
            self.start()

        while True:
            if self.index >= len(self.frames):
                if self.loop:
                    self.index = 0
                else:
                    break
            yield self.frames[self.index]
            self.index += 1

class ZMQSource(IQSource):
    """Receive I/Q samples from GNU Radio via ZMQ PUB socket."""

    def __init__(self, host: str = "localhost", port: int = 5555, dtype=np.complex64):
        self.host = host
        self.port = port
        self.dtype = dtype
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.queue = queue.Queue(maxsize=1000)
        self.context = zmq.Context()
        self.socket = None

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        logger.info(f"ZMQ source started on {self.host}:{self.port}")

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.socket:
            self.socket.close()
        logger.info("ZMQ source stopped")

    def _receive_loop(self) -> None:
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{self.host}:{self.port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout for graceful exit

        while self.running:
            try:
                data = self.socket.recv()
                iq = np.frombuffer(data, dtype=self.dtype)
                self.queue.put(iq, block=False)
            except zmq.Again:
                continue
            except queue.Full:
                pass  # drop oldest if full?
            except Exception as e:
                logger.error(f"ZMQ receive error: {e}")

    def __iter__(self) -> Iterator[np.ndarray]:
        while self.running or not self.queue.empty():
            try:
                yield self.queue.get(timeout=1)
            except queue.Empty:
                break

class SyntheticSource(IQSource):
    """Generate synthetic frames on-the-fly using ModulationGenerator."""

    def __init__(self, config: Dict, class_names: list):
        self.config = config
        self.class_names = class_names
        self.generator = ModulationGenerator(config)
        self.current_idx = 0
        self.class_cycle = list(range(len(class_names)))

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            class_idx = self.class_cycle[self.current_idx % len(self.class_cycle)]
            mod_type = self.class_names[class_idx]
            snr = np.random.uniform(*self.config["snr_range"])
            iq, measured_snr = self.generator.generate_frame(mod_type, snr)
            yield iq
            self.current_idx += 1