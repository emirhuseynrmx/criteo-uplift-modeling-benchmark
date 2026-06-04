from __future__ import annotations

import os
import threading
import time

import psutil


class PeakRSSMonitor:
    """Sample process RSS while a model is fitting."""

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self.peak_rss_mb = 0.0
        self.start_rss_mb = 0.0
        self.thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            rss = self.process.memory_info().rss / (1024**2)
            self.peak_rss_mb = max(self.peak_rss_mb, rss)
            time.sleep(self.interval)

    def __enter__(self) -> PeakRSSMonitor:
        self.start_rss_mb = self.process.memory_info().rss / (1024**2)
        self.peak_rss_mb = self.start_rss_mb
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        rss = self.process.memory_info().rss / (1024**2)
        self.peak_rss_mb = max(self.peak_rss_mb, rss)

    @property
    def peak_delta_mb(self) -> float:
        return max(0.0, self.peak_rss_mb - self.start_rss_mb)
