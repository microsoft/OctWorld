"""Seed-owned auxiliary streams for single-process world generation.

Sampling/augmentation keep their historical torch.Generator streams. These
helpers isolate libraries that only expose global random APIs; they do not
promise deterministic CUDA reductions or identical results across backends.
"""

from contextlib import contextmanager
import hashlib
import operator
import random
from threading import RLock

import numpy as np
import torch


_RNG_LOCK = RLock()


def validate_seed(seed: int) -> int:
    seed = operator.index(seed)
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    return seed


def seed_everything(seed: int) -> None:
    seed = validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_seed(seed: int, stream: str, call: int = 0) -> int:
    """Versioned, stable mapping; never use Python's process-salted hash()."""
    payload = f"octworld-rng-v1:{validate_seed(seed)}:{stream}:{call}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % 2**63


@contextmanager
def isolated_rng(seed: int):
    """Restore Python/NumPy/Torch CPU/all visible CUDA RNG, even on errors.

    Serialized across these contexts. This is not isolation from unrelated
    threads calling global random APIs; one pipeline call at a time is required.
    """
    seed = validate_seed(seed)
    with _RNG_LOCK:
        numpy_state, python_state = np.random.get_state(), random.getstate()
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        try:
            with torch.random.fork_rng(devices=devices):
                seed_everything(seed)
                yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


class CaseRandomStreams:
    """Separate named generators and call counters, recreated for every rollout."""

    def __init__(self, seed: int):
        self.seed = validate_seed(seed)
        self.calls = {}
        self.generators = {}

    def generator(self, stream: str) -> torch.Generator:
        if stream not in self.generators:
            self.generators[stream] = torch.Generator("cpu").manual_seed(derive_seed(self.seed, stream))
        return self.generators[stream]

    @contextmanager
    def scope(self, stream: str):
        call = self.calls.get(stream, 0)
        self.calls[stream] = call + 1
        with isolated_rng(derive_seed(self.seed, stream, call)):
            yield

    def describe(self) -> dict:
        names = sorted(set(self.calls) | set(self.generators))
        return {"version": "octworld-rng-v1", "case_seed": self.seed,
                "derivation": "sha256('octworld-rng-v1:{seed}:{stream}:{call}')[:8], little-endian mod 2**63",
                "streams": {name: {"initial_seed": derive_seed(self.seed, name),
                                    "scoped_calls": self.calls.get(name, 0),
                                    "has_cpu_generator": name in self.generators}
                            for name in names}}