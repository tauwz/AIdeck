from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatAudio = NDArray[np.float32]


@dataclass(frozen=True)
class AudioBuffer:
    samples: FloatAudio
    sample_rate: int

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError("Audio must be a two-dimensional stereo buffer.")
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be a positive integer.")
        if not np.isfinite(samples).all():
            raise ValueError("Audio buffer contains non-finite samples.")
        object.__setattr__(self, "samples", np.ascontiguousarray(samples))

    @property
    def frame_count(self) -> int:
        return self.samples.shape[0]

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate
