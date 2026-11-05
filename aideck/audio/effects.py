from typing import Protocol

import numpy as np

from aideck.audio.buffer import FloatAudio
from aideck.domain.effect import EffectState


class AudioEffect(Protocol):
    def process(self, samples: FloatAudio, sample_rate: int) -> FloatAudio: ...


class GainEffect:
    def __init__(self, gain: float = 1.0) -> None:
        if not np.isfinite(gain) or gain < 0:
            raise ValueError("Effect gain must be a finite, non-negative value.")
        self.gain = gain

    def process(self, samples: FloatAudio, sample_rate: int) -> FloatAudio:
        return samples * np.float32(self.gain)


class DelayEffect:
    def __init__(self, delay_seconds: float = 0.25, wet: float = 0.3) -> None:
        if not np.isfinite(delay_seconds) or delay_seconds < 0:
            raise ValueError("Delay time must be finite and non-negative.")
        if not np.isfinite(wet) or not 0 <= wet <= 1:
            raise ValueError("Delay wet level must be between 0 and 1.")
        self.delay_seconds = delay_seconds
        self.wet = wet

    def process(self, samples: FloatAudio, sample_rate: int) -> FloatAudio:
        delay_frames = round(self.delay_seconds * sample_rate)
        if delay_frames == 0 or delay_frames >= samples.shape[0] or self.wet == 0:
            return samples.copy()

        output = samples.copy()
        output[delay_frames:] += samples[:-delay_frames] * np.float32(self.wet)
        return output


def build_effect(state: EffectState) -> AudioEffect | None:
    if not state.enabled:
        return None
    if state.effect_type == "gain":
        if set(state.parameters) - {"gain"}:
            raise ValueError("Gain effect only accepts the 'gain' parameter.")
        return GainEffect(state.parameters.get("gain", 1.0))
    if state.effect_type == "delay":
        if set(state.parameters) - {"delay_seconds", "wet"}:
            raise ValueError(
                "Delay effect accepts 'delay_seconds' and 'wet' parameters."
            )
        return DelayEffect(
            delay_seconds=state.parameters.get("delay_seconds", 0.25),
            wet=state.parameters.get("wet", 0.3),
        )
    raise ValueError(f"Unsupported audio effect type: {state.effect_type!r}.")
