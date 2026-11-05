from collections.abc import Mapping, Sequence
import math

import numpy as np

from aideck.audio.buffer import AudioBuffer
from aideck.audio.effects import AudioEffect
from aideck.domain.deck import TrackLayerState
from aideck.domain.mixer import MixerState


class EmptyMixerError(ValueError):
    """Raised when there is no playing audio to render."""


def render_mix(
    mixer: MixerState,
    media: Mapping[str, AudioBuffer],
    sample_rate: int = 44100,
    effects_by_deck: Mapping[str, Sequence[AudioEffect]] | None = None,
) -> AudioBuffer:
    if sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer.")
    if not math.isfinite(mixer.crossfader) or not 0 <= mixer.crossfader <= 1:
        raise ValueError("Crossfader position must be between 0 and 1.")

    effects_by_deck = effects_by_deck or {}
    active_layers: dict[str, list[tuple[TrackLayerState, AudioBuffer]]] = {}
    frame_count = 0
    for deck in mixer.decks.values():
        loaded_layers: list[tuple[TrackLayerState, AudioBuffer]] = []
        for layer in deck.layers():
            if layer.media_id is None or not layer.playing:
                continue
            audio = media.get(layer.media_id)
            if audio is None:
                raise ValueError(
                    f"Media {layer.media_id!r} assigned to deck {deck.name} "
                    "is not available."
                )
            loaded_layers.append((layer, audio))

        if any(layer.solo for layer, _ in loaded_layers):
            loaded_layers = [
                (layer, audio)
                for layer, audio in loaded_layers
                if layer.solo
            ]
        loaded_layers = [
            (layer, audio)
            for layer, audio in loaded_layers
            if not layer.muted
        ]
        active_layers[deck.name] = loaded_layers
        for _, audio in loaded_layers:
            if audio.sample_rate != sample_rate:
                raise ValueError(
                    f"Deck {deck.name} audio must use the mix sample rate "
                    f"({sample_rate} Hz)."
                )
            frame_count = max(frame_count, audio.frame_count)

    if frame_count == 0:
        raise EmptyMixerError("There is no playing audio loaded on the mixer.")

    mixed = np.zeros((frame_count, 2), dtype=np.float32)
    left_gain = np.float32(np.cos(mixer.crossfader * np.pi / 2))
    right_gain = np.float32(np.sin(mixer.crossfader * np.pi / 2))

    for deck in mixer.decks.values():
        layers = active_layers[deck.name]
        if not layers:
            continue

        deck_sum = np.zeros((frame_count, 2), dtype=np.float32)
        for layer, buffer in layers:
            audio = buffer.samples
            deck_sum[: audio.shape[0]] += audio * np.float32(layer.gain)

        for effect in effects_by_deck.get(deck.name, ()):
            deck_sum = effect.process(deck_sum, sample_rate)
            if deck_sum.shape != (frame_count, 2):
                raise ValueError(
                    f"Effect on deck {deck.name} changed the audio buffer shape."
                )

        side_gain = left_gain if deck.name in ("A", "C") else right_gain
        mixed += deck_sum * np.float32(deck.channel_gain) * side_gain

    np.clip(mixed, -1.0, 1.0, out=mixed)
    return AudioBuffer(samples=mixed, sample_rate=sample_rate)
