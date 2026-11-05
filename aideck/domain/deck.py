from dataclasses import dataclass, field
import math

from aideck.domain.effect import EffectState


DECK_NAMES = ("A", "B", "C", "D")
SUB_TRACK_COUNT = 4
MediaId = str


@dataclass
class TrackLayerState:
    media_id: MediaId | None = None
    gain: float = 1.0
    playing: bool = True
    muted: bool = False
    solo: bool = False

    def __post_init__(self) -> None:
        if self.media_id is not None and not self.media_id:
            raise ValueError("Media ID cannot be empty.")
        if not math.isfinite(self.gain) or self.gain < 0:
            raise ValueError("Layer gain must be a finite, non-negative value.")


@dataclass
class DeckState:
    name: str
    main_track: TrackLayerState = field(default_factory=TrackLayerState)
    sub_tracks: list[TrackLayerState] = field(
        default_factory=lambda: [
            TrackLayerState() for _ in range(SUB_TRACK_COUNT)
        ]
    )
    channel_gain: float = 1.0
    sync_enabled: bool = False
    effects: list[EffectState] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = self.name.upper()
        if self.name not in DECK_NAMES:
            raise ValueError(f"Deck name must be one of {', '.join(DECK_NAMES)}.")
        if len(self.sub_tracks) != SUB_TRACK_COUNT:
            raise ValueError(
                f"Each deck must have exactly {SUB_TRACK_COUNT} sub-tracks."
            )
        if not math.isfinite(self.channel_gain) or self.channel_gain < 0:
            raise ValueError("Deck channel gain must be a finite, non-negative value.")

    def layers(self) -> list[TrackLayerState]:
        return [self.main_track, *self.sub_tracks]
