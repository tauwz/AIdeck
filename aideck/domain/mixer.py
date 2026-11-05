from dataclasses import dataclass, field
import math

from aideck.domain.deck import DECK_NAMES, DeckState


@dataclass
class MixerState:
    decks: dict[str, DeckState] = field(
        default_factory=lambda: {name: DeckState(name) for name in DECK_NAMES}
    )
    crossfader: float = 0.5
    sync_master: str | None = None

    def __post_init__(self) -> None:
        if set(self.decks) != set(DECK_NAMES):
            raise ValueError(f"Mixer must contain decks {', '.join(DECK_NAMES)}.")
        if any(deck.name != name for name, deck in self.decks.items()):
            raise ValueError("Each mixer deck must match its deck name.")
        if not math.isfinite(self.crossfader) or not 0 <= self.crossfader <= 1:
            raise ValueError("Crossfader position must be between 0 and 1.")
        if self.sync_master is not None and self.sync_master not in self.decks:
            raise ValueError("Sync master must refer to an existing deck.")
