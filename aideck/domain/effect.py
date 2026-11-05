from dataclasses import dataclass, field
import math


@dataclass
class EffectState:
    effect_type: str
    enabled: bool = True
    parameters: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.effect_type:
            raise ValueError("Effect type cannot be empty.")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for name, value in self.parameters.items()
        ):
            raise ValueError("Effect parameters must be named finite numbers.")
