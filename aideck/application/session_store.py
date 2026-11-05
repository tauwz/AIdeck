import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from aideck.domain.deck import DeckState, TrackLayerState
from aideck.domain.effect import EffectState
from aideck.domain.mixer import MixerState


SESSION_VERSION = 1


class SessionFormatError(ValueError):
    """Raised when a session file is malformed or unsupported."""


class SessionStore:
    def save(self, path: str | Path, mixer: MixerState) -> None:
        session_path = Path(path).expanduser().resolve()
        session_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _serialize_mixer(mixer)

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=session_path.parent,
                prefix=f".{session_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(payload, temporary_file, indent=2, sort_keys=True)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            temporary_path.replace(session_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def load(self, path: str | Path) -> MixerState:
        session_path = Path(path).expanduser()
        try:
            with session_path.open(encoding="utf-8") as session_file:
                payload = json.load(session_file)
        except (OSError, json.JSONDecodeError) as error:
            raise SessionFormatError(f"Could not read session {session_path}: {error}") from error
        try:
            return _deserialize_mixer(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise SessionFormatError(
                f"Invalid session file {session_path}: {error}"
            ) from error


def _serialize_layer(layer: TrackLayerState) -> dict[str, Any]:
    return {
        "media_id": layer.media_id,
        "gain": layer.gain,
        "playing": layer.playing,
        "muted": layer.muted,
        "solo": layer.solo,
    }


def _serialize_mixer(mixer: MixerState) -> dict[str, Any]:
    return {
        "version": SESSION_VERSION,
        "crossfader": mixer.crossfader,
        "sync_master": mixer.sync_master,
        "decks": {
            name: {
                "channel_gain": deck.channel_gain,
                "sync_enabled": deck.sync_enabled,
                "main_track": _serialize_layer(deck.main_track),
                "sub_tracks": [
                    _serialize_layer(layer) for layer in deck.sub_tracks
                ],
                "effects": [
                    {
                        "effect_type": effect.effect_type,
                        "enabled": effect.enabled,
                        "parameters": effect.parameters,
                    }
                    for effect in deck.effects
                ],
            }
            for name, deck in mixer.decks.items()
        },
    }


def _deserialize_layer(payload: dict[str, Any]) -> TrackLayerState:
    media_id = payload["media_id"]
    if media_id is not None and not isinstance(media_id, str):
        raise ValueError("Layer media_id must be a string or null.")
    return TrackLayerState(
        media_id=media_id,
        gain=_number(payload["gain"], "layer gain"),
        playing=_boolean(payload["playing"], "layer playing"),
        muted=_boolean(payload["muted"], "layer muted"),
        solo=_boolean(payload["solo"], "layer solo"),
    )


def _deserialize_mixer(payload: Any) -> MixerState:
    if not isinstance(payload, dict):
        raise ValueError("Session root must be an object.")
    version = payload["version"]
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != SESSION_VERSION
    ):
        raise ValueError(f"Unsupported session version {version!r}.")

    decks_payload = payload["decks"]
    if not isinstance(decks_payload, dict):
        raise ValueError("Session decks must be an object.")
    decks: dict[str, DeckState] = {}
    for name, deck_payload in decks_payload.items():
        if not isinstance(deck_payload, dict):
            raise ValueError(f"Deck {name} must be an object.")
        sub_tracks_payload = deck_payload["sub_tracks"]
        effects_payload = deck_payload["effects"]
        if not isinstance(sub_tracks_payload, list) or not isinstance(effects_payload, list):
            raise ValueError(f"Deck {name} sub-tracks and effects must be arrays.")
        effects = [
            EffectState(
                effect_type=_string(effect["effect_type"], "effect type"),
                enabled=_boolean(effect["enabled"], "effect enabled"),
                parameters=_parameters(effect["parameters"]),
            )
            for effect in effects_payload
        ]
        decks[name] = DeckState(
            name=name,
            main_track=_deserialize_layer(deck_payload["main_track"]),
            sub_tracks=[
                _deserialize_layer(layer) for layer in sub_tracks_payload
            ],
            channel_gain=_number(deck_payload["channel_gain"], "deck gain"),
            sync_enabled=_boolean(deck_payload["sync_enabled"], "deck sync"),
            effects=effects,
        )

    sync_master = payload["sync_master"]
    if sync_master is not None and not isinstance(sync_master, str):
        raise ValueError("Sync master must be a deck name or null.")
    return MixerState(
        decks=decks,
        crossfader=_number(payload["crossfader"], "crossfader"),
        sync_master=sync_master,
    )


def _number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number.")
    return float(value)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean.")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _parameters(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("Effect parameters must be an object.")
    return {
        _string(name, "effect parameter name"): _number(
            number, f"effect parameter {name}"
        )
        for name, number in value.items()
    }
