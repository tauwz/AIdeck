import argparse
import sys
import wave
from pathlib import Path

import numpy as np

from aideck.audio.buffer import AudioBuffer
from aideck.audio.decoder import AudioDecodeError, MiniaudioDecoder
from aideck.audio.effects import build_effect
from aideck.audio.mixer import render_mix
from aideck.application.media_library import MediaLibraryService, SQLiteMediaCatalog
from aideck.application.session_store import SessionStore
from aideck.domain.deck import DECK_NAMES
from aideck.domain.mixer import MixerState


def _default_data_directory() -> Path:
    return Path.home() / ".local" / "share" / "aideck"


def _parse_assignment(value: str) -> tuple[str, str]:
    deck_name, separator, path = value.partition("=")
    deck_name = deck_name.upper()
    if not separator or deck_name not in DECK_NAMES or not path:
        raise argparse.ArgumentTypeError(
            "expected DECK=FILE, for example A=track.wav"
        )
    return deck_name, path


def _parse_slot_assignment(value: str) -> tuple[str, int, str]:
    slot, separator, path = value.partition("=")
    deck_name, colon, slot_number = slot.partition(":")
    deck_name = deck_name.upper()
    if (
        not separator
        or not colon
        or deck_name not in DECK_NAMES
        or not path
        or not slot_number.isdigit()
        or not 1 <= int(slot_number) <= 4
    ):
        raise argparse.ArgumentTypeError(
            "expected DECK:SLOT=FILE, for example A:1=loop.mp3"
        )
    return deck_name, int(slot_number) - 1, path


def _write_wav(path: Path, audio: AudioBuffer) -> None:
    pcm16 = np.rint(np.clip(audio.samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output_file:
        output_file.setnchannels(2)
        output_file.setsampwidth(2)
        output_file.setframerate(audio.sample_rate)
        output_file.writeframes(pcm16.tobytes())


def _inspect(path: str, decoder: MiniaudioDecoder) -> int:
    audio = decoder.decode(path)
    print(f"File: {Path(path)}")
    print(f"Duration: {audio.duration_seconds:.3f} s")
    print(f"Decoded format: stereo float32, {audio.sample_rate} Hz")
    print(f"Frames: {audio.frame_count}")
    return 0


def _render_mix(
    deck_assignments: list[tuple[str, str]],
    slot_assignments: list[tuple[str, int, str]],
    output_path: str,
    crossfader: float | None,
    decoder: MiniaudioDecoder,
    session_path: str | None = None,
    save_session_path: str | None = None,
) -> int:
    if session_path and (deck_assignments or slot_assignments):
        raise ValueError("--session cannot be combined with --deck or --slot.")

    session_store = SessionStore()
    mixer = (
        session_store.load(session_path)
        if session_path
        else MixerState(crossfader=0.5 if crossfader is None else crossfader)
    )
    if crossfader is not None:
        mixer.crossfader = crossfader
    media: dict[str, AudioBuffer] = {}

    if session_path:
        media_ids = {
            layer.media_id
            for deck in mixer.decks.values()
            for layer in deck.layers()
            if layer.media_id is not None and layer.playing
        }
        for media_id in media_ids:
            media[media_id] = decoder.decode(media_id)
    else:
        for deck_name, path in deck_assignments:
            deck = mixer.decks[deck_name]
            if deck.main_track.media_id is not None:
                raise ValueError(f"Deck {deck_name} has more than one main track.")
            media_id = str(Path(path).expanduser().resolve())
            deck.main_track.media_id = media_id
            media[media_id] = decoder.decode(media_id)

        for deck_name, slot_index, path in slot_assignments:
            layer = mixer.decks[deck_name].sub_tracks[slot_index]
            if layer.media_id is not None:
                raise ValueError(
                    f"Deck {deck_name} sub-track {slot_index + 1} is assigned twice."
                )
            media_id = str(Path(path).expanduser().resolve())
            layer.media_id = media_id
            if media_id not in media:
                media[media_id] = decoder.decode(media_id)

    effects_by_deck = {
        name: [
            effect
            for effect_state in deck.effects
            if (effect := build_effect(effect_state)) is not None
        ]
        for name, deck in mixer.decks.items()
    }
    audio = render_mix(
        mixer,
        media,
        decoder.sample_rate,
        effects_by_deck=effects_by_deck,
    )
    _write_wav(Path(output_path), audio)
    if save_session_path:
        session_store.save(save_session_path, mixer)
    print(
        f"Rendered {audio.duration_seconds:.3f} s to {output_path} "
        f"({audio.sample_rate} Hz stereo WAV)."
    )
    if save_session_path:
        print(f"Saved session to {save_session_path}.")
    return 0


def _library_command(args: argparse.Namespace, decoder: MiniaudioDecoder) -> int:
    database_path = args.database or _default_data_directory() / "library.sqlite3"
    service = MediaLibraryService(SQLiteMediaCatalog(database_path), decoder)
    if args.library_action == "import":
        for path in args.files:
            item = service.import_file(path)
            print(
                f"Imported {item.title} [{item.file_format}] "
                f"{item.duration_seconds:.2f}s ({item.media_id})"
            )
        return 0

    items = service.search(args.query)
    for item in items:
        print(
            f"{item.title}\t{item.file_format}\t"
            f"{item.duration_seconds:.2f}s\t{item.media_id}"
        )
    if not items:
        print("No matching media.")
    return 0


def _show_session(path: str) -> int:
    mixer = SessionStore().load(path)
    print(f"Session: {Path(path)}")
    print(f"Crossfader: {mixer.crossfader:.2f}")
    print(f"Sync master: {mixer.sync_master or 'none'}")
    for name, deck in mixer.decks.items():
        main_track = deck.main_track.media_id or "empty"
        slots = ", ".join(
            layer.media_id or "empty" for layer in deck.sub_tracks
        )
        print(
            f"Deck {name}: main={main_track}; sub-tracks=[{slots}]; "
            f"sync={'on' if deck.sync_enabled else 'off'}; "
            f"effects={len(deck.effects)}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aideck")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect", help="inspect a WAV or MP3 file")
    inspect_parser.add_argument("file")

    render_parser = commands.add_parser(
        "render", help="mix deck tracks and sub-tracks to a WAV file"
    )
    render_parser.add_argument(
        "--deck",
        action="append",
        default=[],
        type=_parse_assignment,
        metavar="DECK=FILE",
        help="load a main track; repeat for multiple decks",
    )
    render_parser.add_argument(
        "--slot",
        action="append",
        default=[],
        type=_parse_slot_assignment,
        metavar="DECK:SLOT=FILE",
        help="load a sub-track (slot 1-4); repeat as needed",
    )
    render_parser.add_argument("--crossfader", type=float)
    render_parser.add_argument("--output", required=True)
    render_parser.add_argument("--session", help="render a previously saved session")
    render_parser.add_argument(
        "--save-session", help="save the current deck assignments and mixer state"
    )

    library_parser = commands.add_parser("library", help="manage the media library")
    library_parser.add_argument(
        "--database",
        help="SQLite catalog path (defaults to the user's application data directory)",
    )
    library_commands = library_parser.add_subparsers(
        dest="library_action", required=True
    )
    import_parser = library_commands.add_parser("import", help="import WAV or MP3 media")
    import_parser.add_argument("files", nargs="+")
    search_parser = library_commands.add_parser("search", help="search imported media")
    search_parser.add_argument("query", nargs="?", default="")

    session_parser = commands.add_parser("session", help="inspect a saved session")
    session_commands = session_parser.add_subparsers(
        dest="session_action", required=True
    )
    show_parser = session_commands.add_parser("show", help="show saved session state")
    show_parser.add_argument("file")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    decoder = MiniaudioDecoder()

    try:
        if args.command == "inspect":
            return _inspect(args.file, decoder)
        if args.command == "render":
            return _render_mix(
                args.deck,
                args.slot,
                args.output,
                args.crossfader,
                decoder,
                args.session,
                args.save_session,
            )
        if args.command == "library":
            return _library_command(args, decoder)
        if args.command == "session":
            return _show_session(args.file)
        raise ValueError(f"Unknown command: {args.command}")
    except (AudioDecodeError, ValueError, OSError, wave.Error) as error:
        print(f"aideck: error: {error}", file=sys.stderr)
        return 2
