import tempfile
import subprocess
import sys
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from aideck.audio.buffer import AudioBuffer
from aideck.audio.decoder import AudioDecodeError, AudioFileInfo, MiniaudioDecoder
from aideck.audio.effects import GainEffect, build_effect
from aideck.audio.mixer import EmptyMixerError, render_mix
from aideck.cli import main
from aideck.application.media_library import (
    MediaLibraryService,
    SQLiteMediaCatalog,
)
from aideck.application.session_store import SessionFormatError, SessionStore
from aideck.domain.deck import DeckState, TrackLayerState
from aideck.domain.effect import EffectState
from aideck.domain.mixer import MixerState


SAMPLE_RATE = 44100
MP3_FIXTURE = Path(__file__).parent / "fixtures" / "tone.mp3"


def make_buffer(samples: list[tuple[float, float]]) -> AudioBuffer:
    return AudioBuffer(np.asarray(samples, dtype=np.float32), SAMPLE_RATE)


def write_test_wav(path: Path, samples: np.ndarray) -> None:
    pcm16 = np.rint(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output_file:
        output_file.setnchannels(2)
        output_file.setsampwidth(2)
        output_file.setframerate(SAMPLE_RATE)
        output_file.writeframes(pcm16.tobytes())


class AudioBufferTests(unittest.TestCase):
    def test_requires_stereo_finite_samples_and_positive_rate(self) -> None:
        with self.assertRaises(ValueError):
            AudioBuffer(np.zeros((8, 1), dtype=np.float32), SAMPLE_RATE)
        with self.assertRaises(ValueError):
            AudioBuffer(np.array([[np.nan, 0]], dtype=np.float32), SAMPLE_RATE)
        with self.assertRaises(ValueError):
            AudioBuffer(np.zeros((1, 2), dtype=np.float32), 0)


class MixerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mixer = MixerState()
        self.media: dict[str, AudioBuffer] = {}

    def assign(
        self,
        layer: TrackLayerState,
        media_id: str,
        samples: list[tuple[float, float]],
    ) -> None:
        layer.media_id = media_id
        self.media[media_id] = make_buffer(samples)

    def test_sums_deck_main_track_and_sub_tracks_before_effects(self) -> None:
        deck = self.mixer.decks["A"]
        self.assign(deck.main_track, "main", [(0.2, 0.2)])
        self.assign(deck.sub_tracks[0], "loop", [(0.3, 0.3)])
        self.mixer.crossfader = 0.0

        output = render_mix(
            self.mixer,
            self.media,
            effects_by_deck={"A": [GainEffect(0.5)]},
        )

        np.testing.assert_allclose(output.samples, [[0.25, 0.25]])

    def test_muted_and_solo_layers_are_respected(self) -> None:
        deck = self.mixer.decks["A"]
        self.assign(deck.main_track, "main", [(0.2, 0.2)])
        self.assign(deck.sub_tracks[0], "solo", [(0.4, 0.4)])
        self.assign(deck.sub_tracks[1], "muted", [(0.8, 0.8)])
        deck.sub_tracks[0].solo = True
        deck.sub_tracks[1].muted = True
        self.mixer.crossfader = 0.0

        output = render_mix(self.mixer, self.media)

        np.testing.assert_allclose(output.samples, [[0.4, 0.4]])

    def test_crossfader_routes_left_and_right_decks(self) -> None:
        self.assign(
            self.mixer.decks["A"].main_track, "left", [(0.5, 0.5)]
        )
        self.assign(
            self.mixer.decks["B"].main_track, "right", [(0.5, 0.5)]
        )

        self.mixer.crossfader = 0.0
        left_only = render_mix(self.mixer, self.media)
        self.mixer.crossfader = 1.0
        right_only = render_mix(self.mixer, self.media)

        np.testing.assert_allclose(left_only.samples, [[0.5, 0.5]])
        np.testing.assert_allclose(right_only.samples, [[0.5, 0.5]])

    def test_empty_mixer_is_reported(self) -> None:
        with self.assertRaises(EmptyMixerError):
            render_mix(self.mixer, self.media)

    def test_deck_has_exactly_four_sub_tracks(self) -> None:
        with self.assertRaises(ValueError):
            DeckState("A", sub_tracks=[])

    def test_missing_assigned_media_is_reported(self) -> None:
        self.mixer.decks["A"].main_track.media_id = "missing"
        with self.assertRaisesRegex(ValueError, "not available"):
            render_mix(self.mixer, self.media)

    def test_effect_registry_builds_supported_effects_and_rejects_unknown(self) -> None:
        effect = build_effect(EffectState("gain", parameters={"gain": 0.5}))
        self.assertIsInstance(effect, GainEffect)
        self.assertIsNone(build_effect(EffectState("future", enabled=False)))
        with self.assertRaisesRegex(ValueError, "Unsupported audio effect"):
            build_effect(EffectState("future"))


class DomainTests(unittest.TestCase):
    def test_mixer_state_has_four_decks_and_validates_crossfader(self) -> None:
        mixer = MixerState()
        self.assertEqual(tuple(mixer.decks), ("A", "B", "C", "D"))
        with self.assertRaises(ValueError):
            MixerState(crossfader=1.1)
        with self.assertRaises(ValueError):
            MixerState(sync_master="E")

    def test_deck_has_independent_sub_track_state(self) -> None:
        deck = DeckState("a")
        deck.sub_tracks[0].muted = True
        self.assertEqual(deck.name, "A")
        self.assertFalse(deck.sub_tracks[1].muted)


class FakeDecoder:
    def inspect(self, path: str | Path) -> AudioFileInfo:
        return AudioFileInfo(2.5, SAMPLE_RATE, 2)

    def decode(self, path: str | Path) -> AudioBuffer:
        return make_buffer([(0.25, 0.25)])


class MediaLibraryTests(unittest.TestCase):
    def test_import_search_deduplicates_and_assigns_media(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            media_path = Path(directory) / "Loop.wav"
            write_test_wav(media_path, np.array([[0.1, 0.1]], dtype=np.float32))
            catalog = SQLiteMediaCatalog(Path(directory) / "library.sqlite3")
            service = MediaLibraryService(catalog, FakeDecoder())

            first = service.import_file(media_path)
            duplicate = service.import_file(media_path)
            matches = service.search("loop")
            deck = DeckState("A")
            service.assign_to_deck(deck, first.media_id)
            service.assign_to_sub_track(deck, 2, first.media_id)

            self.assertEqual(first, duplicate)
            self.assertEqual([first], matches)
            self.assertEqual(str(media_path.resolve()), deck.main_track.media_id)
            self.assertEqual(str(media_path.resolve()), deck.sub_tracks[2].media_id)

    def test_assign_rejects_unknown_media_and_invalid_slot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = MediaLibraryService(
                SQLiteMediaCatalog(Path(directory) / "library.sqlite3"),
                FakeDecoder(),
            )
            deck = DeckState("A")
            with self.assertRaisesRegex(ValueError, "not in the library"):
                service.assign_to_deck(deck, "missing")
            with self.assertRaisesRegex(ValueError, "between 1 and 4"):
                service.assign_to_sub_track(deck, 4, "missing")


class SessionTests(unittest.TestCase):
    def test_round_trip_preserves_mixer_deck_slot_and_effect_state(self) -> None:
        mixer = MixerState(crossfader=0.25, sync_master="A")
        deck = mixer.decks["B"]
        deck.main_track.media_id = "/media/main.mp3"
        deck.main_track.gain = 0.75
        deck.sub_tracks[1].media_id = "/media/loop.wav"
        deck.sub_tracks[1].muted = True
        deck.channel_gain = 0.8
        deck.sync_enabled = True
        deck.effects.append(
            EffectState("delay", enabled=False, parameters={"wet": 0.35})
        )

        with tempfile.TemporaryDirectory() as directory:
            session_path = Path(directory) / "set.json"
            store = SessionStore()
            store.save(session_path, mixer)
            restored = store.load(session_path)

        self.assertEqual(restored.crossfader, 0.25)
        self.assertEqual(restored.sync_master, "A")
        restored_deck = restored.decks["B"]
        self.assertEqual(restored_deck.main_track.media_id, "/media/main.mp3")
        self.assertEqual(restored_deck.main_track.gain, 0.75)
        self.assertEqual(restored_deck.sub_tracks[1].media_id, "/media/loop.wav")
        self.assertTrue(restored_deck.sub_tracks[1].muted)
        self.assertEqual(restored_deck.channel_gain, 0.8)
        self.assertTrue(restored_deck.sync_enabled)
        self.assertEqual(restored_deck.effects[0].effect_type, "delay")
        self.assertFalse(restored_deck.effects[0].enabled)
        self.assertEqual(restored_deck.effects[0].parameters, {"wet": 0.35})

    def test_rejects_invalid_and_unsupported_session_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session_path = Path(directory) / "broken.json"
            store = SessionStore()
            session_path.write_text("{", encoding="utf-8")
            with self.assertRaises(SessionFormatError):
                store.load(session_path)
            session_path.write_text('{"version": 99}', encoding="utf-8")
            with self.assertRaisesRegex(SessionFormatError, "Unsupported session"):
                store.load(session_path)


class DecoderTests(unittest.TestCase):
    def test_decodes_wav_to_stereo_float_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tone.wav"
            write_test_wav(path, np.array([[0.25, -0.25]], dtype=np.float32))

            decoded = MiniaudioDecoder().decode(path)

        self.assertEqual(decoded.sample_rate, SAMPLE_RATE)
        self.assertEqual(decoded.samples.shape, (1, 2))
        np.testing.assert_allclose(decoded.samples, [[0.25, -0.25]], atol=1e-4)

    def test_decodes_mp3_fixture_to_stereo_float_audio(self) -> None:
        decoded = MiniaudioDecoder().decode(MP3_FIXTURE)
        info = MiniaudioDecoder().inspect(MP3_FIXTURE)

        self.assertEqual(decoded.sample_rate, SAMPLE_RATE)
        self.assertEqual(decoded.samples.ndim, 2)
        self.assertEqual(decoded.samples.shape[1], 2)
        self.assertGreater(decoded.frame_count, 0)
        self.assertGreater(info.duration_seconds, 0.05)
        self.assertEqual(info.channel_count, 2)

    def test_inspects_wav_metadata_without_loading_audio_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tone.wav"
            write_test_wav(
                path,
                np.array([[0.25, -0.25], [0.1, -0.1]], dtype=np.float32),
            )

            info = MiniaudioDecoder().inspect(path)

        self.assertAlmostEqual(info.duration_seconds, 2 / SAMPLE_RATE)
        self.assertEqual(info.sample_rate, SAMPLE_RATE)
        self.assertEqual(info.channel_count, 2)

    def test_rejects_missing_and_unsupported_media(self) -> None:
        decoder = MiniaudioDecoder()
        with self.assertRaises(AudioDecodeError):
            decoder.decode("missing.wav")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "track.flac"
            path.touch()
            with self.assertRaises(AudioDecodeError):
                decoder.decode(path)


class CommandLineTests(unittest.TestCase):
    def test_documented_terminal_workflow_with_wav_and_mp3(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav_path = root / "loop.wav"
            database_path = root / "library.sqlite3"
            session_path = root / "set.json"
            first_mix_path = root / "mix.wav"
            restored_mix_path = root / "restored-mix.wav"
            write_test_wav(
                wav_path,
                np.array([[0.15, 0.15]] * 100, dtype=np.float32),
            )
            project_root = Path(__file__).resolve().parents[1]

            def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [sys.executable, "-m", "aideck", *arguments],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )

            inspect_result = run_cli("inspect", str(MP3_FIXTURE))
            import_result = run_cli(
                "library",
                "--database",
                str(database_path),
                "import",
                str(wav_path),
                str(MP3_FIXTURE),
            )
            search_result = run_cli(
                "library",
                "--database",
                str(database_path),
                "search",
                "tone",
            )
            render_result = run_cli(
                "render",
                "--deck",
                f"A={MP3_FIXTURE}",
                "--slot",
                f"A:1={wav_path}",
                "--crossfader",
                "0",
                "--output",
                str(first_mix_path),
                "--save-session",
                str(session_path),
            )
            show_result = run_cli("session", "show", str(session_path))
            restore_result = run_cli(
                "render",
                "--session",
                str(session_path),
                "--output",
                str(restored_mix_path),
            )

            for result in (
                inspect_result,
                import_result,
                search_result,
                render_result,
                show_result,
                restore_result,
            ):
                self.assertEqual(
                    result.returncode,
                    0,
                    msg=f"command failed: {result.args}\n{result.stderr}",
                )
            self.assertIn("Frames:", inspect_result.stdout)
            self.assertIn("[mp3]", import_result.stdout)
            self.assertIn("tone", search_result.stdout)
            self.assertIn("Saved session", render_result.stdout)
            self.assertIn("Deck A:", show_result.stdout)
            with wave.open(str(first_mix_path), "rb") as first:
                first_frames = first.readframes(first.getnframes())
            with wave.open(str(restored_mix_path), "rb") as restored:
                restored_frames = restored.readframes(restored.getnframes())
            self.assertEqual(first_frames, restored_frames)

    def test_inspect_command_reports_file_details(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.wav"
            write_test_wav(input_path, np.array([[0.5, -0.5]], dtype=np.float32))
            output = StringIO()

            with redirect_stdout(output):
                result = main(["inspect", str(input_path)])

        self.assertEqual(result, 0)
        self.assertIn("Decoded format: stereo float32, 44100 Hz", output.getvalue())

    def test_inspect_command_reports_mp3_details(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            result = main(["inspect", str(MP3_FIXTURE)])

        self.assertEqual(result, 0)
        self.assertIn("File:", output.getvalue())
        self.assertIn("Frames:", output.getvalue())

    def test_render_command_writes_mixed_wav(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.wav"
            output_path = Path(directory) / "mix.wav"
            write_test_wav(input_path, np.array([[0.5, -0.5]], dtype=np.float32))

            with redirect_stdout(StringIO()):
                result = main(
                    [
                        "render",
                        "--deck",
                        f"A={input_path}",
                        "--crossfader",
                        "0",
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(result, 0)
            with wave.open(str(output_path), "rb") as rendered:
                self.assertEqual(rendered.getnchannels(), 2)
                self.assertEqual(rendered.getframerate(), SAMPLE_RATE)
                self.assertEqual(rendered.getnframes(), 1)

    def test_render_command_mixes_mp3_and_wav_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "loop.wav"
            output_path = Path(directory) / "mp3-wav-mix.wav"
            write_test_wav(
                wav_path,
                np.array([[0.15, 0.15]] * 100, dtype=np.float32),
            )

            with redirect_stdout(StringIO()):
                result = main(
                    [
                        "render",
                        "--deck",
                        f"A={MP3_FIXTURE}",
                        "--slot",
                        f"A:1={wav_path}",
                        "--crossfader",
                        "0",
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(result, 0)
            with wave.open(str(output_path), "rb") as rendered:
                self.assertEqual(rendered.getnchannels(), 2)
                self.assertEqual(rendered.getframerate(), SAMPLE_RATE)
                self.assertGreater(rendered.getnframes(), 0)

    def test_rendered_mix_can_be_saved_and_loaded_as_a_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.wav"
            output_path = Path(directory) / "first-mix.wav"
            restored_path = Path(directory) / "restored-mix.wav"
            session_path = Path(directory) / "set.json"
            write_test_wav(input_path, np.array([[0.5, -0.5]], dtype=np.float32))

            with redirect_stdout(StringIO()):
                save_result = main(
                    [
                        "render",
                        "--deck",
                        f"A={input_path}",
                        "--crossfader",
                        "0",
                        "--output",
                        str(output_path),
                        "--save-session",
                        str(session_path),
                    ]
                )
                restore_result = main(
                    [
                        "render",
                        "--session",
                        str(session_path),
                        "--output",
                        str(restored_path),
                    ]
                )

            self.assertEqual(save_result, 0)
            self.assertEqual(restore_result, 0)
            with wave.open(str(output_path), "rb") as first:
                first_frames = first.readframes(first.getnframes())
            with wave.open(str(restored_path), "rb") as restored:
                restored_frames = restored.readframes(restored.getnframes())
            self.assertEqual(first_frames, restored_frames)

    def test_moved_session_media_reports_the_original_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            media_path = Path(directory) / "original.wav"
            moved_path = Path(directory) / "moved.wav"
            initial_output = Path(directory) / "initial.wav"
            restored_output = Path(directory) / "restored.wav"
            session_path = Path(directory) / "set.json"
            write_test_wav(
                media_path, np.array([[0.5, 0.5]], dtype=np.float32)
            )

            with redirect_stdout(StringIO()):
                save_result = main(
                    [
                        "render",
                        "--deck",
                        f"A={media_path}",
                        "--crossfader",
                        "0",
                        "--output",
                        str(initial_output),
                        "--save-session",
                        str(session_path),
                    ]
                )
            media_path.rename(moved_path)
            error_output = StringIO()
            with redirect_stderr(error_output):
                restore_result = main(
                    [
                        "render",
                        "--session",
                        str(session_path),
                        "--output",
                        str(restored_output),
                    ]
                )

            self.assertEqual(save_result, 0)
            self.assertEqual(restore_result, 2)
            self.assertIn(str(media_path.resolve()), error_output.getvalue())
            self.assertIn("does not exist", error_output.getvalue())
            self.assertFalse(restored_output.exists())

    def test_session_render_applies_saved_enabled_effects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.wav"
            output_path = Path(directory) / "effect-mix.wav"
            session_path = Path(directory) / "effect-set.json"
            write_test_wav(input_path, np.array([[0.5, 0.5]], dtype=np.float32))
            mixer = MixerState(crossfader=0.0)
            mixer.decks["A"].main_track.media_id = str(input_path.resolve())
            mixer.decks["A"].effects.append(
                EffectState("gain", parameters={"gain": 0.5})
            )
            SessionStore().save(session_path, mixer)

            with redirect_stdout(StringIO()):
                result = main(
                    [
                        "render",
                        "--session",
                        str(session_path),
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(result, 0)
            with wave.open(str(output_path), "rb") as rendered:
                samples = np.frombuffer(rendered.readframes(1), dtype="<i2")
            np.testing.assert_allclose(samples / 32767, [0.25, 0.25], atol=1e-4)

    def test_library_import_and_search_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            media_path = Path(directory) / "library-test.wav"
            database_path = Path(directory) / "library.sqlite3"
            write_test_wav(media_path, np.array([[0.1, 0.1]], dtype=np.float32))
            output = StringIO()

            with redirect_stdout(output):
                import_result = main(
                    [
                        "library",
                        "--database",
                        str(database_path),
                        "import",
                        str(media_path),
                    ]
                )
                search_result = main(
                    [
                        "library",
                        "--database",
                        str(database_path),
                        "search",
                        "library-test",
                    ]
                )

            self.assertEqual(import_result, 0)
            self.assertEqual(search_result, 0)
            self.assertIn("Imported library-test", output.getvalue())
            self.assertIn("library-test", output.getvalue())

    def test_library_imports_both_wav_and_mp3(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "library-test.wav"
            database_path = Path(directory) / "library.sqlite3"
            write_test_wav(wav_path, np.array([[0.1, 0.1]], dtype=np.float32))
            output = StringIO()

            with redirect_stdout(output):
                result = main(
                    [
                        "library",
                        "--database",
                        str(database_path),
                        "import",
                        str(wav_path),
                        str(MP3_FIXTURE),
                    ]
                )

            self.assertEqual(result, 0)
            self.assertIn("Imported library-test [wav]", output.getvalue())
            self.assertIn("Imported tone [mp3]", output.getvalue())

    def test_session_show_command_reports_deck_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session_path = Path(directory) / "set.json"
            mixer = MixerState()
            mixer.decks["D"].main_track.media_id = "/media/track.wav"
            SessionStore().save(session_path, mixer)
            output = StringIO()

            with redirect_stdout(output):
                result = main(["session", "show", str(session_path)])

        self.assertEqual(result, 0)
        self.assertIn("Deck D: main=/media/track.wav", output.getvalue())


if __name__ == "__main__":
    unittest.main()
