
# Four-Deck DJ Mixer Prototype

A Python desktop DJ mixer prototype focused on reliable playback and a clear,
maintainable audio architecture. The initial version does not include AI or
analytics.

## Core requirements

- Four independent decks: A, B, C, and D.
- Each deck plays a main track and has four sub-track slots. The main track
  and its sub-tracks can play in parallel.
- Sub-tracks have independent play/stop, volume, mute, and solo controls.
- Deck transport includes play/pause, cue, seek, looping, and Sync on/off.
- A synced deck follows a selected master deck's tempo and beat phase. With
  Sync off, it follows its own timing.
- Each deck has its own ordered FX chain. FX are applied to the deck output
  after the main track and sub-tracks are summed; sub-tracks do not have FX
  chains.
- The mixer provides per-deck gain, EQ, level meters, a crossfader, and master
  output.
- The library imports and searches audio, and supports loading tracks to decks
  and samples to sub-track slots. Deck and slot assignments reference library
  media instead of duplicating it.
- Initial file support is WAV and MP3. The decoder is isolated behind a media
  adapter so additional formats supported by the chosen or future decoder can
  be added without changing deck, mixer, or session models.
- Sessions save and restore media assignments, mixer settings, Sync state, and
  deck FX settings.
- Missing media, unsupported files, and unavailable audio devices are reported
  clearly.

## Audio routing

```text
Deck main track + four sub-tracks
                |
                v
            Deck sum
                |
                v
        Deck FX chain (A/B/C/D)
                |
                v
  Mixer channel: gain / EQ / meter
                |
                v
           Crossfader
                |
                v
          Master output
```

Sync controls playback timing; it is not an audio-processing stage. FX are
applied only to the summed deck output.

## Terminal-first development

Develop and validate the audio core from the terminal before building the
graphical interface. The first command-line tools inspect WAV/MP3 files and
render a test mix from deck main tracks and sub-track slots to WAV. This makes
decoder, routing, and effects behavior testable without UI debugging.

The eventual desktop interface uses one workspace with four compact deck
panels in a 2-by-2 layout, a shared mixer and crossfader, and a collapsible
library. Each deck panel keeps track information, waveform/progress,
transport, tempo and Sync, four sub-track rows, and its FX chain visible.
Detailed FX controls and audio-device settings can be kept in expandable
sections or dialogs.

## Architecture and future extensions

The project separates responsibilities by dependency direction:

```text
CLI / future UI
       |
       v
Application services
      / \
     v   v
 Domain  Audio renderer <--- decoder and FX implementations
             |
             v
           Domain
```

`aideck/domain/` holds plain Python deck and mixer state, including media IDs,
the four sub-track slots, gain/mute/solo/playing controls, Sync state,
crossfader position, and serializable FX configuration. It does not import
NumPy, audio buffers, file decoders, or UI modules.

`aideck/audio/` contains decoded audio buffers, the media decoder adapter,
effects, and rendering. The renderer combines domain state with decoded media
buffers supplied by its caller. The audio engine never opens or decodes files.
Adding a format should require decoder support and import validation, not
changes to deck or mixer logic.

`aideck/application/` contains the media library service and SQLite catalog,
plus versioned session persistence. `aideck/audio/` maps supported FX
configurations to DSP processors. The CLI is a headless application boundary
for exercising these components; the desktop UI will use the same services
rather than reach into audio internals. Audio processing must not block on UI
work, file access, or network operations.

Future AI generation and analytics should run asynchronously outside the
real-time audio callback and return results for preview or use. A future live
AI DSP effect would instead implement the same real-time constraints as other
deck FX and be loaded into that deck's FX chain. Neither capability is part of
the initial prototype.

## Technology

- **PySide6** for the desktop interface.
- **sounddevice / PortAudio** for audio device I/O.
- **miniaudio** for decoding initial WAV and MP3 media.
- **NumPy** for audio buffers and sample processing.

These are starting dependencies, not a commitment to a specific audio-engine
implementation. Keep decoding behind an adapter so another decoder can be
added or substituted when additional formats are needed. The first technical
milestone is to validate low-latency, dropout-free playback and routing before
expanding the FX set.

## Terminal workflow

Run commands from the repository root with the project environment activated:

```bash
python -m aideck inspect path/to/track.mp3
python -m aideck library import path/to/track.wav path/to/loop.mp3
python -m aideck library search loop
python -m aideck render \
  --deck A=path/to/track.wav \
  --slot A:1=path/to/loop.mp3 \
  --deck B=path/to/another-track.wav \
  --crossfader 0.5 \
  --output mix.wav \
  --save-session set.json
python -m aideck session show set.json
python -m aideck render --session set.json --output restored-mix.wav
python -m unittest discover -s tests -v
```

The renderer is an offline development/test tool, not the real-time audio
device player. It exercises the shared decoder and mixer logic without
requiring an audio device. The initial renderer supports the built-in `gain`
and `delay` effect types; unsupported enabled effect types are reported
explicitly.

For this prototype, sessions store absolute paths to source media and do not
automatically relink moved files. If a source file is moved or renamed, render
reports the original missing path; restore the file there or load the media
again in a future session. A relocatable media-root scheme can be added later.

## Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Linux, PortAudio's shared library may also need to be installed using the
operating system's package manager. Audio-device availability depends on the
host system.

The first headless audio-core implementation is available as `python -m
aideck`; the graphical interface and real-time device playback remain future
work.
