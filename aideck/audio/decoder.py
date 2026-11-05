from pathlib import Path
from dataclasses import dataclass
from typing import Protocol

import miniaudio
import numpy as np

from aideck.audio.buffer import AudioBuffer


@dataclass(frozen=True)
class AudioFileInfo:
    duration_seconds: float
    sample_rate: int
    channel_count: int


class AudioDecoder(Protocol):
    def decode(self, path: str | Path) -> AudioBuffer: ...

    def inspect(self, path: str | Path) -> AudioFileInfo: ...


class AudioDecodeError(Exception):
    """Raised when a media file cannot be decoded for mixer playback."""


class MiniaudioDecoder:
    SUPPORTED_SUFFIXES = {".wav", ".mp3"}

    def __init__(self, sample_rate: int = 44100) -> None:
        if sample_rate <= 0:
            raise ValueError("Sample rate must be a positive integer.")
        self.sample_rate = sample_rate

    def inspect(self, path: str | Path) -> AudioFileInfo:
        media_path = self._validate_path(path)
        try:
            info = miniaudio.get_file_info(str(media_path))
        except (miniaudio.DecodeError, OSError, ValueError) as error:
            raise AudioDecodeError(f"Could not inspect {media_path}: {error}") from error
        return AudioFileInfo(
            duration_seconds=info.duration,
            sample_rate=info.sample_rate,
            channel_count=info.nchannels,
        )

    def decode(self, path: str | Path) -> AudioBuffer:
        media_path = self._validate_path(path)

        try:
            decoded = miniaudio.decode_file(
                str(media_path),
                output_format=miniaudio.SampleFormat.FLOAT32,
                nchannels=2,
                sample_rate=self.sample_rate,
            )
        except (miniaudio.DecodeError, OSError, ValueError) as error:
            raise AudioDecodeError(f"Could not decode {media_path}: {error}") from error

        samples = np.asarray(decoded.samples, dtype=np.float32).reshape(-1, 2)
        return AudioBuffer(samples=samples, sample_rate=self.sample_rate)

    def _validate_path(self, path: str | Path) -> Path:
        media_path = Path(path).expanduser()
        if not media_path.is_file():
            raise AudioDecodeError(f"Audio file does not exist: {media_path}")
        if media_path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            supported = ", ".join(sorted(self.SUPPORTED_SUFFIXES))
            raise AudioDecodeError(
                f"Unsupported audio format for {media_path}; supported formats: {supported}."
            )
        return media_path.resolve()
