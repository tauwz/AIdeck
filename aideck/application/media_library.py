from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Protocol

from aideck.audio.decoder import AudioDecoder
from aideck.domain.deck import DeckState


@dataclass(frozen=True)
class MediaItem:
    media_id: str
    path: str
    title: str
    file_format: str
    duration_seconds: float
    sample_rate: int
    channel_count: int


class MediaCatalog(Protocol):
    def add(self, item: MediaItem) -> MediaItem: ...

    def get(self, media_id: str) -> MediaItem | None: ...

    def search(self, query: str = "") -> list[MediaItem]: ...


class SQLiteMediaCatalog:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS media (
                    media_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    file_format TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    sample_rate INTEGER NOT NULL,
                    channel_count INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS media_title_idx ON media(title)"
            )

    def add(self, item: MediaItem) -> MediaItem:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO media (
                    media_id, path, title, file_format, duration_seconds,
                    sample_rate, channel_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    title = excluded.title,
                    file_format = excluded.file_format,
                    duration_seconds = excluded.duration_seconds,
                    sample_rate = excluded.sample_rate,
                    channel_count = excluded.channel_count
                """,
                (
                    item.media_id,
                    item.path,
                    item.title,
                    item.file_format,
                    item.duration_seconds,
                    item.sample_rate,
                    item.channel_count,
                ),
            )
            row = connection.execute(
                "SELECT * FROM media WHERE path = ?", (item.path,)
            ).fetchone()
        return _media_item(row)

    def get(self, media_id: str) -> MediaItem | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM media WHERE media_id = ?", (media_id,)
            ).fetchone()
        return _media_item(row) if row is not None else None

    def search(self, query: str = "") -> list[MediaItem]:
        with self._connect() as connection:
            if query:
                pattern = f"%{query}%"
                rows = connection.execute(
                    """
                    SELECT * FROM media
                    WHERE title LIKE ? OR path LIKE ?
                    ORDER BY title COLLATE NOCASE, path
                    """,
                    (pattern, pattern),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM media ORDER BY title COLLATE NOCASE, path"
                ).fetchall()
        return [_media_item(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection


def _media_item(row: sqlite3.Row) -> MediaItem:
    return MediaItem(
        media_id=row["media_id"],
        path=row["path"],
        title=row["title"],
        file_format=row["file_format"],
        duration_seconds=row["duration_seconds"],
        sample_rate=row["sample_rate"],
        channel_count=row["channel_count"],
    )


class MediaLibraryService:
    def __init__(self, catalog: MediaCatalog, decoder: AudioDecoder) -> None:
        self.catalog = catalog
        self.decoder = decoder

    def import_file(self, path: str | Path, title: str | None = None) -> MediaItem:
        media_path = Path(path).expanduser().resolve()
        info = self.decoder.inspect(media_path)
        item = MediaItem(
            media_id=str(media_path),
            path=str(media_path),
            title=title.strip() if title and title.strip() else media_path.stem,
            file_format=media_path.suffix.lower().removeprefix("."),
            duration_seconds=info.duration_seconds,
            sample_rate=info.sample_rate,
            channel_count=info.channel_count,
        )
        return self.catalog.add(item)

    def search(self, query: str = "") -> list[MediaItem]:
        return self.catalog.search(query.strip())

    def assign_to_deck(self, deck: DeckState, media_id: str) -> None:
        if self.catalog.get(media_id) is None:
            raise ValueError(f"Media {media_id!r} is not in the library.")
        deck.main_track.media_id = media_id

    def assign_to_sub_track(
        self,
        deck: DeckState,
        slot: int,
        media_id: str,
    ) -> None:
        if not 0 <= slot < len(deck.sub_tracks):
            raise ValueError("Sub-track slot must be between 1 and 4.")
        if self.catalog.get(media_id) is None:
            raise ValueError(f"Media {media_id!r} is not in the library.")
        deck.sub_tracks[slot].media_id = media_id
