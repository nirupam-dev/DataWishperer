"""
File manager — Disk-level operations for uploaded CSV files.

Handles file storage, retrieval, and deletion on the local filesystem.
Works alongside ``FileRepository`` which manages metadata in the database.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional
from uuid import uuid4

from backend.core.config import StorageSettings, get_settings
from backend.core.logging_config import get_logger
from backend.core.security import sanitize_filename

logger = get_logger(__name__)


class FileManager:
    """
    Manages CSV files on the local filesystem.

    Responsible for:
        - Storing uploaded files with unique names
        - Retrieving file paths
        - Deleting files
        - Calculating disk usage

    Args:
        settings: Storage settings. Defaults to global settings.
    """

    def __init__(self, settings: Optional[StorageSettings] = None) -> None:
        self._settings = settings or get_settings().storage
        self._upload_dir = self._settings.upload_path
        self._upload_dir.mkdir(parents=True, exist_ok=True)

    def save_file(self, filename: str, content: bytes) -> tuple[str, Path]:
        """
        Save uploaded file content to disk with a unique filename.

        The original filename is sanitized and prepended with a UUID
        fragment to prevent collisions.

        Args:
            filename: Original filename from the upload.
            content: Raw file bytes.

        Returns:
            A tuple of ``(file_id, stored_path)`` where ``file_id`` is a
            UUID string and ``stored_path`` is the absolute path on disk.
        """
        file_id = str(uuid4())
        safe_name = sanitize_filename(filename)
        stored_name = f"{file_id[:8]}_{safe_name}"
        stored_path = self._upload_dir / stored_name

        stored_path.write_bytes(content)
        logger.info(
            "Saved file to disk: '%s' -> '%s' (%d bytes)",
            filename, stored_path, len(content),
        )
        return file_id, stored_path

    def get_file_path(self, stored_path: str) -> Path:
        """
        Resolve and validate a stored file path.

        Args:
            stored_path: The stored path string from the database.

        Returns:
            The resolved ``Path`` object.

        Raises:
            FileNotFoundError: If the file does not exist on disk.
        """
        path = Path(stored_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found on disk: {stored_path}")
        return path

    def delete_file(self, stored_path: str) -> bool:
        """
        Delete a file from disk.

        Args:
            stored_path: The stored path string.

        Returns:
            ``True`` if the file was deleted, ``False`` if it didn't exist.
        """
        path = Path(stored_path)
        if path.exists():
            path.unlink()
            logger.info("Deleted file from disk: '%s'", stored_path)
            return True
        logger.warning("File not found for deletion: '%s'", stored_path)
        return False

    def get_disk_usage_bytes(self) -> int:
        """
        Calculate total disk usage of all uploaded files.

        Returns:
            Total size in bytes.
        """
        total = sum(f.stat().st_size for f in self._upload_dir.iterdir() if f.is_file())
        return total

    def get_disk_usage_mb(self) -> float:
        """
        Calculate total disk usage in megabytes.

        Returns:
            Total size in MB, rounded to 2 decimal places.
        """
        return round(self.get_disk_usage_bytes() / (1024 * 1024), 2)

    def list_files(self) -> list[Path]:
        """
        List all files in the upload directory.

        Returns:
            List of file ``Path`` objects.
        """
        return sorted(
            [f for f in self._upload_dir.iterdir() if f.is_file()],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

    def cleanup_orphaned_files(self, known_paths: set[str]) -> int:
        """
        Delete files on disk that have no corresponding database record.

        Args:
            known_paths: Set of stored paths that exist in the database.

        Returns:
            Number of orphaned files deleted.
        """
        deleted = 0
        for file_path in self._upload_dir.iterdir():
            if file_path.is_file() and str(file_path) not in known_paths:
                file_path.unlink()
                deleted += 1
                logger.info("Cleaned up orphaned file: '%s'", file_path)
        return deleted
