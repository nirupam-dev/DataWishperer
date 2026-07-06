"""
File repository — CRUD operations for uploaded CSV file metadata.

Stores file metadata in the database while the actual file lives on disk
(managed by ``FileManager``).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session as SASession

from backend.core.exceptions import FileNotFoundError_
from backend.core.logging_config import get_logger
from backend.models.database import FileModel, get_db_session
from backend.models.schemas import ColumnInfo, FileMetadata

logger = get_logger(__name__)


class FileRepository:
    """
    Data access object for uploaded file metadata.

    Args:
        db: An active SQLAlchemy session.
    """

    def __init__(self, db: Optional[SASession] = None) -> None:
        self._db = db or get_db_session()

    def save(self, metadata: FileMetadata) -> FileModel:
        """
        Persist file metadata to the database.

        Serializes column information as JSON for storage.

        Args:
            metadata: The Pydantic ``FileMetadata`` to persist.

        Returns:
            The created ORM ``FileModel``.
        """
        columns_json = json.dumps(
            [col.model_dump() for col in metadata.columns],
            default=str,
        )

        db_file = FileModel(
            id=metadata.file_id,
            original_name=metadata.original_name,
            stored_path=metadata.stored_path,
            row_count=metadata.row_count,
            col_count=metadata.col_count,
            file_size_bytes=metadata.file_size_bytes,
            column_metadata=columns_json,
            uploaded_at=metadata.uploaded_at,
        )
        self._db.add(db_file)
        self._db.commit()
        self._db.refresh(db_file)
        logger.info(
            "Saved file metadata: id=%s, name='%s', rows=%d, cols=%d",
            metadata.file_id, metadata.original_name,
            metadata.row_count, metadata.col_count,
        )
        return db_file

    def get_by_id(self, file_id: str) -> Optional[FileModel]:
        """
        Retrieve file metadata by ID.

        Args:
            file_id: The file UUID.

        Returns:
            The ``FileModel`` or ``None``.
        """
        return self._db.query(FileModel).filter(FileModel.id == file_id).first()

    def get_by_id_or_raise(self, file_id: str) -> FileModel:
        """
        Retrieve file metadata by ID or raise an error.

        Args:
            file_id: The file UUID.

        Returns:
            The ``FileModel``.

        Raises:
            FileNotFoundError_: If no file exists with the given ID.
        """
        file_model = self.get_by_id(file_id)
        if file_model is None:
            raise FileNotFoundError_(file_id)
        return file_model

    def get_metadata(self, file_id: str) -> FileMetadata:
        """
        Retrieve and deserialize full file metadata including column info.

        Args:
            file_id: The file UUID.

        Returns:
            A fully populated ``FileMetadata`` schema.

        Raises:
            FileNotFoundError_: If no file exists with the given ID.
        """
        db_file = self.get_by_id_or_raise(file_id)

        columns: List[ColumnInfo] = []
        if db_file.column_metadata:
            raw_columns = json.loads(db_file.column_metadata)
            columns = [ColumnInfo(**col) for col in raw_columns]

        return FileMetadata(
            file_id=db_file.id,
            original_name=db_file.original_name,
            stored_path=db_file.stored_path,
            row_count=db_file.row_count,
            col_count=db_file.col_count,
            file_size_bytes=db_file.file_size_bytes,
            memory_usage_mb=round(db_file.file_size_bytes / (1024 * 1024), 2),
            columns=columns,
            uploaded_at=db_file.uploaded_at,
        )

    def list_all(self, limit: int = 50) -> List[FileModel]:
        """
        List all uploaded files, most recent first.

        Args:
            limit: Maximum number of files to return.

        Returns:
            List of ``FileModel`` instances.
        """
        return (
            self._db.query(FileModel)
            .order_by(FileModel.uploaded_at.desc())
            .limit(limit)
            .all()
        )

    def count_all(self) -> int:
        """Return the total number of uploaded files."""
        return self._db.query(FileModel).count()

    def delete(self, file_id: str) -> None:
        """
        Delete file metadata from the database.

        Does NOT delete the file from disk — that is ``FileManager``'s job.

        Args:
            file_id: The file UUID.
        """
        db_file = self.get_by_id_or_raise(file_id)
        self._db.delete(db_file)
        self._db.commit()
        logger.info("Deleted file metadata: %s", file_id)

    def close(self) -> None:
        """Close the underlying database session."""
        self._db.close()
