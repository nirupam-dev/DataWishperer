"""
File service — Orchestrates CSV upload, validation, and metadata extraction.

This is the business logic layer for file operations. It coordinates
between security validation, disk storage, CSV analysis, and database
persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core.config import StorageSettings, get_settings
from backend.core.logging_config import get_logger
from backend.core.security import validate_upload
from backend.models.schemas import ColumnInfo, FileMetadata, FileUploadResponse
from backend.storage.file_manager import FileManager
from backend.storage.repositories.file_repo import FileRepository
from backend.utils.csv_analyzer import CSVAnalyzer

logger = get_logger(__name__)


class FileService:
    """
    Handles the complete file upload and management lifecycle.

    Coordinates:
        - Security validation (extension, size, content)
        - Disk storage with unique filenames
        - CSV profiling and metadata extraction
        - Database persistence of metadata
        - Preview row generation

    Args:
        file_manager: Disk file storage manager.
        file_repo: Database repository for file metadata.
        csv_analyzer: CSV profiling utility.
        storage_settings: Storage configuration.
    """

    def __init__(
        self,
        file_manager: Optional[FileManager] = None,
        file_repo: Optional[FileRepository] = None,
        csv_analyzer: Optional[CSVAnalyzer] = None,
        storage_settings: Optional[StorageSettings] = None,
    ) -> None:
        self._file_manager = file_manager or FileManager()
        self._file_repo = file_repo or FileRepository()
        self._csv_analyzer = csv_analyzer or CSVAnalyzer()
        self._storage_settings = storage_settings or get_settings().storage

    def upload_file(
        self,
        filename: str,
        content: bytes,
    ) -> FileUploadResponse:
        """
        Process a complete file upload.

        Pipeline:
            1. Validate the upload (extension, size, content, columns)
            2. Save to disk with a unique filename
            3. Analyze the CSV and extract metadata
            4. Persist metadata to the database
            5. Generate preview rows

        Args:
            filename: Original filename from the upload.
            content: Raw file bytes.

        Returns:
            A ``FileUploadResponse`` with metadata and preview.

        Raises:
            InvalidFileError: If validation fails.
            FileTooLargeError: If the file is too large.
            TooManyColumnsError: If too many columns.
        """
        # 1. Validate
        validate_upload(filename, content, self._storage_settings)

        # 2. Save to disk
        file_id, stored_path = self._file_manager.save_file(filename, content)

        # 3. Analyze
        try:
            metadata = self._csv_analyzer.analyze(
                csv_path=stored_path,
                file_id=file_id,
                original_name=filename,
            )
        except Exception as e:
            # Clean up on analysis failure
            self._file_manager.delete_file(str(stored_path))
            raise

        # 4. Persist metadata
        self._file_repo.save(metadata)

        # 5. Generate preview
        preview_rows = self._csv_analyzer.get_preview_rows(stored_path, max_rows=10)

        logger.info(
            "File uploaded successfully: '%s' (%d rows × %d cols)",
            filename, metadata.row_count, metadata.col_count,
        )

        return FileUploadResponse(
            file_id=file_id,
            filename=filename,
            row_count=metadata.row_count,
            col_count=metadata.col_count,
            file_size_mb=round(len(content) / (1024 * 1024), 2),
            columns=metadata.columns,
            preview_rows=preview_rows,
        )

    def get_file_metadata(self, file_id: str) -> FileMetadata:
        """
        Retrieve full file metadata by ID.

        Args:
            file_id: The file UUID.

        Returns:
            Complete ``FileMetadata`` including column profiles.

        Raises:
            FileNotFoundError_: If the file doesn't exist.
        """
        return self._file_repo.get_metadata(file_id)

    def get_file_path(self, file_id: str) -> str:
        """
        Get the disk path for a file by ID.

        Args:
            file_id: The file UUID.

        Returns:
            Absolute path to the CSV file on disk.
        """
        metadata = self._file_repo.get_metadata(file_id)
        return metadata.stored_path

    def get_preview_rows(self, file_id: str, max_rows: int = 100) -> List[Dict[str, Any]]:
        """
        Get preview rows for display in the UI.

        Args:
            file_id: The file UUID.
            max_rows: Maximum rows to return.

        Returns:
            List of row dictionaries.
        """
        metadata = self._file_repo.get_metadata(file_id)
        return self._csv_analyzer.get_preview_rows(metadata.stored_path, max_rows)

    def get_data_quality_report(self, file_id: str) -> Dict[str, Any]:
        """
        Generate a data quality report for a file.

        Args:
            file_id: The file UUID.

        Returns:
            Quality report dictionary.
        """
        metadata = self._file_repo.get_metadata(file_id)
        return self._csv_analyzer.get_data_quality_report(metadata.stored_path)

    def list_files(self) -> List[Dict[str, Any]]:
        """
        List all uploaded files with basic info.

        Returns:
            List of file info dictionaries.
        """
        files = self._file_repo.list_all()
        return [
            {
                "id": f.id,
                "name": f.original_name,
                "rows": f.row_count,
                "cols": f.col_count,
                "size_bytes": f.file_size_bytes,
                "uploaded_at": f.uploaded_at,
            }
            for f in files
        ]

    def delete_file(self, file_id: str) -> None:
        """
        Delete a file from both disk and database.

        Args:
            file_id: The file UUID.
        """
        db_file = self._file_repo.get_by_id_or_raise(file_id)
        self._file_manager.delete_file(db_file.stored_path)
        self._file_repo.delete(file_id)
        logger.info("Deleted file: %s ('%s')", file_id, db_file.original_name)

    def get_disk_usage_mb(self) -> float:
        """Return total disk usage of uploaded files in MB."""
        return self._file_manager.get_disk_usage_mb()

    def get_file_count(self) -> int:
        """Return total number of uploaded files."""
        return self._file_repo.count_all()
