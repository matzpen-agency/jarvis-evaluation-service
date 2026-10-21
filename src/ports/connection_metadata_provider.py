"""
connection_metadata_provider.py — Abstract interface for data-connection metadata.

Implementations: JeenConnectionMetadataProvider, BackendTableResolver (no connection)
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ConnectionMetadataProvider(ABC):
    """Resolves the SQL dialect and table/column schema of a data connection."""

    @abstractmethod
    async def get_dialect(self, connection_id: int | None) -> str | None:
        """
        Return the sqlglot dialect name of the connection's database
        (e.g. "postgres", "trino"), or None if unknown / not SQL.
        """
        ...

    async def get_connection_name(self, connection_id: int | None) -> str | None:
        """Display name of the connection, or None if unknown."""
        return None

    @abstractmethod
    async def get_table_schema_map(
        self, connection_id: int | None = None
    ) -> dict[str, dict[str, str]]:
        """
        Return {table_key: {column_name: db_type}} with lower-case keys.

        Each table is reachable by its bare name and by "schema.table"
        (and "catalog.schema.table" when the connection has catalogs).
        Returns an empty dict when metadata is unavailable.
        """
        ...
