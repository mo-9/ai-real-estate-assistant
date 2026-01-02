from pathlib import Path
from typing import Optional, Union
from yarl import URL

from .base import BaseDataProvider
from .csv_provider import CSVDataProvider
from .json_provider import JSONDataProvider


class DataProviderFactory:
    """Factory for creating data providers based on source type."""

    @staticmethod
    def create_provider(source: Union[str, Path, URL]) -> BaseDataProvider:
        """
        Create the appropriate data provider for the given source.

        Args:
            source: Path to file or URL

        Returns:
            An instance of a BaseDataProvider subclass
        """
        source_str = str(source)
        
        # Determine extension
        if source_str.lower().endswith('.json'):
            return JSONDataProvider(source)
        elif source_str.lower().endswith(('.csv', '.xlsx', '.xls')):
            return CSVDataProvider(source)
        else:
            # Default to CSV provider for unknown extensions (legacy behavior compatibility)
            # or could check content type if it's a URL
            return CSVDataProvider(source)
