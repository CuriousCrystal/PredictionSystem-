"""
Data loading and initial exploration utilities.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_DATASET

logger = logging.getLogger(__name__)


def load_data(filepath: Path | str | None = None) -> pd.DataFrame:
    """
    Load the student performance dataset from CSV.

    Args:
        filepath: Path to the CSV file. Defaults to ``config.DEFAULT_DATASET``.

    Returns:
        DataFrame with the raw student data.

    Raises:
        FileNotFoundError: If the CSV file doesn't exist at the given path.
    """
    filepath = Path(filepath) if filepath is not None else DEFAULT_DATASET

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at '{filepath}'")

    data = pd.read_csv(filepath)
    logger.info("Loaded dataset: %d rows × %d columns from %s",
                data.shape[0], data.shape[1], filepath.name)
    return data


def show_data_summary(data: pd.DataFrame) -> None:
    """
    Print a comprehensive summary of the dataset.

    Args:
        data: Raw DataFrame.
    """
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"\nShape: {data.shape[0]} rows × {data.shape[1]} columns")
    print(f"\nColumns: {list(data.columns)}")
    print(f"\nData Types:\n{data.dtypes}")
    print(f"\nMissing Values:\n{data.isnull().sum()}")
    print(f"\nClass Distribution:\n{data['Class'].value_counts()}")
    print(f"\nNumerical Feature Statistics:\n{data.describe()}")
    print("=" * 60)
