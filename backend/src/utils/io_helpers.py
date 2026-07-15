"""
src/utils/io_helpers.py
========================
Save and load helpers for models, DataFrames, and figures.

All functions are thin, opinionated wrappers that:
- Create parent directories automatically.
- Log every save/load operation.
- Raise informative errors on failure.
- Are station-aware (namespaced by station_id).
"""

from __future__ import annotations

import joblib
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def save_dataframe(
    df: pd.DataFrame,
    path: str | Path,
    index: bool = False,
) -> Path:
    """
    Save a DataFrame to CSV, creating parent directories as needed.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    path : str | Path
        Destination file path (must end in ``.csv``).
    index : bool
        Whether to write the row index. Default False.

    Returns
    -------
    Path
        Resolved path of the saved file.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=index)
    logger.info(f"DataFrame saved → {dest} | shape={df.shape}")
    return dest


def load_dataframe(
    path: str | Path,
    parse_dates: Optional[list] = None,
    index_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a DataFrame from CSV.

    Parameters
    ----------
    path : str | Path
        Source CSV file path.
    parse_dates : list, optional
        Column names to parse as datetime.
    index_col : str, optional
        Column to use as the DataFrame index.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"DataFrame file not found: {src}")
    df = pd.read_csv(src, parse_dates=parse_dates, index_col=index_col)
    logger.info(f"DataFrame loaded ← {src} | shape={df.shape}")
    return df


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def save_model(model: Any, path: str | Path) -> Path:
    """
    Serialise a scikit-learn / XGBoost model using joblib.

    Parameters
    ----------
    model : Any
        A fitted model object with a ``predict`` method.
    path : str | Path
        Destination path (e.g., ``outputs/models/xgb_v1.joblib``).

    Returns
    -------
    Path
        Resolved path of the saved model.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest)
    logger.info(f"Model saved → {dest}")
    return dest


def load_model(path: str | Path) -> Any:
    """
    Load a joblib-serialised model from disk.

    Parameters
    ----------
    path : str | Path
        Source model file path.

    Returns
    -------
    Any
        The deserialised model object.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Model file not found: {src}")
    model = joblib.load(src)
    logger.info(f"Model loaded ← {src}")
    return model


# ---------------------------------------------------------------------------
# JSON helpers (metrics, metadata)
# ---------------------------------------------------------------------------

def save_json(data: dict, path: str | Path) -> Path:
    """
    Save a dictionary as a pretty-printed JSON file.

    Parameters
    ----------
    data : dict
        JSON-serialisable dictionary.
    path : str | Path
        Destination file path.

    Returns
    -------
    Path
        Resolved path of the saved file.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info(f"JSON saved → {dest}")
    return dest


def load_json(path: str | Path) -> dict:
    """
    Load a JSON file into a dictionary.

    Parameters
    ----------
    path : str | Path
        Source JSON file path.

    Returns
    -------
    dict

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"JSON file not found: {src}")
    with src.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info(f"JSON loaded ← {src}")
    return data


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def save_figure(fig: Any, path: str | Path, dpi: int = 150) -> Path:
    """
    Save a Matplotlib or Plotly figure to disk.

    Parameters
    ----------
    fig : matplotlib.figure.Figure | plotly.graph_objects.Figure
        The figure object.
    path : str | Path
        Destination path (extension determines format: .png, .pdf, .html).
    dpi : int
        Resolution for raster formats (Matplotlib only). Default 150.

    Returns
    -------
    Path
        Resolved path of the saved figure.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    ext = dest.suffix.lower()
    try:
        # Matplotlib
        if hasattr(fig, "savefig"):
            fig.savefig(dest, dpi=dpi, bbox_inches="tight")
        # Plotly
        elif hasattr(fig, "write_html") and ext == ".html":
            fig.write_html(str(dest))
        elif hasattr(fig, "write_image"):
            fig.write_image(str(dest))
        else:
            raise TypeError(f"Unsupported figure type: {type(fig)}")
        logger.info(f"Figure saved → {dest}")
    except Exception as exc:
        logger.error(f"Failed to save figure to {dest}: {exc}")
        raise

    return dest
