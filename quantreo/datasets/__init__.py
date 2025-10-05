import pandas as pd
import importlib.resources


def load_generated_ohlcv():
    """
    Charge le dataset OHLCV généré.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe contenant les colonnes ['open', 'high', 'low', 'close', 'volume']
        avec un index DatetimeIndex nommé 'time'.
    """
    path = importlib.resources.files("quantreo.datasets") / "generated_ohlcv.csv"
    return pd.read_csv(path, parse_dates=["time"], index_col="time")


def load_generated_ohlcv_with_time() -> pd.DataFrame:
    """
    Load the generated OHLCV dataset including high_time and low_time columns.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with the following columns:
        ['open', 'high', 'low', 'close', 'volume', 'high_time', 'low_time']
        and a DatetimeIndex named 'time'.
    """
    path = importlib.resources.files("quantreo.datasets") / "generated_ohlcv_with_time.csv"
    df = pd.read_csv(path, index_col="time", parse_dates=["time", "high_time", "low_time"])

    df = df.astype(
        {
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        }
    )

    return df


def load_generated_ticks() -> pd.DataFrame:
    """
    Load the generated tick-level dataset with price and volume.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with the following columns:
        ['price', 'volume']
        and a DatetimeIndex named 'datetime'.
    """
    path = importlib.resources.files("quantreo.datasets") / "generated_ticks.csv"
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    df = df.astype({"price": "float64", "volume": "int64"})
    return df
