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
    with importlib.resources.open_text('quantreo.datasets', 'generated_ohlcv.csv') as f:
        df = pd.read_csv(f, index_col='time', parse_dates=['time'])

    df = df.astype({
        'open': 'float64',
        'high': 'float64',
        'low': 'float64',
        'close': 'float64',
        'volume': 'float64'
    })

    return df