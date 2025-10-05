import pytest
from quantreo.datasets import load_generated_ohlcv


@pytest.fixture
def ohlcv_sample():
    return load_generated_ohlcv().loc["2016"]
