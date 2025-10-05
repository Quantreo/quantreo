from quantreo.features_engineering.candle.info import candle_information


def test_derivatives_with_dataset(ohlcv_sample):
    df = ohlcv_sample.copy()
    df["candle_way"], df["filling"], df["amplitude"] = candle_information(
        df=df, open_col="open", high_col="high", low_col="low", close_col="close"
    )
    assert "candle_way" in df.columns
    assert "filling" in df.columns
    assert "amplitude" in df.columns
    assert df.notna().all().all()
