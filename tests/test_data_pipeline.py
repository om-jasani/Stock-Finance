"""Regression test for the scaler data-leakage bug: MinMaxScaler must be
fit only on the chronological training slice, never on validation/test
data. Constructs a synthetic series where the test-period prices are far
outside the training-period range - if the scaler had leaked and seen the
full series, its learned max would reflect the test-period spike."""
import numpy as np
import pandas as pd

from src.utils.data_pipeline import DataPipeline


def _make_synthetic_ohlcv(n_train=200, n_val=40, n_test=40, train_price=100.0, test_price=1000.0):
    n = n_train + n_val + n_test
    index = pd.date_range('2020-01-01', periods=n, freq='D')

    close = np.concatenate([
        train_price + np.random.RandomState(0).normal(0, 1, n_train),
        train_price + np.random.RandomState(1).normal(0, 1, n_val),
        test_price + np.random.RandomState(2).normal(0, 1, n_test),
    ])
    df = pd.DataFrame({
        'Open': close, 'High': close + 1, 'Low': close - 1,
        'Close': close, 'Volume': np.full(n, 1_000_000.0),
    }, index=index)
    return df


def test_scaler_is_not_fit_on_test_period_values():
    df = _make_synthetic_ohlcv()
    pipeline = DataPipeline()

    pipeline.prepare_training_data(df, sequence_length=10, val_size=0.15, test_size=0.15)

    close_scaler = pipeline.feature_scalers['Close']
    # The scaler's learned max must reflect only the ~100-level training
    # prices, never the ~1000-level test-period spike.
    assert close_scaler.data_max_[0] < 200, (
        f"Scaler data_max_ ({close_scaler.data_max_[0]}) suggests it saw "
        f"test-period values - the scaler leaked future data"
    )


def test_prepare_training_data_split_shapes_are_consistent():
    df = _make_synthetic_ohlcv()
    pipeline = DataPipeline()

    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_training_data(
        df, sequence_length=10, val_size=0.15, test_size=0.15
    )

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    assert len(X_train) > len(X_val) > 0
    assert len(X_test) > 0
    # y is in scaled [0, 1] space (see prepare_training_data), not raw price
    assert y_train.min() >= 0 and y_train.max() <= 1
