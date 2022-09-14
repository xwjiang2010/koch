import numpy as np
import pytest
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (  # mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
)
from sklearn.utils._testing import assert_almost_equal

from prediction.src.metrics.regression_metrics import calc_bias, calc_mape

RANDOM_NUM_GEN = np.random.RandomState(42)


def test_calc_mape():
    y_true = RANDOM_NUM_GEN.exponential(size=100)
    y_pred = 1.2 * y_true
    assert calc_mape(y_true, y_pred) == pytest.approx(0.2)


def test_regression_metrics(n_samples=50):
    y_true = np.arange(n_samples)
    y_pred = y_true + 1

    assert_almost_equal(mean_squared_error(y_true, y_pred), 1.0)
    assert_almost_equal(
        mean_squared_log_error(y_true, y_pred),
        mean_squared_error(np.log(1 + y_true), np.log(1 + y_pred)),
    )
    assert_almost_equal(mean_absolute_error(y_true, y_pred), 1.0)
    assert_almost_equal(median_absolute_error(y_true, y_pred), 1.0)
    # mape = mean_absolute_percentage_error(y_true, y_pred)
    # assert np.isfinite(mape)
    # assert mape > 1e6


def test_regression_metrics_at_limits():
    assert_almost_equal(mean_squared_error([0.0], [0.0]), 0.00, 2)
    assert_almost_equal(mean_squared_error([0.0], [0.0], squared=False), 0.00, 2)
    assert_almost_equal(mean_squared_log_error([0.0], [0.0]), 0.00, 2)
    assert_almost_equal(mean_absolute_error([0.0], [0.0]), 0.00, 2)
    # assert_almost_equal(mean_absolute_percentage_error([0.0], [0.0]), 0.00, 2)
    assert_almost_equal(median_absolute_error([0.0], [0.0]), 0.00, 2)
    # assert_almost_equal(max_error([0.0], [0.0]), 0.00, 2)
    # assert_almost_equal(explained_variance_score([0.0], [0.0]), 1.00, 2)
    assert_almost_equal(r2_score([0.0, 1], [0.0, 1]), 1.00, 2)
    err_msg = (
        "Mean Squared Logarithmic Error cannot be used when targets "
        "contain negative values."
    )
    with pytest.raises(ValueError, match=err_msg):
        mean_squared_log_error([-1.0], [-1.0])
    err_msg = (
        "Mean Squared Logarithmic Error cannot be used when targets "
        "contain negative values."
    )
    with pytest.raises(ValueError, match=err_msg):
        mean_squared_log_error([1.0, 2.0, 3.0], [1.0, -2.0, 3.0])
    err_msg = (
        "Mean Squared Logarithmic Error cannot be used when targets "
        "contain negative values."
    )
    with pytest.raises(ValueError, match=err_msg):
        mean_squared_log_error([1.0, -2.0, 3.0], [1.0, 2.0, 3.0])


@pytest.mark.parametrize("metric", [r2_score])
def test_regression_single_sample(metric):
    y_true = [0]
    y_pred = [1]
    warning_msg = "not well-defined with less than two samples."

    # Trigger the warning
    with pytest.warns(UndefinedMetricWarning, match=warning_msg):
        score = metric(y_true, y_pred)
        assert np.isnan(score)


def test_calc_bias():
    y_true = RANDOM_NUM_GEN.exponential(size=100)
    y_pred = np.multiply(1.2, y_true)

    assert calc_bias(y_true, y_pred) == pytest.approx(20)


def test_calc_bias_weighted():
    yTrue = RANDOM_NUM_GEN.exponential(size=100)
    yPred = np.multiply(1.2, yTrue)

    first_sample_all_weight = np.zeros(yTrue.shape)
    first_sample_all_weight[0] = 1

    bias_expected = (yPred[0] - yTrue[0]) / yTrue[0]
    bias = calc_bias(yTrue, yPred, weights=first_sample_all_weight)

    assert bias == bias_expected
