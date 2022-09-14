import numpy as np
from sklearn.metrics import r2_score


def rmse(y_true, y_pred):
    """Root mean squared error for regression loss
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    """
    from math import sqrt

    from sklearn.metrics import mean_squared_error

    actual = np.array(y_true)
    predicted = np.array(y_pred)
    return sqrt(mean_squared_error(actual, predicted))


def adjusted_r2_score(y_true, y_pred, p=1):
    """Adjusted r-squared error for regression loss
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    p: number of explanatory variables
    """

    actual = np.array(y_true)
    predicted = np.array(y_pred)
    r2 = r2_score(actual, predicted)
    n = len(actual)
    num = n - 1
    denom = n - p - 1
    if denom == 0:
        raise ValueError("Can't calculated adjusted r-squared due to n-p-1==0")

    adj_r2 = 1 - (1 - r2) * (num / denom)
    return adj_r2


##-------------------------------------MAPEs----------------------------
def calc_mape(actual, predicted):
    """
    Calculation of mean absolute error with
    actual value as denominator
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    _validate(actual, predicted)

    # The logic is slightly complicated here to handle the following cases:
    # if actual == 0:
    #    if predicted == 0:
    #       mape = 0
    #    else:
    #       mape = 1
    # else:
    #   mape == standard MAPE capped at 1 for each error.

    # Calculate percent error where actual != 0, else 1
    mape_array = np.abs(
        np.divide(
            actual - predicted, actual, out=np.ones_like(actual), where=actual != 0
        )
    )

    # Handle special case where actual == predicted
    # (this is really covering where they both equal 0)
    mape_array = np.where(actual == predicted, 0, mape_array)

    # Cap individual percent errors at 1
    # mape_array = np.minimum(1, mape_array)
    return np.nanmean(mape_array)


def calc_mape_remove_neg(actual, predicted):
    """
    Calculation of mean absolute error with
    actual value as denominator
    Before calcuating MAPE remove replace negative predictions with 0
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    _validate(actual, predicted)

    # clipping negative predictions to zero
    predicted = np.clip(predicted, 0, None)
    mape_array = np.minimum(1, np.abs((predicted - actual) / actual))
    # TODO: penalize the overprediction case when actual is zero and prediction is non-zero
    return np.nanmean(mape_array)


def calc_pred_mape_remove_neg_zero(actual, predicted, zero_pred_error=1):
    """
    Calculation of mean absolute error with
    predicted value as denominator
    Before calcuating MAPE:
    1. Replace negative predictions with 0
    2. Limit predicted 0s by either provided value or default value of 1
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    _validate(actual, predicted)

    error = 0.0
    for i in range(len(actual)):
        a = actual[i]
        p = predicted[i]
        if p < 0:
            p = 0
        if p == 0:
            if a > 0:
                error += zero_pred_error
        error += min(1, np.abs((p - a) / p))
    error /= len(actual)
    if type(error) == np.ndarray:
        error = error[0]
    return error


def calc_pred_mape_remove_neg(actual, predicted):
    """
    Calculation of mean absolute error with
    predicted value as denominator
    Before calcuating MAPE:
    1. replace negative predictions with 0
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    _validate(actual, predicted)
    # clipping negative predictions to zero
    predicted = np.clip(predicted, 0, None)
    mape_array = np.minimum(1, np.abs((predicted - actual) / predicted))
    return np.nanmean(mape_array)


def calc_pred_mape_remove_neg_add_ep(actual, predicted, epsilon=10e-10):
    """
    Calculation of mean absolute error with
    predicted value as denominator
    Before calcuating MAPE:
    1. replace negative predictions with 0
    Add epsilon to the predicted in the denominator
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    _validate(predicted, actual)

    # clipping negative predictions to zero
    predicted = np.clip(predicted, 0, None)
    mape_array = np.minimum(1, np.abs((predicted - actual) / (predicted + epsilon)))
    return np.nanmean(mape_array)


def calc_bias(actual, predicted, weights=1):

    actual = np.array(actual)
    predicted = np.array(predicted)

    _validate(actual, predicted)

    bias_array = np.multiply((predicted - actual) / actual, weights)

    return np.sum(bias_array) / np.sum(weights)


def _validate(predicted, actual):
    """
    Predicted and actual need to be
    equal length to do our calculations.

    Actuals must have some length in order
    to do our calculations.
    """

    if len(actual) != len(predicted):
        # TODO: Consider a fatal exception here.
        raise ValueError(
            "Arrays have unequal lengths of {} and {}".format(
                len(actual), len(predicted)
            )
        )
    elif len(actual) == 0:
        raise ValueError("Actuals cannot have zero length")
