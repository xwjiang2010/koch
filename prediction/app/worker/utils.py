def cap(loss_function, value):
    if loss_function == "MAPE":
        return max(0, value)
    else:
        return value
