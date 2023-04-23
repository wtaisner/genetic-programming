def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1
