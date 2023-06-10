import math


def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def maximal(left, right):
    return max(left, right)


def minimal(left, right):
    return min(left, right)


def int_gcd(left, right):
    return int(math.gcd(int(left), int(right)))
