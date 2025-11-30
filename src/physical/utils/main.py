def double_fac(n):
    if n == 2:
        return 2

    elif n == 1:
        return 1

    return double_fac(n-2)*n

