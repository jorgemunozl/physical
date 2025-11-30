import matplotlib.pyplot as plt
import numpy as np
import math


def legendre_polinomial(n, x):
    """
    Genre from recursion, n>=1
    """
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        p1 = (2*(n)-1)*x*legendre_polinomial(n-1, x)
        p2 = (n-1)*legendre_polinomial(n-2, x)
        return (1/n)*(p1 - p2)


def legendre_2(x):
    return 1/2*(3*np.pow(x, 2)-1)


def legendre_5(x):
    return 1/8*(63*np.pow(x, 5)-70*np.pow(x, 3)+15*x)


def legendre_derivative(n, x):
    """
    Returns the derivative of the n legendre function
    """
    diff = legendre_polinomial(n-1, x) - x*legendre_polinomial(n, x)
    return n/(1-np.pow(x, 2))*diff


def P_lm(l, m, x):
    """
    Associated Legendre P_l^m(x) for 0 <= m <= l
    using standard recurrences.
    """
    if m < 0 or m > l:
        raise ValueError("Need 0 <= m <= l")

    # P_m^m(x)
    P_mm = (-1)**m * double_factorial(2*m - 1) * (1 - x**2)**(m/2)
    if l == m:
        return P_mm

    # P_{m+1}^m(x)
    P_m1m = x * (2*m + 1) * P_mm
    if l == m + 1:
        return P_m1m

    # Upward recurrence in l
    P_lm2 = P_mm   # P_m^m
    P_lm1 = P_m1m  # P_{m+1}^m

    for ell in range(m + 2, l + 1):
        P_l = ((2*ell - 1) * x * P_lm1 - (ell + m - 1) * P_lm2) / (ell - m)
        P_lm2, P_lm1 = P_lm1, P_l

    return P_l


def double_factorial(n):
    if n <= 0:
        return 1
    result = 1
    for k in range(n, 0, -2):
        result *= k
    return result


def d_m_legendre(l, m, x):
    """
    m-th derivative d^m/dx^m P_l(x) using associated Legendre.
    """
    if m < 0 or m > l:
        raise ValueError("Need 0 <= m <= l")
    Plm = P_lm(l, m, x)
    return (-1)**m * (1 - x**2)**(-m/2) * Plm


def f_legendre_1_1(x):
    return -1*np.sqrt(1-np.pow(x, 2))


def f_legendre_1_2(x):
    return -3*x*np.sqrt(1-np.pow(x, 2))


if __name__ == "__main__":
    x = np.linspace(0, 10, 1000)
    z = f_legendre_1_2(x) + 1
    y = P_lm(2, 1, x)
    plt.plot(x, y)
    plt.plot(x, z)
    plt.show()
