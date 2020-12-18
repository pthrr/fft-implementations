#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Collection of Cooley-Tukey fft implementations
"""

import numpy as np
import matplotlib.pyplot as plt

def ditfft2(x, n):
    """
        Radix-2 decimation-in-time fft implementation
        x: list of complex values
        n: length of x
        returns: list of complex values in frequency domain
    """
    xf = np.empty(n, dtype=np.complex64)

    if n == 1:
        xf[0] = x[0]
    else:
        nh = int(n/2)
        x1 = np.empty(nh, dtype=np.complex64)
        x2 = np.empty(nh, dtype=np.complex64)

        # split x into even and uneven elements
        x1 = x[::2]
        x2 = np.roll(x, -1)[::2]

        # descend
        xf1 = ditfft2(x1, nh)
        xf2 = ditfft2(x2, nh)

        # do calculation
        for k in range(0, nh):
            wkn = np.complex64(np.exp(-2.0*np.pi*1j*(k/n))) # twiddle factor
            xf[k] = xf1[k] + xf2[k] * wkn
            xf[k+nh] = xf1[k] - xf2[k] * wkn

    return xf

def bit_reverse(x, n):
    """
        Theory and Application of Digital Signal Processing
        - Rabiner, Gold (1975) p.365
        x: list of complex values
        n: length of x
        returns: list of complex values in bit-reversed order
    """
    j = 0

    for i in range(n-1):
        nh = n/2

        if i < j: # swap elements only once
            temp = x[i]
            x[i] = x[j]
            x[j] = temp

        while nh <= j:
            j = int(j-nh)
            nh = nh/2

        j = int(j+nh)

    return x

def dipfft2(x, n):
    """
        Radix-2 decimation-in-place fft implementation
        x: list of complex values
        n: length of x
        returns: list of complex values in frequency domain
    """
    xr = bit_reverse(x, n)
    bits = int(np.log2(n))

    for i in range(1, bits+1):
        s = int(2**i) # stride
        sh = int(s/2)
        ws = np.complex64(np.exp(-2.0*np.pi*1j*(1.0/s))) # base twiddle factor

        for k in range(0, n, s):
            w = 1 # twiddle factor

            for j in range(sh):
                kj = int(k+j)

                # do calculation
                xf1 = xr[kj]
                xf2 = xr[kj+sh]

                xr[kj] = xf1 + xf2 * w
                xr[kj+sh] = xf1 - xf2 * w

                w = w * ws # calculate next higher twiddle factor

    return xr

def dipfft4(x, n):
    pass

if __name__ == "__main__":
    import numpy.testing as npt

    test_vector = np.array([8.0, 4.0, 8.0, 0.0], dtype=np.complex64)
    X_ref = np.fft.fft(test_vector)

    X_dit2 = ditfft2(test_vector, len(test_vector))
    npt.assert_almost_equal(X_ref, X_dit2, decimal=10)

    X_dip2 = dipfft2(test_vector, len(test_vector))
    npt.assert_almost_equal(X_ref, X_dip2, decimal=10)

    print('All tests passed!')