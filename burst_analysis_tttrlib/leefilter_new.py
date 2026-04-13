import numpy as np
from scipy.ndimage import uniform_filter, variance


def leeFilter1D_Add(I, window_size):
    I = np.array(I, dtype=float)
    mean_I = uniform_filter(I, window_size)
    sqr_mean_I = uniform_filter(I**2, window_size)
    var_I = sqr_mean_I - mean_I**2

    overall_variance = variance(I)

    weight_I = var_I / (var_I + overall_variance)
    output_I = mean_I + weight_I * (I - mean_I)
    return output_I


def leeFilter1D_Multi(I, window_size):
    I = np.array(I, dtype=float)
    mean_I = uniform_filter(I, window_size)
    sqr_mean_I = uniform_filter(I**2, window_size)
    var_I = sqr_mean_I - mean_I**2

    weight_I = (
        np.mean(I) * var_I
        / (mean_I**2 / window_size + (var_I * np.mean(I)**2))
    )
    output_I = mean_I + weight_I * (I - mean_I * np.mean(I))
    return output_I


def leeFilter1D_matlab(I, window_size):
    I = np.array(I, dtype=float)
    OIm = I.copy()
    means = uniform_filter(I, window_size)
    sigmas = np.sqrt((I - means) ** 2 / window_size ** 2)
    sigmas = uniform_filter(sigmas, window_size)

    ENLs = (means / sigmas) ** 2
    sx2s = ((ENLs * sigmas**2) - means**2) / (ENLs + 1)
    fbar = means + (sx2s * (I - means) / (sx2s + (means**2 / ENLs)))
    OIm[means != 0] = fbar[means != 0]
    return OIm