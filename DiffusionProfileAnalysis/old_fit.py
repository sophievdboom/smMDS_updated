#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:50:34 2019

@author: quentinpeter
"""


def fit_2_fix_1(profiles, Basis, phi, phi_fix, vary_offset=False):
    """Find the best monodisperse radius

    Parameters
    ----------
    M: 2d array
        The basis matrix. Mij = sum(basisi*basisj)
    b: 1d array
        bi = sum(profile*basisi)
    psquare: float
        psquare = sum(profiles*profile)
    phi:
        MUST BE SORTED
    Rs: 1d float
        The test radii [m]

    Returns
    -------
    radii: float
        The best radius fit
    """
    # Fit monodisperse to get mid point
    mono_fit = fit_monodisperse(profiles, Basis, phi, vary_offset)
    idx_min_mono = mono_fit.arg_x

    # Check shape Basis
    if len(Basis.shape) == 2:
        # add axis for pos
        Basis = Basis[:, np.newaxis]
        profiles = profiles[np.newaxis, :]
    # basis has phi. pos, pixel

    # Compute the matrices needed for res_interp_N
    # equivalent to BB = np.einsum('jik, lik -> ijl', Basis, Basis)?
    BB = np.empty((np.shape(Basis)[1], np.shape(Basis)[0], np.shape(Basis)[0]))
    for i in range(np.shape(Basis)[-2]):
        Bi = Basis[..., i, :]
        BB[i] = (np.tensordot(Bi, Bi, (-1, -1)))
    Bp = np.einsum('jik, ik -> ij', Basis, profiles)
    B = np.einsum('jik -> ij', Basis)
    pp = np.einsum('ik, ik -> i', profiles, profiles)
    p = np.einsum('ik -> i', profiles)
    Nb = np.shape(Basis)[0]
    Npix = np.shape(Basis)[-1]

    # get indix for fixed phi
    idx_fix_minus = np.sum(phi - phi_fix < 0) - 1
    if idx_fix_minus == -1 or idx_fix_minus == len(phi) - 1:
        raise RuntimeError('Fixed phi out of range')
    idx_fix = ((np.log(phi_fix) - np.log(phi[idx_fix_minus]))
                / (np.log(phi[idx_fix_minus + 1]) - np.log(phi[idx_fix_minus]))
                + idx_fix_minus)
    index = np.arange(len(phi))

    if idx_min_mono > idx_fix:
        # Other larger
        idx_2 = index[phi > phi_fix]
    else:
        idx_2 = index[phi < phi_fix]

    idx_min = None
    for i in range(1):
        if idx_min is not None:
            if idx_min == 0:
                idx_min = 1
            elif idx_min == len(idx_2) - 1:
                idx_min = idx_min - 1
            idx_2 = np.lispace(idx_2[idx_min - 1], idx_2[idx_min + 1], 20)
        # Get indices for a diagonal
        indices = np.array(
            [[idx_fix] * len(idx_2), idx_2])
        indices = np.moveaxis(indices, 0, -1)

        # Get curve
        zoom_residual = res_interp_N(
            indices, BB, Bp, B, p, pp, Npix, vary_offset)
        idx_min = np.argmin(zoom_residual)


    idx_res = idx_2[idx_min]
    C_interp = index - np.floor(index)

    phi_res = np.exp((1 - C_interp) * np.log(phi[int(np.floor(idx_res))])
                      + C_interp * np.log(phi[int(np.ceil(idx_res))]))
    raise NotImplementedError()
    fit = FitResult(x=[phi_fix, phi_res],
                    dx=[0, phi_error],
                    x_distribution=np.squeeze(distribution),
                    basis_spectrum=spectrum,
                    residual=np.sum(res_fit, 0),
                    success=True, status=0, interp_coeff=C_interp)
    fit.x_range = phi_range.T
    return fit

def fit_2(profiles, Basis, phi):
    """Find the best monodisperse radius

    Parameters
    ----------
    M: 2d array
        The basis matrix. Mij = sum(basisi*basisj)
    b: 1d array
        bi = sum(profile*basisi)
    psquare: float
        psquare = sum(profiles*profile)
    phi:
        MUST BE SORTED
    Rs: 1d float
        The test radii [m]

    Returns
    -------
    radii: float
        The best radius fit
    """

    # Normalize the basis to fit profiles
    Basis_factor = np.mean(profiles, -1) / np.mean(Basis, -1)
    Basis = Basis * Basis_factor[..., np.newaxis]

    M, b, psquare = get_matrices(profiles, Basis)

    Nb = len(b)

    M_diag = np.diag(M)

    # Get 1d result. This is between the two!
    # Use that to limit search space
    res_1 = psquare + M_diag - 2 * b
    argmin_1 = np.argmin(res_1)

    coeff_basis, Bl_minus_Br_square = interpolate_1pos(
        argmin_1, argmin_1 + 1, np.diag(M), np.diag(M, 1), b)

    idx_min_mono = argmin_1 + coeff_basis

    N = np.min([idx_min_mono, len(b) - idx_min_mono])

    indices = np.array([np.linspace(idx_min_mono, idx_min_mono + N - 1, int(N)),
                        np.linspace(idx_min_mono, idx_min_mono - N - 1, int(N))])
    residual, fraction = residual_2_floating(indices, M, b, psquare)
    argmin_diag = np.argmin(residual)
    if argmin_diag == 0:
        raise RuntimeError("Monodisperse")
    XY = np.square(argmin_diag)

    factor = np.square(argmin_diag + 1) / XY * 1.1
    ratio = np.tan(
        np.linspace(
            np.arctan(XY / np.square(idx_min_mono - 1) * factor),
            np.arctan(np.square(len(b) - idx_min_mono - 1) / XY / factor),
            101)
    )[:, np.newaxis]
    product = np.exp(np.linspace(np.log(XY / factor),
                                  np.log(XY * factor),
                                  101))[np.newaxis, :]
    x = np.sqrt(product * ratio)
    y = np.sqrt(product / ratio)

    indices = np.asarray([idx_min_mono - x, y + idx_min_mono])

    valid = np.logical_and(indices > 0, indices < len(b) - 1)
    valid = np.logical_and(valid[0], valid[1])
    indices = indices[:, valid]

    residual, fraction = residual_2_floating(indices, M, b, psquare)

    # Get best
    idx = np.unravel_index(np.argmin(residual), np.shape(residual))
    frac = fraction[idx]
    index = indices[:, idx]

    if np.min(index) == 0 or np.max(index) == Nb - 1:
        raise RuntimeError("Fit out of range")

    # Get errors
    phi_error = np.zeros(2)
    spectrum = np.zeros(len(b))

    for rn, i in enumerate(index):
        i = int(i)
        j = i + 1
        if j == Nb:
            j = Nb - 2
        Bl_minus_Br_square = (M[i, i] + M[j, j] - M[i, j] - M[j, i])
        if Bl_minus_Br_square == 0:
            raise RuntimeError("No Gradient in Basis")
        error = (np.sqrt((phi[i] - phi[j])**2
                          / Bl_minus_Br_square))
        phi_error[rn] = error

    C0 = index
    min_res = minimize(res_interp_2, C0, args=(M, b, psquare),
                        jac=jac_interp_2,
                        method='BFGS', options={'gtol': 1e-16, 'norm': 2})

    index = np.sort(min_res.x)

    __, frac = residual_2_floating(index, M, b, psquare)

    # C < 0 mean interpolate to the left
    C_interp, idx_min = get_idx(index - np.floor(index),
                                np.asarray(np.floor(index), int))

    # Result
    phi_res = np.exp((1 - C_interp) * np.log(phi[idx_min[:, 0]])
                      + C_interp * np.log(phi[idx_min[:, 1]]))
    if phi_res[1] < phi_res[0]:
        phi_res = np.sort(phi_res)
        frac = 1 - frac
    # phi_res = (1 - c) * phi[idx_min[:, 0]] + c * phi[idx_min[:, 1]]
    prop_phi = np.asarray([1 - frac, frac])
    spectrum[idx_min] = (np.array([1 - C_interp, C_interp]).T
                          * prop_phi[:, np.newaxis])

    spectrum[idx_min] *= Basis_factor[idx_min]
    prop_phi *= np.ravel((1 - C_interp) * Basis_factor[idx_min[:, 0]]
                          + C_interp * Basis_factor[idx_min[:, 1]])
    fit = FitResult(x=phi_res, dx=phi_error, x_distribution=prop_phi,
                    basis_spectrum=spectrum, residual=min_res.fun,
                    success=True, status=0, interp_coeff=C_interp)
    fit.x_range = [[x - dx, x + dx] for x, dx in zip(fit.x, fit.dx)]

    phi_background_error = error_on_fit(
        profiles, Basis, phi, spectrum, idx_min)
    fit.phi_background_error = phi_background_error

    return fit

def jac_interp_2(index, M, b, psquare):
    """Jacobian function of res_interp_2"""
    if np.min(index) < 0 or np.max(index) > len(b) - 1:
        return index * np.nan
    nspecies = 2
    fraction = residual_2_floating(index, M, b, psquare)[1]
    C_phi = np.asarray([1 - fraction, fraction])

    interp_coeff = index - np.floor(index)
    index_floor = np.array(np.floor(index), int)
    index_ceil = np.array(np.ceil(index), int)

    idx = np.asarray([index_floor, index_ceil]).T

    BBk = np.zeros((nspecies, nspecies, 2))
    for i in range(nspecies):
        for j in range(nspecies):
            for k in range(2):
                BBk[i, j, k] = ((1 - interp_coeff[i]) * M[idx[i, 0], idx[j, k]]
                                + interp_coeff[i] * M[idx[i, 1], idx[j, k]])
    FitBk = np.zeros((nspecies, 2))
    for i in range(nspecies):
        for k in range(2):
            FitBk[i, k] = C_phi[0] * BBk[0, i, k] + C_phi[1] * BBk[1, i, k]

    dinterp = np.zeros(nspecies)
    for i in range(nspecies):
        dinterp[i] = 2 * C_phi[i] * (FitBk[i, 1] - FitBk[i, 0]
                                     + b[idx[i, 0]] - b[idx[i, 1]])

    return np.array(dinterp)

def res_interp_2(index, M, b, psquare):
    """Compute the residual for two spicies"""
    try:
        return residual_2_floating(index, M, b, psquare)[0]
    except RuntimeError as e:
        print(e)
        return np.nan

def jac_interp_N(index, BB, Bp, B, p, pp, Npix, vary_offset=False):
    """Jacobian function of res_interp_2"""
    if np.min(index) < 0 or np.max(index) > len(Bp) - 1:
        return index * np.nan

    __, coeff_a, coeff_b = residual_N_floating(
        index, BB, Bp, B, p, pp, Npix, vary_offset)

    interp_coeff = index - np.floor(index)
    index_floor = np.array(np.floor(index), int)
    index_ceil = np.array(np.ceil(index), int)

    C_i = interp_coeff[..., np.newaxis]
    i_idx_f = index_floor[..., np.newaxis]
    j_idx_f = index_floor[..., np.newaxis, :]
    i_idx_c = index_ceil[..., np.newaxis]
    j_idx_c = index_ceil[..., np.newaxis, :]

    BidBi = ((1 - C_i) * (BB[i_idx_f, j_idx_c] - BB[i_idx_f, j_idx_f])
             + C_i * (BB[i_idx_c, j_idx_c] - BB[i_idx_c, j_idx_f]))

    dBi = B[index_ceil] - B[index_floor]

    dBip = Bp[index_ceil] - Bp[index_floor]

    coeff_ai = coeff_a[..., np.newaxis]
    dres = 2 * (coeff_ai * BidBi @ coeff_ai
                + coeff_b * coeff_ai * dBi[..., np.newaxis]
                - coeff_ai * dBip[..., np.newaxis])

    return dres[..., 0]

def residual_2_floating(index, M, b, psquare):
    """Compute the residual and ratio for two spicies"""
    BB, By = get_matrices_interp_N(np.moveaxis(index, 0, -1), M, b)

    # v = y - Bi
    # w = Bj - Bi

    # v * w
    VW = (BB[..., 0, 0] - BB[..., 0, 1] - By[..., 0] + By[..., 1])
    WW = (BB[..., 1, 1] - 2 * BB[..., 0, 1] + BB[..., 0, 0])
    VV = (BB[..., 0, 0] - 2 * By[..., 0]) + psquare

    fraction = np.zeros(np.shape(BB[..., 0, 1]))
    valid = WW != 0
    fraction[valid] = VW[valid] / WW[valid]
    fraction[fraction > 1] = 1
    fraction[fraction < 0] = 0

    # Resibual for each combination
    residual = (- fraction * VW + VV)

    return residual, fraction

# def res_interp_2(C, M, b, psquare, idx):
#    C_phi = [1 - C[0], C[0]]
#    C_interp = C[1:]
#
#    By, BB = get_matrices_interp(idx + C_interp, M, b)
#
#    FitFit = (C_phi[0]**2 * BB[0, 0]
#              + 2 * C_phi[1] * C_phi[0] * BB[0, 1]
#              + C_phi[1]**2 * BB[1, 1])
#    Fity = C_phi[0] * By[0] + C_phi[1] * By[1]
#
#    residual = FitFit - 2 * Fity + psquare
#    return residual
#
#
# def jac_interp_2(C, M, b, psquare, idx):
#    nspecies = 2
#    C_phi = [1 - C[0], C[0]]
#    C_interp = C[1:]
#
#    By, BB = get_matrices_interp(idx + C_interp, M, b)
#    C_interp, idx = get_idx(C_interp, idx)
#
#    FitB = np.zeros(nspecies)
#    for i in range(nspecies):
#        FitB[i] = C_phi[0] * BB[0, i] + C_phi[1] * BB[1, i]
#
#    BBk = np.zeros((nspecies, nspecies, 2))
#    for i in range(nspecies):
#        for j in range(nspecies):
#            for k in range(2):
#                BBk[i, j, k] = ((1 - C_interp[i]) * M[idx[i, 0], idx[j, k]]
#                                + C_interp[i] * M[idx[i, 1], idx[j, k]])
#    FitBk = np.zeros((nspecies, 2))
#    for i in range(nspecies):
#        for k in range(2):
#            FitBk[i, k] = C_phi[0] * BBk[0, i, k] + C_phi[1] * BBk[1, i, k]
#
#    dinterp = np.zeros(nspecies)
#    for i in range(nspecies):
#        dinterp[i] = 2 * C_phi[i] * (FitBk[i, 1] - FitBk[i, 0]
#                                     + b[idx[i, 0]] - b[idx[i, 1]])
#
#    d0 = 2 * (FitB[1] - FitB[0] + By[0] - By[1])
#
#    return np.array([d0, *dinterp])

#    frac = min_res.x[0]
#    C_interp = np.asarray(min_res.x[1:])

#
#    sub_M = M[argmin_1:, :argmin_1+1]
#
#    fact_1 = (sub_M
#              - (M_diag - b)[argmin_1:, np.newaxis]
#              - b[np.newaxis, :argmin_1+1])
#    fact_2 = (M_diag[np.newaxis, :argmin_1+1]
#                - 2 * sub_M
#                + M_diag[argmin_1:, np.newaxis])
#    fact_3 = (M_diag - 2*b)[argmin_1:, np.newaxis] + psquare
#
#
#
#
#    fraction = np.zeros(np.shape(sub_M))
#    valid = fact_2 != 0
#    fraction[valid] = - fact_1[valid]/fact_2[valid]
#    fraction[fraction > 1] = 1
#    fraction[fraction < 0] = 0
#
#    # Resibual for each combination
# residual = (fraction**2 * fact_2
##                    + 2 * fraction * fact_1
# + fact_3)
#
#    residual = (fraction * fact_1 + fact_3)

#    # If basis not fine enough, we still have a fit, but success is False
#    if not np.all(np.diff(idx) > 3):
#        prop_phi = np.asarray([1 - frac, frac])
#        spectrum[idx] = prop_phi
#        fit = FitResult(x=phi[idx], dx=phi_error, x_distribution=prop_phi,
#                        basis_spectrum=spectrum, residual=np.min(residual),
#                        success=False, status=1)
#        return fit

    # Find interpolation for each position
#    C0 = [frac, 0, 0]