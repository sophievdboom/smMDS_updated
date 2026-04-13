# -*- coding: utf-8 -*-
"""
This file is used to create synthetic profile of diffusion and electrophorese.

Created on Mon Jan  9 09:32:10 2017

@author: quentinpeter

Copyright (C) 2017  Quentin Peter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
try:
    from .compute_profiles_c import compute_profiles
except ImportError:
    from .compute_profiles_py import compute_profiles


def getprofiles(Cinit, Q, Radii, readingpos, Wy, Wz, viscosity, temperature,
                Zgrid=None, muEoD=0, *, fullGrid=False, zpos=None,
                Boltzmann_constant=1.38e-23, Zmirror=True, stepMuE=False,
                step_factor=None, yboundary='Neumann', infos=None):
    """Returns the theorical profiles for the input variables

    Parameters
    ----------
    Cinit:  1d array or 2d array
            The initial profile. If 1D (shape (x, ) not (x, 1)) Zgrid is
            used to pad the array
    Q:  float
        The flux in the channel in [ul/h]
    Radii: 1d array
        The simulated radius. Must be in increasing order [m]
        OR: The mobilities (if stepMuE is True)
    readingpos: 1d array float
        Position to read at
    Wy: float
        Channel width [m]
    Wz: float
        Channel height [m]
    Zgrid:  integer, defaults 1
        Number of Z pixel if Cinit is unidimentional
    muEoD: float, default 0
        mobility times transverse electric field divided by diffusion constant
    fullGrid: bool , false
        Should return full grid?
    zpos: float, default None
        Z position of the profile. None for mean
    viscosity: float
        viscosity
    kT: float
        kT
    Zmirror: Bool, default True
        Should the Z mirror be used to bet basis functions
    stepMuE: Bool, default False
        Radii is in fact muEs
    step_factor: float, default 1
        Factor to change dx size if the step size seems too big (Useless?)
    yboundary: 'Neumann' or 'Dirichlet'
        constant derivative or value

    Returns
    -------
    profilespos: 3d array
        The list of profiles for the 12 positions at the required radii
    """

    D = get_D(
        Radii, viscosity, temperature, muEoD, stepMuE, Boltzmann_constant)
    phi, beta, mu_prime_E = get_unitless_parameters(
        Q, D, readingpos, Wy, Wz, muEoD)
    if zpos is not None:
        zpos = zpos/Wz

    phishape = np.shape(phi)
    phi = np.ravel(phi)

    profilespos, phi, dphi = get_unitless_profiles(
        Cinit, phi, beta, Zgrid=Zgrid,
        mu_prime_E=mu_prime_E, fullGrid=fullGrid, zpos=zpos,
        Zmirror=Zmirror, step_factor=step_factor, yboundary=yboundary)

    # reshape correctly
    profilespos.shape = (*phishape, *profilespos.shape[1:])

    # Get the fit error from rounding
    idx = np.argmin(readingpos)
    if readingpos[idx] == 0:
        idx = 1
    error = dphi / phi[idx]
    if infos is not None:
        infos['Fit error'] = error

    return profilespos


def get_D(Radii, viscosity, temperature, muEoD=0, stepMuE=False,
          Boltzmann_constant=1.38e-23):
    """
    """
    Radii = np.array(Radii)
    kT = Boltzmann_constant * temperature

    if stepMuE:
        if muEoD == 0:
            raise RuntimeError("Can't calculate for 0 qE")
        D = Radii / muEoD
    else:
        if np.any(Radii <= 0):
            raise RuntimeError("Can't work with negative radii!")
        D = kT / (6 * np.pi * viscosity * Radii)

    return D


def get_unitless_parameters(Q, D, readingpos, Wy, Wz, muEoD=0):
    """
    """
    # Prepare input and Initialize arrays
    readingpos = np.asarray(readingpos)

    mu_prime_E = muEoD * Wy
    beta = Wz / Wy
    Q = Q / (3600 * 1e9)  # transorm in m^3/s

    phi = readingpos[np.newaxis] * D[..., np.newaxis] / Q * beta

    return phi, beta, mu_prime_E


"""
The PDE is:
    dC/dx = D/V * (d2C/dy2 + d2C/dz2) - muE/V * dC/dy

We replace:
    x' = x * D / Q * beta
    y' = y / Wy
    z' = z / Wy
    beta = Wz / Wy
    V' = V * Wy^2 * beta / Q
    mu' = mu * Wy / D

The unitless equation is:
    V' dx'C = (dy'^2D + dz'^2C) - mu'E dy'C

"""
# @profile


def step_matrix_from_dic(
        step_matrix_dictionnary, Zgrid, Ygrid, beta,
        mu_prime_E, Zmirror, yboundary, step_factor=None, dphi_max=None):
    """
    Get a step matrix from a dictionary and populate the dictionnary for
    Faster access
    """
    poiseuille_prime = None

    # Get value for step_factor
    if step_factor is None:
        poiseuille_prime = poiseuille_unitless(Zgrid, Ygrid, beta)
        dphi, step_factor = get_dphi(Zgrid, Ygrid, beta,
                                     mu_prime_E=mu_prime_E, dphi_max=dphi_max,
                                     poiseuille_prime=poiseuille_prime)

    # Get key to identify matrix uniquely
    key = (Zgrid, Ygrid, beta, mu_prime_E, Zmirror, yboundary, step_factor)

    # Create dictionnary if doesn't exist
    if key in step_matrix_dictionnary:
        return step_matrix_dictionnary[key]
    else:
        F, dphi = stepMatrix(
            Zgrid, Ygrid, beta,
            mu_prime_E=mu_prime_E,
            Zmirror=Zmirror, step_factor=step_factor,
            yboundary=yboundary, poiseuille_prime=poiseuille_prime)

        Flist = F[np.newaxis]
        Fdic = {"Flist": Flist, 'dphi': dphi}
        step_matrix_dictionnary[key] = Fdic
        return Fdic

# @profile


def get_unitless_profiles(Cinit, phi, beta,
                          Zgrid=None, mu_prime_E=0, *, fullGrid=False, zpos=None,
                          Zmirror=True, step_factor=None, yboundary='Neumann',
                          step_matrix_dictionnary=None):
    """Returns the theorical profiles for the input variables

    Parameters
    ----------
    Cinit:  1d array or 2d array
            The initial profile. If 1D (shape (x, ) not (x, 1)) Zgrid is
            used to pad the array
    phi:    1d array
            Use approximately
            phi = reading_position * D / Q * beta
    beta:   float
            height over width
    Zgrid:  integer, defaults 1
            Number of Z pixel if Cinit is unidimentional
    muEoD:  float, default 0
            mobility times transverse electric field divided by diffusion constant
    fullGrid: bool , false
            Should return full grid?
    zpos:   float, default None
            Z position of the profile. None for mean
    viscosity: float
            viscosity
    kT:     float
            kT
    Zmirror: Bool, default True
            Should the Z mirror be used to bet basis functions
    step_factor: float, default 1
            Factor to change dx size if the step size seems too big
    yboundary: 'Neumann' or 'Dirichlet'
            constant derivative or value
    step_matrix_dictionnary: dict
            If not None, use instead of internal one

    Returns
    -------
    profilespos: 3d array
        The list of profiles for the 12 positions at the required radii
    """
#    from matplotlib.pyplot import figure, hist, plot, show
#    figure()
#    hist(np.log(phi), 100)
    # Clean input
    phi = np.asarray(phi)
    Cinit = np.asarray(Cinit)

    if np.any(phi < 0) or not np.all(np.isfinite(phi)):
        raise RuntimeError("The time varible is incorrect")

    # Functions to access F

    if step_matrix_dictionnary is None:
        if not hasattr(get_unitless_profiles, 'dirFList'):
            get_unitless_profiles.dirFList = {}
        step_matrix_dictionnary = get_unitless_profiles.dirFList

    # Zgrid
    if Zgrid is None:
        if Zmirror or len(Cinit.shape) < 2:
            raise RuntimeError('No Zgrid specified')
        Zgrid = np.shape(Cinit)[1]

    ZgridEffective = Zgrid
    if Zmirror:
        ZgridEffective = (Zgrid + 1) // 2

    # Cinit
    if len(Cinit.shape) < 2:
        Cinit = np.tile(Cinit[np.newaxis, :], (ZgridEffective, 1))
        if zpos is None:
            Cinit = Cinit / Zgrid
    else:
        if Cinit.shape[0] != ZgridEffective:
            raise RuntimeError("Cinit Z dim and Zgrid not aligned.")

    Ygrid = Cinit.shape[1]
    Nphi = len(phi)

    # get maximum acceptable dphi
    dphi_max = None
    if step_factor is None and len(phi) > 1:
        dphi_max = np.min(np.abs(np.diff(phi))) / 20
        if not dphi_max > 0:
            raise RuntimeError('dphi too small!')
        dphi_max_alt = 1e-2 * np.min(phi)  # max 1% error
        dphi_max = np.min([dphi_max, dphi_max_alt])

    # get step matrix
    Fdic = step_matrix_from_dic(
        step_matrix_dictionnary, Zgrid, Ygrid, beta,
        mu_prime_E, Zmirror, yboundary, step_factor, dphi_max)

    dphi = Fdic['dphi']

    # Get Nsteps for each radius and position
    Nsteps = np.asarray(np.round(phi / dphi), dtype=int)
    phi = dphi * Nsteps

    # transform Nsteps to binary array
    pow2 = 1 << np.arange(int(np.floor(np.log2(Nsteps.max()) + 1)))
    pow2 = pow2[:, None]
    NSteps_binary = np.bitwise_and(Nsteps[None, :], pow2) > 0

    # Sort for less calculations
    idx_sorted = np.lexsort(NSteps_binary[::-1])

    NSteps_binary = NSteps_binary.T
    profilespos = np.tile(np.ravel(Cinit), (Nphi, 1))
    profilespos = compute_profiles(
        NSteps_binary, idx_sorted, profilespos, Fdic)

    # reshape correctly
    profilespos.shape = (Nphi, ZgridEffective, Ygrid)

    if Zmirror:
        profilespos = np.concatenate(
            (profilespos, profilespos[..., -1 - Zgrid % 2::-1, :]), -2)
        Cinit = np.concatenate((Cinit, Cinit[-1 - Zgrid % 2::-1, :]), 0)

    # If full grid, stop here
    if fullGrid:
        return profilespos, phi, dphi

    if zpos is not None:
        idx = int(np.floor(Zgrid * zpos))
        # Border position
        if zpos == 1:
            idx = Zgrid - 1
        profilespos = profilespos[..., idx, :]
    else:
        # Take sum
        profilespos = np.sum(profilespos, -2)

    return profilespos, phi, dphi

# @profile


def getElectroProfiles(Cinit, Q, absmuEoDs, muEs, readingpos, Wy,
                       Wz, viscosity, temperature, Zgrid=None, *,
                       fullGrid=False, zpos=None,
                       boltzmann=1.38e-23,
                       Zmirror=True, step_factor=None,
                       yboundary='Neumann'):
    """Returns the theorical profiles for the input variables

    Parameters
    ----------
    Cinit:  1d array or 2d array
            The initial profile. If 1D (shape (x, ) not (x, 1)) Zgrid is
            used to pad the array
    Q:  float
        The flux in the channel in [ul/h]
    absmuEoDs: array floats
        absolute values of the muE/D to test
    muE: array float
        values od muE to test
    readingpos: 1d array float
        Position to read at
    Wy: float
        Channel width [m]
    Wz: float
        Channel height [m]
    Zgrid:  integer, defaults 1
        Number of Z pixel if Cinit is unidimentional
    fullGrid: bool , false
        Should return full grid?=
    zpos: float, default None
        Z position of the profile. None for mean
    viscosity: float
        viscosity
    kT: float
        kT
    Zmirror: Bool, default True
        Should the Z mirror be used to bet basis functions
    step_factor: float, default 1
        Factor to change dx size if the step size seems too big (Useless?)
    yboundary: 'Neumann' or 'Dirichlet'
        constant derivative or value

    Returns
    -------
    profilespos: 3d array
        The list of profiles for the 12 positions at the required radii

    """
#    kT = boltzmann * temperature
    muEs = np.asarray(muEs)
    absmuEoDs = np.abs(absmuEoDs)
    NqE = len(absmuEoDs)
    negmuE = muEs[muEs < 0]
    posmuE = muEs[muEs > 0]

    Nrp = len(readingpos)
    Ygrid = Cinit.shape[-1]

    def getret(muEs, muEoDs):
        NuEs = len(muEs)
        if fullGrid:
            rets = np.zeros((NqE, NuEs, Nrp, Zgrid, Ygrid))
        else:
            rets = np.zeros((NqE, NuEs, Nrp, Ygrid))

        for muEoD, ret in zip(muEoDs, rets):
            ret[:] = getprofiles(Cinit, Q, muEs, readingpos, Wy=Wy, Wz=Wz, Zgrid=Zgrid,
                                 muEoD=muEoD, fullGrid=fullGrid, viscosity=viscosity,
                                 temperature=temperature,
                                 Zmirror=Zmirror,
                                 zpos=zpos,
                                 stepMuE=True,
                                 step_factor=step_factor,
                                 yboundary=yboundary)
        return rets

    N_neg_muEs = len(negmuE)
    N_pos_muEs = len(posmuE)
    NmuEs = N_neg_muEs + N_pos_muEs
    if fullGrid:
        rets = np.zeros((NqE, NmuEs, Nrp, Zgrid, Ygrid))
    else:
        rets = np.zeros((NqE, NmuEs, Nrp, Ygrid))

    if N_neg_muEs > 0:
        rets[:, :N_neg_muEs] = getret(negmuE, -absmuEoDs)
    if N_pos_muEs > 0:
        rets[:, N_neg_muEs:] = getret(posmuE, absmuEoDs)

    return rets

# @profile


def poiseuille(*, Zgrid, Ygrid, Q, Wy, beta,
               yinterface=False, zinterface=False):
    """
    Compute the poiseuille flow profile

    Parameters
    ----------
    Zgrid:  integer
        Number of Z pixel
    Ygrid:  integer
        Number of Y pixel
    beta:  float
        height over width
    get_interface: Bool, defaults False
        Also returns poisuille flow between pixels
    Returns
    -------
    V: 2d array
        The poiseuille flow
    if get_interface is True:
    Viy: 2d array
        The poiseuille flow between y pixels
    Viz: 2d array
        The poiseuille flow between z pixels
    """

    V = poiseuille_unitless(Zgrid, Ygrid, beta, yinterface, zinterface)
    Q = Q / (3600 * 1e9)  # transorm in m^3/s
    return V * Q / (Wy**2 * beta)


def poiseuille_unitless(
        Zgrid, Ygrid, beta, yinterface=False, zinterface=False):
    """
    Compute the poiseuille flow profile

    Parameters
    ----------
    Zgrid:  integer
        Number of Z pixel
    Ygrid:  integer
        Number of Y pixel
    beta:  float
        height over width
    get_interface: Bool, defaults False
        Also returns poisuille flow between pixels
    Returns
    -------
    V: 2d array
        The poiseuille flow
    if get_interface is True:
    Viy: 2d array
        The poiseuille flow between y pixels
    Viz: 2d array
        The poiseuille flow between z pixels
    """
    # Check if already calculated
    key = (Zgrid, Ygrid, beta, yinterface, zinterface)
    if hasattr(poiseuille_unitless, 'saved_V'):
        if key in poiseuille_unitless.saved_V.keys():
            return poiseuille_unitless.saved_V[key]
    else:
        poiseuille_unitless.saved_V = {}

    # Poiseuille flow
    if yinterface:
        j = np.arange(1, Ygrid)[None, :, None, None]
    else:
        j = np.arange(Ygrid)[None, :, None, None] + .5

    if zinterface:
        i = np.arange(1, Zgrid)[:, None, None, None]
    else:
        i = np.arange(Zgrid)[:, None, None, None] + .5

    nz = np.arange(1, 100, 2)[None, None, :, None]
    ny = np.arange(1, 100, 2)[None, None, None, :]

    V = (16 * beta**2 / np.pi**4 *
         np.sum(1 / (nz * ny * (nz**2 + ny**2 * beta**2)) *
               (np.sin(nz * np.pi * i / Zgrid) *
                np.sin(ny * np.pi * j / Ygrid)), axis=(2, 3)))
    
    Vmean = (64 * beta**2 / np.pi**6 *
             np.sum(1 / (nz**2 * ny**2 * (nz**2 + ny**2 * beta**2))))

    V /= Vmean

    poiseuille_unitless.saved_V[key] = V

    return V


def get_dphi(Zgrid, Ygrid, beta, *,
             mu_prime_E=0, step_factor=None, dphi_max=None,
             poiseuille_prime=None):
    """
    Get dphi for step matrix

    Parameters
    ----------
    Zgrid:  integer
        Number of Z pixel
    Ygrid:  integer
        Number of Y pixel
    beta: float
        aspect ratio
    mu_prime_E: float, default 0
        In case of electrophoresis, q*E/k/T = muE/D[m^-1]
    step_factor: float <1
        Factor to change the value of dx
        if None, adjusted to dphi_max
    dphi_max: float
        Maximum acceptable dphi. If both step_factor and dphi_max are None,
        1 is used
    poiseuille_prime:
        The poseuille flow to avoid recomputing


    Returns
    -------
    dphi: float
        The step
    step_factor: float
        The factor applied
    """

    if poiseuille_prime is None:
        poiseuille_prime = poiseuille_unitless(Zgrid, Ygrid, beta)

    # Get steps
    dy = 1 / Ygrid
    dz = beta / Zgrid

    # get dx
    # The formula gives F=1+dx*D/V*(1/dy^2*dyyC+1/dz^2*dzzC)
    # Choosing dx as dx=dy^2*Vmin/D, The step matrix is:
    dphi = np.min((dy, dz))**2 * poiseuille_prime.min() / 2
    if mu_prime_E != 0:
        dphi2 = poiseuille_prime.min() / np.abs(mu_prime_E) * dy / 2
        dphi = np.min([dphi, dphi2])

    if step_factor is None:
        if dphi_max is None:
            step_factor = 1
        else:
            step_factor = np.exp(np.floor(np.log(dphi_max / dphi)))

    if step_factor > 1:
        step_factor = 1

    dphi *= step_factor

    return dphi, step_factor


def stepMatrix(Zgrid, Ygrid, beta, *, mu_prime_E=0, outV=None,
               method='Trapezoid', step_factor=None, dphi_max=None,
               Zmirror=False, yboundary='Neumann', poiseuille_prime=None):
    """
    Compute the step matrix and corresponding position step

    Parameters
    ----------
    Zgrid:  integer
        Number of Z pixel
    Ygrid:  integer
        Number of Y pixel
    Wz: float
        Channel height [m]
    Wy: float
        Channel width [m]
    muEoD: float, default 0
        In case of electrophoresis, q*E/k/T = muE/D[m^-1]
    outV: 2d float array
        array to use for the return
    method: string, default 'Trapezoid'
        Method for integration
        'Trapezoid': Mixed integration
         'Explicit': explicit integration
         'Implicit': implicit integration
    step_factor: float
        Factor to change the value of dx if None, adjusted to dphi_max
    dphi_max: float
        Maximum acceptable dphi. If both step_factor and dphi_max are None,
        1 is used
    Zmirror: bool, default False
        should we use a mirror for Z?
    yboundary: 'Neumann' or 'Dirichlet'
        constant derivative or value

    Returns
    -------
    F:  2d array
        The step matrix (independent on Q)
    dxtD: float
        The position step multiplied by the diffusion coefficient
    step_factor: float
        Returned if step_factor is None
    """

    realZgrid = Zgrid
    # Get Poiseille flow
    if poiseuille_prime is None:
        poiseuille_prime = poiseuille_unitless(Zgrid, Ygrid, beta)

    if outV is not None:
        outV[:] = poiseuille_prime

    # Get steps
    dy = 1 / Ygrid
    dz = beta / realZgrid

    dphi, step_factor = get_dphi(
        realZgrid, Ygrid, beta,
        mu_prime_E=mu_prime_E, step_factor=step_factor,
        dphi_max=dphi_max, poiseuille_prime=poiseuille_prime)

    # If the Z is a mirror, make adjustments
    Zodd = False
    if Zmirror:
        Zodd = Zgrid % 2 == 1
        Zgrid = (Zgrid + 1) // 2
        poiseuille_prime = poiseuille_prime[:Zgrid, :]

    # flatten poiseuille_over_Q
    poiseuille_prime = np.ravel(poiseuille_prime)

    # Get the dF matrix
    qy = getQy(Zgrid, Ygrid, boundary=yboundary)
    Cyy = (1 / poiseuille_prime)[:, np.newaxis] * \
        ((qy[-1] - 2 * qy[0] + qy[1]) / dy**2)

    if Zgrid > 1:
        qz = getQz(Zgrid, Ygrid, Zmirror, Zodd)
        Czz = (1 / poiseuille_prime)[:, np.newaxis] * \
            ((qz[-1] - 2 * qz[0] + qz[1]) / dz**2)
    else:
        Czz = 0

    if mu_prime_E == 0:
        Cy = 0
    else:
        ViyoQ = poiseuille_unitless(realZgrid, Ygrid, beta, yinterface=True)
        if Zmirror:
            ViyoQ = ViyoQ[:Zgrid, :]
#        Cy = getCy5(muEoD, dxtD, V, Zgrid, Ygrid, dy, boundary=yboundary)
        Cy = getCy(
            mu_prime_E,
            dphi,
            ViyoQ,
            Zgrid,
            Ygrid,
            dy,
            boundary=yboundary)

    dF = dphi * (Cyy + Czz - mu_prime_E * Cy)

    # Get F
    I = np.eye(Ygrid * Zgrid, dtype=float)
    if method == 'Explicit':
        # Explicit
        F = I + dF
    elif method == 'Implicit':
        # implicit
        F = np.linalg.inv(I - dF)
    elif method == 'Trapezoid':
        # Trapezoid
        F = np.linalg.inv(I - .5 * dF)@(I + .5 * dF)
    else:
        raise RuntimeError("Unknown integration Method: {}".format(method))

    # The maximal eigenvalue should be <= 1! otherwhise no stability
    # The above dphi should put it to 1
#    from numpy.linalg import eigvals
#    assert(np.max(np.abs(eigvals(F)))<=1.)

    return F, dphi


def getQy(Zgrid, Ygrid, boundary='Neumann'):
    """Get matrices to access neibours in y with correct boundary conditions

    Parameters
    ----------
    Zgrid:  integer
        Number of Z pixel
    Ygrid:  integer
        Number of Y pixel
    bounday: 'Neumann' or 'Dirichlet'
        constant derivative or value


    Returns
    -------
    qy:  3d array
        A list of matrices to access [-2, +2] y neighbors
    """
    # Create the q matrices
    q = np.zeros((5, Zgrid * Ygrid, Zgrid * Ygrid))
    for i in range(-2, 3):
        q[i] = np.diag(np.ones(Zgrid * Ygrid - np.abs(i)), i)

    for i in range(0, Zgrid * Ygrid, Ygrid):
        # border
        q[-2, i:i + 2, i] = 1
        q[-1, i, i] = 1
        q[1, i + Ygrid - 1, i + Ygrid - 1] = 1
        q[2, i + Ygrid - 2:i + Ygrid, i + Ygrid - 1] = 1
        if i > 0:
            q[:, i:i + 2, i - 2:i] = 0
            q[:, i - 2:i, i:i + 2] = 0

    if boundary == 'Dirichlet':
        q[:, ::Ygrid, :] = 0
        q[:, Ygrid - 1::Ygrid, :] = 0
    return q


def getQz(Zgrid, Ygrid, Zmirror, Zodd):
    """Get matrices to access neibours in z with correct boundary conditions

    Parameters
    ----------
    Zgrid:  integer
        Number of Z pixel
    Ygrid:  integer
        Number of Y pixel
    Zmirror: bool
        Is there a mirror at the bottom?
    Zodd: bool
        Is the mirror at the center ot the edge of the pixel?


    Returns
    -------
    qz:  3d array
        A list of matrices to access [-2, +2] z neighbors
    """
    if Zmirror and Zodd:
        shift = 1
    else:
        shift = 0

    def midx(i, j):
        def single(l):
            if l >= Zgrid:
                l = Zgrid - l - 1 - shift
            if l < -Zgrid:
                l = -Zgrid - l - 1
            if l < 0:
                L = np.arange(-Ygrid, 0) + (l + 1) * Ygrid
            else:
                L = np.arange(Ygrid) + l * Ygrid
            return L
        I, J = single(i), single(j)
        return I, J

    # Create the q matrices
    q = np.zeros((5, Zgrid * Ygrid, Zgrid * Ygrid))
    for i in range(-2, 3):
        q[i] = np.diag(np.ones(Ygrid * (Zgrid - np.abs(i))), i * Ygrid)
        # Border
        if i == -2:
            q[i][midx(1, 0)] = 1
            q[i][midx(0, 1)] = 1

        if i == -1:
            q[i][midx(0, 0)] = 1

        if i == 1:
            q[i][midx(-1, -1 - shift)] = 1

        if i == 2:
            q[i][midx(-2, -1 - shift)] = 1
            q[i][midx(-1, -2 - shift)] = 1

    return q


def getCy5(mu_prime_E, dx, V, Zgrid, Ygrid, dy, boundary='Neumann'):
    """Get Cy using the 5 point stencil technique

    Parameters
    ----------
    muEoD: float
        q*E/k/T = muE/D[m^-1]
    dxtD: float
        Time step multiplied by the diffusion coefficient
    V: 1d array
        Poiseulle flow. Size should be Zgrid*Ygrid
    Zgrid:  integer
        Number of Z pixel
    Ygrid:  integer
        Number of Y pixel
    dy: float
        Y step.
    bounday: 'Neumann' or 'Dirichlet'
        constant derivative or value


    Returns
    -------
    Cy:  2d array
        The 1/V*(d/dy) matrix

    """
    q = getQy(Zgrid, Ygrid, boundary=boundary)

    Cy = q[-2] - 8 * q[-1] + 8 * q[1] - q[2]
    Cy /= (12 * dy)
    Cy = (1 / V)[:, np.newaxis] * Cy
    return Cy


def getCy(mu_prime_E, dx, Viy, Zgrid, Ygrid, dy, boundary='Neumann'):
    """Get Cy using the flux conserving Fromm technique

    Parameters
    ----------
    muEoD: float
        q*E/k/T = muE/D[m^-1]
    dxtD: float
        Time step multiplied by the diffusion coefficient
    Viy: 1d array
        Poiseulle flow at the middle y points. Size should be Zgrid*(Ygrid-1)
    Zgrid:  integer
        Number of Z pixel
    Ygrid:  integer
        Number of Y pixel
    dy: float
        Y step.
    bounday: 'Neumann' or 'Dirichlet'
        constant derivative or value


    Returns
    -------
    Cy:  2d array
        The 1/V*(d/dy) matrix

    """
    # ravel the Viy with 1 zero between each z
    iVyp = np.ravel(np.concatenate((1 / Viy, np.zeros((Zgrid, 1))), 1))
    iVym = np.concatenate(([0], iVyp[:-1]))

    iVyp = iVyp[:, np.newaxis]
    iVym = iVym[:, np.newaxis]

    q = getQy(Zgrid, Ygrid, boundary=boundary)

    sigdy = np.zeros((3, Zgrid * Ygrid, Zgrid * Ygrid))
    for i in range(-1, 2):
        # Fromm choise of sigma
        sigdy[i] = (q[i + 1] - q[i - 1]) / 2

    # Get nu
    nu = mu_prime_E * dx

    # We want to represent the following equation as a matrix
    """\frac{1}{\Delta y}\left[
        \left( \frac{q_{i-1}}{V_{i-1/2}} - \frac{q_{i}}{V_{i+1/2}} \right)
        + \frac{1}{2}\left(
            \left( \frac{\Delta y \sigma_{i-1}}{V_{i-1/2}}
                - \frac{\Delta y \sigma_{i}}{V_{i+1/2}} \right)
            - \frac{u\Delta x}{\Delta y} \left(
                \frac{\Delta y \sigma_{i-1}}{V_{i-1/2}^2}
                - \frac{\Delta y \sigma_{i}}{V_{i+1/2}^2}
    \right)\right) \right]
    """

    neg = mu_prime_E < 0

    Cy = (iVym * q[-1 + neg] - iVyp * q[0 + neg]
          + (.5 - neg) * ((iVym * sigdy[-1 + neg] - iVyp * sigdy[0 + neg])
                          - nu / dy * (iVym**2 * sigdy[-1 + neg]
                                       - iVyp**2 * sigdy[0 + neg])))

    Cy /= dy

    # minus to be a differntial operator
    return -Cy
