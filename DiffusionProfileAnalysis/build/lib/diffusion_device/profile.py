# -*- coding: utf-8 -*-
"""
Fitting and processing of profiles.

Created on Fri Mar 17 10:25:47 2017

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
from scipy.ndimage.filters import gaussian_filter1d as gfilter
from scipy.signal import savgol_filter

from .basis_generate import getprofiles
from . import display_data
from .profiles_fitting import fit_all, normalise_basis_factor


def size_profiles(infos, metadata, settings):
    """Size the profiles

     Parameters
    ----------
    profiles: 2d array
        List of profiles to fit
    pixel_size:float
        The pixel size in [m]
    metadata: dict
        The metadata
    settings: dict
        The settings

    Returns
    -------
    if nspecies==1:
        radii: float
            The best radius fit
    else:
        Rs, spectrum, the radii and corresponding spectrum
    """
    # load variables
    profiles = infos["Profiles"]
    nspecies = settings["KEY_STG_NSPECIES"]
    vary_offset = settings["KEY_STG_VARY_OFFSET"]
    test_radii = get_test_radii(settings)
    profile_slice = ignore_slice(
        settings["KEY_STG_IGNORE"], infos["Pixel size"])

    fit_index, readingpos = get_fit_index_and_pos(settings, metadata)
    fit_init, fit_profiles, fit_noise = get_fit_data(
        settings, fit_index, infos)
    # Separate the first profile
    fit_index = fit_index[1:]

    # Get basis function
    profiles_arg_dir = get_profiles_arg_dir(metadata, settings)
    Basis = getprofiles(fit_init, Radii=test_radii,
                        readingpos=readingpos,
                        infos=infos,
                        **profiles_arg_dir)
    fit_Basis = Basis[..., profile_slice]

    fit_profiles = fit_profiles / fit_noise
    fit_Basis = fit_Basis / fit_noise

    if settings["KEY_STG_FIT_SQUARE"]:
        fit_profiles = np.square(fit_profiles)
        fit_Basis = np.square(fit_Basis)

    # Get best fit
    fit = fit_all(
        fit_profiles,
        fit_Basis,
        test_radii,
        nspecies=nspecies,
        vary_offset=vary_offset,
    )

    infos["Radius error std"] = fit.dx
    infos["Radius range"] = fit.x_range

    phi_background_error = getattr(fit, 'phi_background_error', None)
    radius = (fit.x, fit.x_distribution)

    if phi_background_error is not None:
        # Get error on fit
        radius_error = np.zeros((nspecies, *np.shape(profiles))) * np.nan
        radius_error[:, fit_index, ..., profile_slice] = \
            phi_background_error
    else:
        radius_error = None

    Mfreepar = 1
    if nspecies == 1:
        radius = radius[0]
        if radius < np.min(test_radii):
            raise RuntimeError(
                'The test radius are too big! ({} < {})'.format(
                    radius, np.min(test_radii)))
        if radius > np.max(test_radii):
            raise RuntimeError(
                'The test radius are too small! ({} > {})'.format(
                    radius, np.max(test_radii)))
        if np.any(infos['Fit error'] > 1e-2):
            raise RuntimeError("The relative error is too large "
                               f"({100*infos['Fit error']:.2f}%), "
                               "please adapt the step factor."
                               "Make sure the radii is in log.")
        radius_error = radius_error[0]
    else:
        # TODO: fix nspecies == 0
        if nspecies != 0:
            # 2n-1 free parameter
            Mfreepar = 2 * nspecies - 1

    infos["Radius"] = radius
    infos["Radius error x"] = radius_error

    fits = get_fits(fit_init, infos, metadata, settings)

    get_fit_infos(profiles, fit_profiles, fits, profile_slice, Mfreepar,
                  infos, settings)

    infos["Fitted Profiles"] = fits
    return infos


def get_fits(init, infos, metadata, settings):
    """Size the profiles

     Parameters
    ----------
    init_profile: 1d array
        init profile
    metadata: dict
        The metadata
    settings: dict
        The settings
    """
    # load variables
    profiles = infos["Profiles"]
    nspecies = settings["KEY_STG_NSPECIES"]
    vary_offset = settings["KEY_STG_VARY_OFFSET"]
    profile_slice = ignore_slice(
        settings["KEY_STG_IGNORE"], infos["Pixel size"])

    fit_index, readingpos = get_fit_index_and_pos(settings, metadata)

    fit_init_index = fit_index[0]
    fit_index = fit_index[1:]

    # Get basis function
    profiles_arg_dir = get_profiles_arg_dir(metadata, settings)

    fits = np.zeros_like(profiles) * np.nan
    fits[fit_init_index] = init
    radius = infos["Radius"]
    if nspecies == 1:
        # fill data if needed
        if not np.isnan(radius):
            fits[fit_index] = getprofiles(
                init, Radii=[radius], readingpos=readingpos,
                infos=infos, **profiles_arg_dir)[0]
    else:
        radii, radii_spectrum = radius
        Basis = getprofiles(
                init, Radii=radii, readingpos=readingpos,
                infos=infos, **profiles_arg_dir)
        # fill data if needed
        fits[fit_index] = np.sum(
            radii_spectrum[:, np.newaxis, np.newaxis] * Basis, axis=0)

    # Normalise fits
    fact_a, fact_b = normalise_basis_factor(
            fits[fit_index, ..., profile_slice],
            profiles[fit_index, ..., profile_slice], vary_offset)
    fits[fit_index] = fact_a[..., np.newaxis] * \
        fits[fit_index] + np.array(fact_b)[..., np.newaxis]

    return fits


def ignore_slice(ignore, pixel_size):
    """Get a slice to ignore sides
    """
    ignore = int(ignore / pixel_size)
    if ignore == 0:
        profile_slice = slice(None)
    else:
        profile_slice = slice(ignore, -ignore)
    return profile_slice


def get_test_radii(settings):
    """Get test radii"""
    if settings["KEY_STG_RLOG"]:
        if settings["KEY_STG_R"] is None:
            raise RuntimeError("Please use the number of steps.")
        rmin, rmax, Nr = settings["KEY_STG_R"]
        test_radii = np.exp(np.linspace(np.log(rmin), np.log(rmax), int(Nr)))
    else:
        if settings["KEY_STG_R"] is not None:
            test_radii = np.linspace(*settings["KEY_STG_R"])
        else:
            test_radii = np.arange(*settings["KEY_STG_R_STEP"])

    if len(test_radii) == 0:
        raise RuntimeError("The test radius are incorrectly specified.")

    return test_radii


def get_reading_position(metadata, settings):
    """get_reading_position"""
    readingpos = np.asarray(metadata["KEY_MD_RPOS"])
    imslice = settings["KEY_STG_SLICE"]
    if imslice is not None:
        shift = np.resize([1, -1], len(readingpos)) * imslice[0]
        readingpos = readingpos + shift

    return readingpos


def get_profiles_arg_dir(metadata, settings):
    """get_profiles_arg_dir"""
    return {
        'Q': metadata["KEY_MD_Q"],
        'Wz': metadata["KEY_MD_WZ"],
        'Wy': metadata["KEY_MD_WY"],
        'temperature': metadata["KEY_MD_T"],
        'viscosity': metadata["KEY_MD_ETA"],
        'Zgrid': settings["KEY_STG_ZGRID"],
        'step_factor': settings["KEY_STG_DXFACTOR"],
        'zpos': metadata["KEY_MD_SCANZ"],
        }


def get_fit_index_and_pos(settings, metadata):
    """Get fit index and reading position"""
    # Select fit index
    fit_index = settings["KEY_STG_FITPOS"]
    readingpos = get_reading_position(metadata, settings)
    if fit_index is not None:
        fit_index = np.sort(fit_index)
    else:
        fit_index = np.arange(len(readingpos))

    # Put the index in order
    fit_readingpos = readingpos[fit_index]
    fit_index = fit_index[np.argsort(fit_readingpos)]
    fit_readingpos = np.sort(fit_readingpos)
    # First reading pos is initial profile
    fit_readingpos = fit_readingpos[1:] - fit_readingpos[0]
    return fit_index, fit_readingpos


def get_fit_data(settings, fit_index, infos):
    """get_fit_data"""
    profiles = infos["Profiles"]
    fit_profiles = profiles[fit_index]
    profile_slice = ignore_slice(
        settings["KEY_STG_IGNORE"], infos["Pixel size"])
    prof_noise = infos["Profiles noise std"]
    # If we have a different noise for each point
    if len(np.shape(prof_noise)) > 0:
        fit_noise = prof_noise[fit_index]
        fit_noise = fit_noise[1:]
        fit_noise = fit_noise[..., profile_slice]
    else:
        fit_noise = prof_noise

    # treat init profile
    initmode = settings["KEY_STG_POS0FILTER"]
    fit_init = init_process(fit_profiles[0], initmode, profile_slice)
    fit_profiles = fit_profiles[1:][..., profile_slice]

    # Check if init is large enough
    threshold = 3 * np.median(infos["Profiles noise std"])
    if np.max(fit_init[profile_slice]) < threshold:
        raise RuntimeError("signal to noise too low")

    return fit_init, fit_profiles, fit_noise


def get_fit_infos(profiles, fit_profiles, fits, profile_slice, Mfreepar,
                  infos, settings):

    infos["Signal over noise"] = np.mean(
        (profiles / infos["Profiles noise std"])[..., profile_slice])
    slicesize = np.sum(np.ones_like(fit_profiles)[..., profile_slice])
    nu = slicesize - Mfreepar
    reduced_least_square = ((np.nansum(np.square(
        ((profiles - fits) / infos["Profiles noise std"])[..., profile_slice])
        )) / nu)
    infos["Reduced least square"] = np.sqrt(reduced_least_square)

    ratio = infos["Reduced least square"] / infos["Signal over noise"]
    if settings["KEY_STG_LSE_THRESHOLD"] and ratio > 1:
        raise RuntimeError("Least square error too large")


def center(prof, subtract_mean=False):
    """
    Uses correlation between Y and the mirror image of Y to get the center

    Parameters
    ----------
    prof:  1d array
        Profile

    Returns
    -------
    center: float
        The center position in pixel units

    """

    # We must now detect the position of the center. We use correlation
    # Correlation is equivalent to least squares (A-B)^2=-2AB+ some constants
    prof = np.array(prof)
    if subtract_mean:
        prof -= np.nanmean(prof)
    prof[np.isnan(prof)] = 0
    Yi = prof[::-1]
    corr = np.correlate(prof, Yi, mode='full')
    X = np.arange(len(corr))
    args = np.argsort(corr)
    x = X[args[-7:]]
    y = corr[args[-7:]]
    coeffs = np.polyfit(x, np.log(y), 2)
    center = -coeffs[1] / (2 * coeffs[0])
    center = (center - (len(corr) - 1) / 2) / 2 + (len(prof) - 1) / 2
    return center


def baseline(prof, frac=.05):
    """
    Get the apparent slope from the base of the profile

    Parameters
    ----------
    prof:  1d array
        Profile
    frac: float, defaults .05
        Fraction of the profile to use

    Returns
    -------
    baseline: 1d array
        The baseline

    """
    # we use 5% on left side to get the correct 0:
    # Get left and right zeros
    argvalid = np.argwhere(np.isfinite(prof))
    lims = np.squeeze([argvalid[0], argvalid[-1]])
    left = int(lims[0] + frac * np.diff(lims))
    right = int(lims[1] - frac * np.diff(lims))
    leftZero = np.nanmean(prof[lims[0]:left])
    rightZero = np.nanmean(prof[right:lims[1]])

    # Send profile to 0
    baseline = np.linspace(leftZero, rightZero, len(prof))
    return baseline


def flat_baseline(prof, frac=.05):
    """
    Get the apparent slope from the base of the profile

    Parameters
    ----------
    prof:  1d array
        Profile
    frac: float, defaults .05
        Fraction of the profile to use

    Returns
    -------
    baseline: 1d array
        The flat baseline

    """
    # we use 5% on left side to get the correct 0:
    # Get left and right zeros
    leftZero = np.nanmean(prof[:int(frac * len(prof))])
    rightZero = np.nanmean(prof[-int(frac * len(prof)):])

    # Send profile to 0
    ret = np.zeros(prof.shape) + np.mean([leftZero, rightZero])
    return ret


def image_angle(image, maxAngle=np.pi / 7):
    """
    Analyse an image with y invariance to extract a small angle.

    Parameters
    ----------
    image:  2d array
        image with y invariance
    maxAngle: float, defaults np.pi/7
        Maximal rotation angle

    Returns
    -------
    angle: float
        The rotation angle

    """
    # Difference left 50% with right 50%
    # We want to slice in two where we have data
    argvalid = np.argwhere(np.isfinite(np.nanmean(image, 1)))
    lims = np.squeeze([argvalid[0], argvalid[-1]])
    # should we flatten this?
    top = np.nanmean(image[lims[0]:np.mean(lims, dtype=int)], 0)
    bottom = np.nanmean(image[np.mean(lims, dtype=int):lims[1]], 0)
    # Remouve nans
    top[np.isnan(top)] = 0
    bottom[np.isnan(bottom)] = 0
    # correlate
    C = np.correlate(top - np.mean(top), bottom - np.mean(bottom), mode='full')

    pos = np.arange(len(C)) - (len(C) - 1) / 2
    disty = ((lims[1] - lims[0]) / 2)
    Angles = np.arctan(pos / disty)

    valid = np.abs(Angles) < maxAngle
    x = pos[valid]
    c = C[valid]

    argleft = c.argmax() - 5
    if argleft < 0:
        argleft = 0
    x = x[argleft:c.argmax() + 6]
    y = np.log(gfilter(c, 2)[argleft:c.argmax() + 6])

    if np.any(np.isnan(y)):
        raise RuntimeError('The signal is too noisy!')

    coeff = np.polyfit(x, y, 2)
    x = -coeff[1] / (2 * coeff[0])
    angle = np.arctan(x / disty)

    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.arctan(pos/disty), C)
    plt.plot([maxAngle, maxAngle], [np.min(C), np.max(C)])
    plt.plot([-maxAngle, -maxAngle], [np.min(C), np.max(C)])
    plt.plot([angle, angle], [np.min(C), np.max(C)])
    #"""
    """
    import matplotlib.pyplot as plt
    x = np.arange(len(top))
    plt.figure()
    plt.plot(x, top)
    plt.plot(x+(C.argmax()-(len(C)-1)/2), bottom)
    plt.title('image angle')
    #"""
    return angle


def init_process(profile, mode, profile_slice):
    """
    Process the initial profile

    Parameters
    ----------
    profile:  1d array
        Profile to analyse
    mode: string
        'none':
            Nothing
        'gaussian':
            Return a gaussian fit
        'tails':
            Remove the tails
        'gfilter':
            Apply a gaussian filter of 2 px std
    profile_slice: slice
        The number of pixels to ignore on the edges

    Returns
    -------
    profile: 1d array
        the processed profile

    """
    Npix_ignore = profile_slice.start
    if Npix_ignore is not None:
        init = np.zeros_like(profile)
        init[profile_slice] = profile[profile_slice]
        init[:Npix_ignore] = profile[Npix_ignore]
        init[-Npix_ignore:] = profile[-Npix_ignore]
    else:
        init = np.array(profile)

    if mode == 'none':
        return init
    elif mode == 'gfilter':
        return gfilter(init, 2)
    elif mode == 'savgol':
        return savgol_filter(init, 31, 5)
    elif mode == 'gaussian' or mode == 'tails':
        Y = init
        X = np.arange(len(Y))
        valid = Y > .5 * Y.max()
        gauss = np.exp(np.poly1d(np.polyfit(X[valid], np.log(Y[valid]), 2))(X))
        if mode == 'gaussian':
            return gauss
        remove = gauss < .01 * gauss.max()
        init[remove] = 0
        return init


def get_fax(profiles):
    """
    returns a faxed verion of the profiles for easier plotting

    Parameters
    ----------
    profiles:  2d array
        List of profiles

    Returns
    -------
    profiles: 1d array
        The faxed profiles

    """
    return np.ravel(np.concatenate(
        (profiles, np.zeros((np.shape(profiles)[0], 1)) * np.nan), axis=1))


def rebin_profiles(profiles, rebin):
    """rebin profiles"""
    remove_axis = False
    if profiles.ndim == 1:
        profiles = profiles[np.newaxis]
        remove_axis = True

    if rebin > 1:
        rebin_profiles = np.zeros(
            (np.shape(profiles)[0], np.shape(profiles)[1] // rebin))
        kern = np.ones(rebin)/rebin
        for i in range(len(profiles)):
            rebin_profiles[i] = np.convolve(
                profiles[i], kern, mode='valid')[::rebin]
        profiles = rebin_profiles
    if remove_axis:
        profiles = profiles[0]
    return profiles


def process_profiles(infos, metadata, settings, outpath):
    """Process profiles according to settings

    Parameters
    ----------
    profiles: 2 dim floats array
        The profiles
    settings: dic
        The settings

    Returns
    -------
    profiles: 2 dim floats array
        The profiles
    """
    profiles = infos["Profiles"]
    pixel_size = infos["Pixel size"]

    rebin = settings["KEY_STG_REBIN"]
    profiles = rebin_profiles(profiles, rebin)
    pixel_size *= rebin

    profiles_filter = settings["KEY_STG_SGFILTER"]
    if profiles_filter is not None:
        filts = savgol_filter(
            profiles, profiles_filter[0], profiles_filter[1], axis=-1)
        display_data.save_plot_filt(profiles, filts, pixel_size,
                                    profiles_filter, outpath)
        profiles = filts

    infos["Pixel size"] = pixel_size
    infos["Profiles noise std"] = rebin_profiles(
            infos["Profiles noise std"], rebin) / np.sqrt(rebin)
    infos["Profiles"] = profiles
    return infos

def sliding_least_square(p1, p2):
    """Compute the sliding least square between p1 and p2."""
    lse = (np.correlate(p1**2, np.ones_like(p2))
           + np.correlate(np.ones_like(p1), p2**2)
            - 2 * np.correlate(p1, p2))
    return lse

