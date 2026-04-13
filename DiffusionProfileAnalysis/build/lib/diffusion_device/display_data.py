# -*- coding: utf-8 -*-
"""
Functions used to display and save results of fitting

Created on Wed Aug  9 19:30:34 2017

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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import plot, figure
import os
from matplotlib.image import NonUniformImage
import shutil
import re
from . import profile as dp
from scipy.ndimage.filters import gaussian_filter1d


def save_plot_filt(profiles, filts, pixel_size, profiles_filter, outpath=None):
    figure()
    X = np.arange(len(dp.get_fax(profiles))) * pixel_size * 1e6

    plot(X, dp.get_fax(profiles), label="data")
    plot(X, dp.get_fax(filts), label="filtered")

    plt.xlabel(r'Position [$\mu$m]')
    plt.ylabel('Normalised amplitude')
    plt.title(
        "Savitzky-Golay: w{}, o{}".format(
            profiles_filter[0], profiles_filter[1]))
    plt.legend()
    if outpath is not None:
        plt.savefig(outpath + '_filt_fig.pdf')


def plot_single(radius, profiles, fits, lse, pixel_size,
                signal_noise, radius_range, prefix='', plot_error=False,
                radius_error_x=None, save_prefix=None):
    # =========================================================================
    # Fit
    # =========================================================================
    if plot_error and radius_error_x is not None and np.ndim(radius_range) > 1:
        N = len(radius)
        for i in range(N):
            save_p = None
            if save_prefix:
                save_p = save_prefix + f'_{i}'
            figure()
            plot_single(
                radius[i], profiles, fits, lse, pixel_size,
                signal_noise, radius_range[i], f"({i + 1}/{N})" + prefix,
                plot_error, radius_error_x[i],
                save_prefix=save_p)
        return

    if len(np.shape(radius)) > 0:
        title = (prefix + 'LSE = {:.3f}, pixel = {:.3f} um'.format(
            lse, pixel_size * 1e6))
    else:
        title = (prefix + 'r= {:.2f} [{:.2f}; {:.2f}]nm, LSE = {:.3f}, '
                 'pixel = {:.3f} um'.format(
                     radius * 1e9,
                     radius_range[0] * 1e9,
                     radius_range[1] * 1e9,
                     lse,
                     pixel_size * 1e6))
    # =========================================================================
    # Plot
    # =========================================================================

    plt.title(title)

    X = np.arange(len(dp.get_fax(profiles))) * pixel_size * 1e6

    plot(X, dp.get_fax(profiles), 'C0', label="Profiles", zorder=10)
    plt.xlabel(r'Position [$\mu$m]')
    plt.ylabel('Normalised amplitude')
    if fits is not None:
        plot(X, dp.get_fax(fits), 'C1', label="Fits", zorder=12)
        plt.fill_between(X, dp.get_fax(fits - signal_noise),
                         dp.get_fax(fits + signal_noise),
                         color="C1", alpha=0.5, zorder=11)
        if plot_error:
            plt.twinx()
            plt.plot([0], [0], 'C0', label="Profiles")
            plt.plot([0], [0], 'C1', label="Fits")
            if radius_error_x is not None:
                for i in range(len(radius_error_x)):
                    radius_error_x[i, np.isfinite(radius_error_x[i])] = (
                        gaussian_filter1d(
                            radius_error_x[i, np.isfinite(radius_error_x[i])],
                            4))
                Y = dp.get_fax(radius_error_x) * 1e9
                plt.fill_between(X, np.zeros_like(Y), Y, color='C2', alpha=0.4,
                                 label="Radius error", zorder=9)
                plt.ylabel('Radius error / nm')
            else:
                square_difference = np.square((fits - profiles) / signal_noise)
                plt.plot(X, dp.get_fax(square_difference), 'C2',
                         label="Square error", zorder=9)
                plt.ylabel('Square error')

    plt.legend()

    if save_prefix:
        plt.tight_layout()
        plt.savefig(save_prefix + '_fig.pdf')


def plot_and_save(infos, settings, outpath=None):
    """Plot the sizing data

    Parameters
    ----------
    radius: float or list of floats
        If 1 species:
            The fitted radius
        If several species:
            radii, spectrum: The radii and corresponding coefficients
    profiles: 2d list of floats
        The extracted profiles
    fits: 2d list of floats
        The fits
    pixel_size: float
        The detected pixel size
    outpath: path
        Folder where to save the figures and data

    """
    radius = infos['Radius']
    profiles = infos['Profiles']
    fits = infos['Fitted Profiles']
    lse = infos["Reduced least square"]
    pixel_size = infos["Pixel size"]
    radius_error = infos["Radius error std"]
    radius_range = infos["Radius range"]
    signal_over_noise = infos["Signal over noise"]
    spectrum = 1

    if len(np.shape(radius)) > 0:
        radius, spectrum = radius
        figure()
        plt.errorbar(
            radius * 1e9, spectrum,
            xerr=np.transpose(
                np.abs(radius_range - radius[:, np.newaxis])) * 1e9,
            fmt='x')
        plt.xscale('log')
        plt.xlabel("Radius [nm]")
        plt.ylabel("Coefficient")
        if outpath is not None:
            plt.savefig(outpath + '_rSpectrum_fig.pdf')
        plt.title('; '.join([f"r= {r:.2f} [{rng[0]:.2f}; {rng[1]:.2f}]nm"
                             for r, rng in zip(radius * 1e9,
                                               np.asarray(radius_range)*1e9)]))
    figure()
    plot_single(radius, profiles, fits, lse, pixel_size,
                infos["Profiles noise std"], radius_range,
                plot_error=settings['KEY_STG_PLOT_ERROR'],
                radius_error_x=infos["Radius error x"],
                save_prefix=outpath)

    # =========================================================================
    # Save
    # =========================================================================

    if outpath is not None:
        with open(outpath + '_result.txt', 'wb') as f:
            f.write("Reduced least square: {:f}\n".format(lse).encode())
            f.write("Apparent pixel size: {:f} um\n".format(pixel_size *
                                                            1e6).encode())
            f.write("Signal over noise: {:f}\n".format(
                signal_over_noise).encode())
            if len(np.shape(radius)) > 0:
                f.write("radius:\n".encode())
                np.savetxt(f, radius)
                f.write("Spectrum:\n".encode())
                np.savetxt(f, spectrum)
                f.write("Radius error std:\n".encode())
                np.savetxt(f, radius_error)

            else:
                f.write("Radius: {:f} nm\n".format(radius * 1e9).encode())
                f.write(
                    "Radius error std: {:f} nm\n".format(
                        radius_error * 1e9).encode())
            f.write("Profiles:\n".encode())
            np.savetxt(f, profiles)
            f.write('Fits:\n'.encode())
            np.savetxt(f, fits)
    return infos


def plot_and_save_stack(infos, settings, outpath=None):
    """Plot the sizing data

    Parameters
    ----------
    radius: list of floats
        A list of:
        If 1 species:
            The fitted radius
        If several species:
            radii, spectrum: The radii and corresponding coefficients
    profiles: 3d list of floats
        The extracted profiles
    fits: 3d list of floats
        The Fits
    pixel_size: list of floats
        The detected pixel size.
    images: array of floats
        The data that was analysed
    overexposed: list of bool
        For each data file, is the file overexposed?
    outpath: path
        Folder where to save the figures and data
    plotpos: array of ints
        Positions to plot if this is a stack

    """
    success = infos.loc[:, "Radius"].notna()

    radius = infos.loc[success, "Radius"]
    x = radius.index
    overexposed = infos.loc[success, "Overexposed"]
    radius_error = infos.loc[success, "Radius error std"]
    radius_range = infos.loc[success, "Radius range"]
    LSE = infos.loc[success, "Reduced least square"]
    signal_over_noise = infos.loc[success, "Signal over noise"]
    profiles_noise_std = infos.loc[success, "Profiles noise std"]
    profiles = infos.loc[success, "Profiles"]
    fits = infos.loc[success, "Fitted Profiles"]
    pixel_size = infos.loc[success, "Pixel size"]

    valid = np.logical_not(overexposed)
    plotpos = settings["KEY_STG_STACK_POSPLOT"]

    intensity = np.array([np.nanmean(p) for p in profiles])
    infos.loc[success, "Mean Intensity"] = intensity

    # If more than 1 analyte
    if len(np.shape(radius.iloc[0])) == 2:
        # IF spectrum
        if np.shape(radius.iloc[0])[1] == settings['KEY_STG_R'][-1]:
            Rs = radius.iloc[0][0] * 1e9
            ylim = (0, len(radius))
            xlim = (np.min(Rs), np.max(Rs))
            figure()
            im = NonUniformImage(plt.gca(), extent=(*xlim, *ylim))
            im.set_data(Rs, np.arange(len(radius)),
                        np.stack(radius.apply(lambda x: x[1])))
            plt.gca().images.append(im)
            plt.xlim(*xlim)
            plt.ylim(*ylim)
            plt.xlabel('Radius [nm]')
            plt.ylabel('Frame number')
            if outpath is not None:
                plt.savefig(outpath + '_R_fig.pdf')
        # if list
        else:
            for i in range(np.shape(radius.iloc[0])[0]):
                figure()
                y = radius.loc[valid].apply(lambda x: x[0][i]) * 1e9
                yerr = np.abs(np.stack(
                        radius_range.loc[valid].apply(
                                lambda x: x[i])).T * 1e9
                        - y.to_numpy())
                plt.errorbar(x[valid], y, yerr=yerr, fmt='x', label='data')
                plt.xlabel('Frame number')
                plt.ylabel('Radius [nm]')
                plt.title(f'Radius {i+1}')
                if outpath is not None:
                    plt.savefig(outpath + f'_R{i+1}_fig.pdf')
            figure()
            plot(np.stack(radius.loc[valid].apply(lambda x: x[1])), 'x')
            plt.xlabel('Frame number')
            plt.ylabel('Fraction')
            plt.legend([f'Radius{i+1}' for i in range(np.shape(radius.iloc[0])[0])])
            if outpath is not None:
                plt.savefig(outpath + '_fractions_fig.pdf')

    else:
        figure()
        plt.errorbar(x[valid], radius[valid] * 1e9,
                     yerr=np.abs(np.stack(radius_range).T
                                 - radius.to_numpy())[..., valid] * 1e9,
                     fmt='x', label='data')
        plt.xlabel('Frame number')
        plt.ylabel('Radius [nm]')
        if np.any(overexposed):
            plt.errorbar(x[overexposed], radius[overexposed] * 1e9,
                         yerr=np.abs(
                np.stack(radius_range.to_numpy()).T - radius.to_numpy()
            )[..., overexposed] * 1e9,
                fmt='x',
                label='overexposed data')
            plt.legend()
        plt.yscale('log')
        if outpath is not None:
            plt.savefig(outpath + '_R_fig.pdf')

    figure()
    plot(x[valid], LSE[valid], 'x', label='regular')
    plt.xlabel('Frame number')
    plt.ylabel('Reduced least square')
    if np.any(overexposed):
        plot(x[overexposed], LSE[overexposed], 'x', label='overexposed')
        plt.legend()
    if outpath is not None:
        plt.savefig(outpath + '_LSE_fig.pdf')

    figure()
    plt.semilogy(x[valid], (LSE / signal_over_noise)[valid],
                 'x', label='regular')
    plt.xlabel('Frame number')
    plt.ylabel('Normalised reduced least square')
    if np.any(overexposed):
        plt.semilogy(x[overexposed], (LSE / signal_over_noise)[overexposed],
                     'x', label='overexposed')
        plt.legend()
    if outpath is not None:
        plt.savefig(outpath + '_LSE_o_SON_fig.pdf')

    figure()
    plot(x[valid], intensity[valid], 'x', label='regular')
    plt.xlabel('Frame number')
    plt.ylabel('Mean intensity')
    if np.any(overexposed):
        plot(x[overexposed], intensity[overexposed], 'x', label='overexposed')
        plt.legend()
    if outpath is not None:
        plt.savefig(outpath + '_mean_intensity_fig.pdf')

    if len(np.shape(pixel_size)) > 0:
        figure()
        plot(x, pixel_size * 1e6, 'x')
        plt.xlabel('Frame number')
        plt.ylabel('Pixel size')
        if outpath is not None:
            plt.savefig(outpath + '_pixel_size_fig.pdf')

    if outpath is not None:
        selected_keys = [
            "Radius",
            "Radius range",
            "Radius error std",
            "Signal over noise",
            "Reduced least square",
            "Mean Intensity",
            "Overexposed",
            "Pixel size",
            "Profiles",
            "Fitted Profiles",
            "Profiles noise std",
            ]
        infos.loc[np.logical_not(infos.loc[:, 'Error']),
                  selected_keys].to_csv(outpath + '_result.csv')

    if plotpos is not None:
        for pos in plotpos:
            argmin = np.argmin(np.abs(pos - profiles.index))
            pos = profiles.index[argmin]
            if profiles[pos] is None:
                continue
            pixs = pixel_size
            if len(np.shape(pixel_size)) > 0:
                pixs = pixel_size[pos]
            figure()
            plot_single(radius[pos], profiles[pos], fits[pos], LSE[pos],
                        pixs, profiles_noise_std[pos],
                        radius_range[pos], prefix=f'{pos}: ',
                        plot_error=settings['KEY_STG_PLOT_ERROR'],
                        radius_error_x=infos.at[pos, "Radius error x"],
                        save_prefix = outpath + '_{}'.format(pos))
    return infos


def prepare_output(outpath, settingsfn, metadatafn):
    """Prepare output folder

    Parameters
    ----------
    outpath: path
        Folder where to save the figures and data
    settingsfn: path
        path to the fit settings file
    metadatafn: path
        path to the fit settings file

    Returns
    -------
    base_name: path
        The prefix to use to save data

    """
    base_name = None
    if outpath is not None:
        settings_name = os.path.splitext(os.path.basename(settingsfn))[0]
        try:
            commonpath = os.path.commonpath([settingsfn, metadatafn])
            metadata_name = os.path.splitext(
                os.path.relpath(metadatafn, commonpath))[0]
        except ValueError:
            metadata_name = os.path.splitext(os.path.basename(metadatafn))[0]

        if re.match("metadata$", metadata_name, re.IGNORECASE):
            """We can not just use metadata as a metadata name."""
            metadata_name = os.path.basename(os.path.dirname(metadatafn))
        if re.match(".+metadata$", metadata_name, re.IGNORECASE):
            """Remove the superfluous metadata at the end"""
            metadata_name = metadata_name[:-8]
        if len(metadata_name) > 0 and metadata_name[-1] == '_':
            """Remove trailing _"""
            metadata_name = metadata_name[:-1]
        newoutpath = os.path.join(
            outpath,
            settings_name)

        folder_path = os.path.join(newoutpath, os.path.dirname(metadata_name))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        shutil.copy(
            settingsfn,
            os.path.join(
                newoutpath,
                settings_name +
                '.json'))

        base_name = os.path.join(
            newoutpath,
            metadata_name)
    return base_name


def plot_wide_profiles(infos, metadata, settings, save_prefix=None,
                       new_figure=False):
    """Print the profiles to check if they are correctly placed."""
    if new_figure:
        plt.figure()
    channel_width = metadata["KEY_MD_WY"]
    N = len(infos["Wide Profiles"])
    for idx, (X, prof) in enumerate(infos["Wide Profiles"]):
        label = 'Profiles' if idx == 0 else None
        plt.plot(X * 1e6, prof, c=plt.cm.plasma(idx / (N - 1)), label=label)
    ylim = plt.ylim()
    plt.plot(np.ones(2) * channel_width / 2 * 1e6, ylim, 'r--', label='Walls')
    plt.plot(-np.ones(2) * channel_width / 2 * 1e6, ylim, 'r--')
    plt.xlabel('Position [um]')
    plt.title("Walls")
    if save_prefix:
        plt.tight_layout()
        plt.savefig(save_prefix + '_walls.pdf')


def plot_wide_profiles_stack(infos, metadata, settings, save_prefix=None):
    """Print the profiles to check if they are correctly placed."""
    plotpos = settings["KEY_STG_STACK_POSPLOT"]

    if plotpos is None:
        return
    for pos in plotpos:
        prefix = None
        if save_prefix:
            prefix = save_prefix + f"_{pos}"
        if pos not in infos["Wide Profiles"]:
            continue
        inf = {"Wide Profiles": infos["Wide Profiles"][pos]}
        figure()
        plot_wide_profiles(inf, metadata, settings, prefix)
        plt.title(f'{pos}: Walls')
