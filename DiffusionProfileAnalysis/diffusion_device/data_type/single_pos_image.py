# -*- coding: utf-8 -*-
"""
Analyse 12 positions device

Created on Fri Mar 17 10:26:20 2017

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
import background_rm as rmbg
import registrator.image as ir
import warnings
import cv2
from scipy import interpolate
from registrator.image import is_overexposed
import tifffile
import matplotlib.pyplot as plt

from .images_files import ImagesFile
from .. import profile as dp

warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)


class SinglePosImage(ImagesFile):
    def __init__(self, metadata, settings, outpath):
        super().__init__(metadata, settings, outpath)

    def load_data(self):
        """load data from metadata

        Parameters
        ----------
        metadata: dict
            The metadata information
        infos: dict
            Dictionnary with other infos

        Returns
        -------
        data: array
            the image
        """
        filename = self.metadata["KEY_MD_FN"]
        data = self.load_images(filename)
        return data

    def process_data(self, data):
        """Do some data processing

        Parameters
        ----------
        data: array
            The data to process
        metadata: dict
            The metadata information
        settings: dict
            The settings
        infos: dict
            Dictionnary with other infos

        Returns
        -------
        data: array
            The processed data
        """
        overexposed = is_overexposed(data)
        data, backgrounds = self.process_background(data)
        data, pixel_size = self.process_images(data, backgrounds)
        infos = {}
        infos["Overexposed"] = overexposed
        infos["Pixel size"] = pixel_size
        infos["Data"] = data
        return infos

    def get_profiles(self, infos):
        """Do some data processing

        Parameters
        ----------
        data: array
            The data to process
        metadata: dict
            The metadata information
        settings: dict
            The settings
        infos: dict
            Dictionnary with other infos

        Returns
        -------
        profiles: array
            The profiles
        """
        pixel_size = infos["Pixel size"]
        channel_width = self.metadata["KEY_MD_WY"]
        Npix = int(channel_width // pixel_size) + 1
        data = infos["Data"]
        profiles = np.zeros((len(data), Npix))
        flowdir = self.metadata["KEY_MD_FLOWDIR"]
        noise = np.zeros(len(data))
        wide_profiles = []

        for i, (im, fd) in enumerate(zip(data, flowdir)):
            if fd == 'u':
                pass
            elif fd == 'l':
                im = np.rot90(im, 1)
            elif fd == 'd':
                im = np.rot90(im, 2)
            elif fd == 'r':
                im = np.rot90(im, 3)
            else:
                raise RuntimeError('Unknown orientation')

            X, prof = self.extract_profile(
                im, pixel_size, channel_width)

            # We restrict the profile to channel width - widthcut
            Npix = int(channel_width // pixel_size) + 1

            Xc = np.arange(Npix) - (Npix - 1) / 2
            Xc *= pixel_size

            finterp = interpolate.interp1d(X, prof, bounds_error=False,
                                           fill_value=0)
            profiles[i] = finterp(Xc)

            mask_out = np.logical_and(np.abs(X) > channel_width / 2,
                                      np.isfinite(prof))
            noise[i] = np.std(prof[mask_out])
            wide_profiles.append((X, prof))

        noise = np.mean(noise)
        infos["Profiles noise std"] = noise
        infos["Profiles"] = profiles
        infos["Wide Profiles"] = wide_profiles
        return infos

    def savedata(self, infos):
        """Save the data"""
        tifffile.imsave(self.outpath + '_ims.tif', infos['Data'])

    def process_images(self, images, backgrounds, rebin=2):
        """
        Get the hydrodynamic radius from the images

        Parameters
        ----------
        images: 1d list of images
            images to treat
        backgrounds: 1d list of image, default None
            background images
            if None, no background image
        metadata: dict
            The metadata
        settings: dict
            The settings
        rebin: int, defaults 2
            Rebin factor to speed up code

        Returns
        -------
        r: float
            Radius in [m]

        """

        # Check images is numpy array
        images = np.asarray(images)
        pixel_size = self.metadata["KEY_MD_PIXSIZE"]
        Wy = self.metadata["KEY_MD_WY"]

        if backgrounds is not None:
            # Check bgs is numpy array
            backgrounds = np.asarray(backgrounds)

        if rebin > 1:
            size = tuple(np.array(np.shape(images)[1:][::-1]) // rebin)
            images = np.array(
                [cv2.resize(im, size, interpolation=cv2.INTER_AREA)
                 for im in images])
            pixel_size *= rebin

            if backgrounds is not None:
                size = tuple(
                    np.array(np.shape(backgrounds)[1:][::-1]) // rebin)
                backgrounds = np.array(
                    [cv2.resize(im, size, interpolation=cv2.INTER_AREA)
                     for im in backgrounds])

        # Get flat images
        if backgrounds is None:
            # Single images
            flatimages = np.asarray(
                [self.flat_image(im, pixel_size, Wy)
                 for im in images])
        else:
            # images and background
            flatimages = np.asarray(
                [self.remove_bg(im, bg, pixel_size, Wy)
                 for im, bg in zip(images, backgrounds)])

        # Orientate
        for flatim in flatimages:
            flatim[:] = ir.rotate_scale(flatim, -dp.image_angle(flatim),
                                        1, borderValue=np.nan)

        return flatimages, pixel_size

    def remove_bg(self, im, bg, pixel_size, chanWidth):
        """
        Remove background from image

        Parameters
        ----------
        im: 2d array
            image
        bg: 2d array
            background
        pixel_size: float
            pixel size in [m]
        chanWidth: float
            channel width  in [m]

        Returns
        -------
        im: 2d array
            The processed image

        """
        im = np.array(im, dtype=float)
        bg = np.array(bg, dtype=float)
        # remove dust peaks on images
        bg[rmbg.getPeaks(bg, maxsize=50 * 50)] = np.nan
        im[rmbg.getPeaks(im, maxsize=50 * 50)] = np.nan

        # Get the X positions (perpendicular to alignent axis) and check wide
        # enough
        X = np.arange(im.shape[1]) * pixel_size
        if not (1.2 * chanWidth < X[-1]):
            raise RuntimeError("image too small to get entire channel.")

        # Get the approximate expected channel position
        channel = np.absolute(X - X[-1] / 2) < .6 * chanWidth

        # Create mask to ignore channel when flattening image
        mask = np.ones(im.shape, dtype=bool)
        mask[:, channel] = False

        # Get data
        return rmbg.remove_curve_background(im, bg, maskim=mask)

    def flat_image(self, im, pixel_size, chanWidth):
        """
        Flatten the image

        Parameters
        ----------
        im: 2d array
            image
        pixel_size: float
            pixel size in [m]
        chanWidth: float
            channel width  in [m]

        Returns
        -------
        im: 2d array
            The flattened image
        """

        im = np.asarray(im, dtype=float)
        # remove peaks
        im[rmbg.getPeaks(im, maxsize=20 * 20)] = np.nan
        # straighten
        angle = dp.image_angle(im - np.nanmedian(im))
        im = ir.rotate_scale(im, -angle, 1, borderValue=np.nan)

        # Get center
        prof = np.nanmean(im, 0)  # TODO: Maybe median?
        flatprof = prof - np.linspace(np.nanmean(prof[:len(prof) // 10]),
                                      np.nanmean(prof[-len(prof) // 10:]),
                                      len(prof))
        flatprof[np.isnan(flatprof)] = 0
        # TODO: Fail ->np.argmax?
        x = np.arange(len(prof)) - dp.center(flatprof)
        x = x * pixel_size

        # Create mask
        channel = np.abs(x) < chanWidth / 2
        mask = np.ones(np.shape(im))
        mask[:, channel] = 0

        # Flatten
        im = im / rmbg.polyfit2d(im, mask=mask) - 1

        """
        from matplotlib.pyplot import figure, imshow, plot
        figure()
        imshow(im)
        imshow(mask, alpha=.5, cmap='Reds')
    #    plot(x, flatprof)
    #    plot(x, np.correlate(flatprof, flatprof[::-1], mode='same'))
        #"""

        return im

    def extract_profile(self, flatim, pixel_size, chanWidth, center=None,
                        *, reflatten=True, ignore=10):
        """
        Get profile from a flat image

        Parameters
        ----------
        flatim: 2d array
            flat image
        pixel_size: float
            pixel size in [m]
        chanWidth: float
            channel width  in [m]
        center: float
            The position of the center of the profile
        reflatten: Bool, defaults True
            Should we reflatten the profile?
        ignore: int, defaults 10
            The number of pixels to ignore if reflattening

        Returns
        -------
        im: 2d array
            The flattened image
        """

        # get profile
        prof = np.nanmean(flatim, 0)

        # Center X
        X = np.arange(len(prof)) * pixel_size

        if center is None:
            center = dp.center(prof) * pixel_size
            inchannel = np.abs(X - center) < .45 * chanWidth
            center = dp.center(prof[inchannel]) + np.argmax(inchannel)

        X = X - center * pixel_size

        # get what is out
        out = np.logical_and(np.abs(X) > .55 * chanWidth, np.isfinite(prof))

        if reflatten:
            # fit ignoring extreme 10 pix
            fit = np.polyfit(X[out][ignore:-ignore],
                             prof[out][ignore:-ignore], 2)
            bgfit = fit[0] * X**2 + fit[1] * X + fit[2]

            # Flatten the profile
            prof = (prof + 1) / (bgfit + 1) - 1

        return X, prof

    def process_profiles(self, infos):
        infos = dp.process_profiles(
            infos, self.metadata, self.settings, self.outpath)
        return infos
