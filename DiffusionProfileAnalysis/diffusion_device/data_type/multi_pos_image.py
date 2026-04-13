# -*- coding: utf-8 -*-
"""
Analyse 4 position channels

Created on Tue Apr  4 11:21:01 2017

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
from tifffile import imread
from registrator.image import is_overexposed
import tifffile
import background_rm as rmbg
import registrator.image as ir

from .images_files import ImagesFile
from .multi_pos_scan import MultiPosScan
from .. import profile as dp


class MultiPosImage(MultiPosScan, ImagesFile):

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
        data = self.load_image(filename)
        if len(np.shape(data)) > 2:
            raise RuntimeError("Too many dimentions for single image data.")
        return data

    def process_data(self, raw_data):
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
        overexposed = is_overexposed(raw_data)
        data, background = self.process_background(raw_data)
        infos = {}
        data, infos = self.process_image(data, background, infos)
        infos['Data'] = data
        infos["Overexposed"] = overexposed
        infos["noise_var"] = self.get_noise_var(raw_data, infos)
        return infos

    def get_noise_var(self, raw_data, infos):
        """ Get the noise corresponding to the image"""
        var, *_ = self.orientate90(raw_data, self.metadata["KEY_MD_FLOWDIR"])
        var = self.rotate_image(var, -infos["image_angle"])

        if "image_intensity" in infos:
            var /= infos["image_intensity"]**2

        if not self.settings["KEY_STG_IMAGE_COORD"]:
            if "offset" in infos:
                if "diffAngle" in infos:
                    var = ir.rotate_scale_shift(
                            var,
                            -infos["diffAngle"],
                            1 / infos["diffScale"],
                            -infos["offset"],
                            borderValue=np.nan)
                else:
                    var = ir.shift_image(
                            var, -infos["offset"], borderValue=np.nan)

        if "image_intensity_reflatten" in infos:
            var /= infos["image_intensity_reflatten"][0]**2

        var[np.isnan(infos["Data"])] = np.nan
        noise_var = self.get_multi_pos_scan(var, infos['Pixel size'])
        noise_var /= np.sum(np.isfinite(var), -2)
        return noise_var

    def get_multi_pos_scan(self, data, pixel_size):
        """get_multi_pos_scan"""
        imslice = self.settings["KEY_STG_SLICE"]
        if imslice is None:
            lin_profiles = np.nanmean(data, -2)
        else:
            lin_profiles = self.imageProfileSlice(
                data, imslice[0], imslice[1], pixel_size)
        return lin_profiles

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
        data = infos['Data']
        lin_profiles = self.get_multi_pos_scan(
                data, infos['Pixel size'])
        # Replace data to use super
        infos['Data'] = lin_profiles
        infos = super().get_profiles(infos)
        infos['Data'] = data
        return infos

    def savedata(self, infos):
        """Save the data"""
        tifffile.imsave(self.outpath + '_im.tif', infos['Data'])

    def process_image(self, image, background, infos):
        """
        Get the hydrodynamic radius from the images

        Parameters
        ----------
        image: 2d image
            image to analyse
        background: path or image
            The background to remove
        metadata: dict
            The metadata
        settings: dict
            The settings

        Returns
        -------
        r: float
            Radius in [m]

        """

        # Check images is numpy array
        image = np.asarray(image)

        # Check shape
        if len(np.shape(image)) not in [2, 3]:
            raise RuntimeError("Incorrect image shape: " +
                               str(np.shape(image)))

        if background is not None:
            background = np.asarray(background)
            # load background if string
            if background.dtype.type == np.str_:
                background = imread(str(background))

            # Check shape
            if not len(np.shape(background)) == 2:
                raise RuntimeError("Incorrect background shape: "
                                   + str(np.shape(background)))
        background, *_ = self.orientate90(background,
                                         self.metadata["KEY_MD_FLOWDIR"])
        image, flowdir, inletpos = self.orientate90(
                image,
                self.metadata["KEY_MD_FLOWDIR"],
                self.metadata["KEY_MD_INLET_LOC"])
        infos["flow direction"] = flowdir
        infos['Inlet location'] = inletpos
        # get profiles
        if background is None:
            # Single image
            image, infos = self.nobg_extract_data(image, infos)
        else:
            # images and background
            image, infos = self.bg_extract_data(
                image, background, infos)

        return image, infos

    def orientate90(self, image, flowdir, inlet_pos=None):
        """ Rotate 2d or 3d image
        """
        if image is None:
            return None, None
        flowdir = np.asarray(flowdir)
        if flowdir[0] in ['l', 'r']:
            image = np.rot90(image, axes=(-2, -1))
            if inlet_pos is not None:
                if inlet_pos == 'top':
                    inlet_pos = 'left'
                elif inlet_pos == 'bottom':
                    inlet_pos = 'right'
                else:
                    raise RuntimeError(
                        'Inlet position incompatible with flow direction!'
                        f' {inlet_pos} and {flowdir}')
            flowdir[flowdir == 'l'] = 'd'
            flowdir[flowdir == 'r'] = 'u'

        if inlet_pos is not None:
            if inlet_pos not in ['right', 'left']:
                raise RuntimeError(
                    'Inlet position incompatible with flow direction!'
                    f' {inlet_pos} and {flowdir}')
        return image, flowdir, inlet_pos

    def imageProfileSlice(self, image, center, width, pixel_size):
        '''Get the image profile corresponding to a center and width

        Parameters
        ----------
        image: 2d array
            The flat image
        center: float
            Y center of the slice [m]
        width: float
            Y width of the slice [m]
        pixel_size: float
            Size of the pixel [m]
        Returns
        -------
        prof: array of float
            The profile corresponding to the slice

        '''
        center = np.shape(image)[-2] // 2 + int(np.round(center / pixel_size))
        width = int(np.round(width / pixel_size))
        amin = (2 * center - width) // 2
        amax = (2 * center + width) // 2
        if amin < 0 or amax > np.shape(image)[-2]:
            raise RuntimeError("Poorly defined slice")
        return np.nanmean(image[..., amin:amax, :], -2)

    def image_infos(self, image):
        """
        Get the image angle, channel width, proteind offset, and origin

        Parameters
        ----------
        image: 2d array
            The image
        number_profiles: integer
            the numbers of channels

        Returns
        -------
        dict: dictionnary
            dictionnary containing infos

        """
        # Detect Angle
        angle = dp.image_angle(image)
        image = self.rotate_image(image, -angle)
        # Get channels infos
        centers, pixel_size = self.straight_image_infos(image)

        retdict = {
            'angle': angle,
            'centers': centers,
            'pixel_size': pixel_size}
        return retdict

    def straight_image_infos(self, image):
        """
        Get the channel width, proteind offset, and origin from a
        straight image

        Parameters
        ----------
        image: 2d array
            The image
        number_profiles: integer
            the numbers of channels

        Returns
        -------
        w: float
            Channel width in pixels
        a: float
            offset of the proteins in the channel
        origin: float
            Position of the first channel center

        """
        # Get profile
        profiles = np.nanmean(image - np.nanmedian(image), 0)

        return self.get_scan_centers(profiles)

    def flat_image(self, image, rep_image, infos, *, frac=.6, subtract=False):
        """
        Flatten input images

        Parameters
        ----------
        image: 2d array
            The image
        chwidth: float
            The channel width in [m]
        wallwidth: float
            The wall width in [m]
        number_profiles: integer
            the numbers of channels
        frac: float
            fraction of the profile taken by fluorescence from channels
        infosOut: dict, defaults None
            dictionnary containing the return value of straight_image_infos
        subtract: Bool
            Should the shape be subtracted instead of divided

        Returns
        -------
        image: 2d array
            The flattened image

        """
        channel_width = self.metadata["KEY_MD_WY"]
        number_profiles = self.metadata["KEY_MD_NCHANNELS"]
        pixel_size = infos["Pixel size"]
        centers = infos["Centers"]

        # get mask
        mask = np.ones(np.shape(rep_image)[-1])
        for i in range(number_profiles):
            amin = centers[i] - frac * channel_width / pixel_size
            amax = centers[i] + frac * channel_width / pixel_size

            if amin < 0:
                amin = centers[i] - .5 * channel_width / pixel_size
                if amin < 0:
                    amin = 0

            if amax > len(mask):
                amax = centers[i] + .5 * channel_width / pixel_size
                if amax > len(mask):
                    amax = len(mask)

            mask[int(amin):int(amax)] = 0
        mask = mask > 0
        mask = np.tile(mask[None, :], (np.shape(rep_image)[0], 1))

        if np.nanmin(image) < 0:
            image -= np.nanmin(image)

        fitted_image = rmbg.polyfit2d(image, mask=mask)

        # Flatten
        if not subtract:
            infos["image_intensity"] = fitted_image
            image = image / fitted_image - 1
        else:
            image = image - fitted_image

        return image, infos

    def nobg_extract_data(self, image, infos, subtract=False):
        '''
        Extract profiles from image

        Parameters
        ----------
        image: 2d or 3d array
            The flat image
        number_profiles: integer
            the numbers of channels
        chwidth: float
            The channel width in [m]
        wallwidth: float
            The wall width in [m]
        flatten: Bool, Defaults False
            Should the image be flatten

        Returns
        -------
        profiles: 2d array
            The four profiles
        '''
        image = np.asarray(image)

        # Get a representative image of the stack (or the image itself)
        rep_image = self.best_image(image)

        # Detect Angle
        angle = dp.image_angle(rep_image)
        image = self.rotate_image(image, -angle)
        rep_image = self.rotate_image(rep_image, -angle)
        infos["image_angle"] = angle

        # Get channels infos
        centers, pixel_size = self.straight_image_infos(rep_image)
        infos["Pixel size"] = pixel_size
        infos["Centers"] = centers

        if self.settings["KEY_STG_BRIGHT_FLAT"]:
            image, infos = self.flat_image(
                    image, rep_image, infos, subtract=subtract)

        return image, infos

    def best_image(self, images):
        if len(np.shape(images)) == 2:
            return images
        return images[np.argmax(np.nanpercentile(images, 99, axis=(-2, -1)))]

    def remove_curve_background_alt(self, im, bg, infos,
                                    maskim=None, maskbg=None, reflatten=False,
                                    image_coord=False):
        """
        Try to flatten without good features :/
        """
        im = np.asarray(im, dtype='float32')
        bg = np.asarray(bg, dtype='float32')

        if maskim is None:
            if len(np.shape(im)) == 2:
                maskim = rmbg.backgroundMask(im)
            else:
                maskim = rmbg.backgroundMask(im[np.argmax(im)])
        if maskbg is None:
            maskbg = rmbg.backgroundMask(bg, nstd=6)

        # Flatten the image and background
        fim = rmbg.polyfit2d(im, 2, mask=maskim)
        fbg = rmbg.polyfit2d(bg, 2, mask=maskbg)

        if np.any(fim <= 0):
            raise RuntimeError("Image mask too small")

        if np.any(fbg <= 0):
            raise RuntimeError("Background mask too small")

        infos['image_intensity'] = fim

        im = im / fim
        bg = bg / fbg

        bg_cpy = np.copy(bg)
        bg_cpy[rmbg.signalMask(bg)] = np.nan

        pbg = np.nanmean(bg_cpy, 0) - 1
        pbg[np.isnan(pbg)] = 0

        squeeze = False
        if len(np.shape(im)) == 2:
            squeeze = True
            im = im[np.newaxis]

        data = np.zeros_like(im)
        offsets = np.zeros((len(im), 2))

        for i, image in enumerate(im):
            image_copy = np.copy(image)
            image_copy[rmbg.signalMask(image)] = np.nan
            pim = np.nanmean(image_copy, 0) - 1
            pim = np.diff(pim)
            pim[np.isnan(pim)] = 0
            cnv = np.correlate(np.abs(pim), np.abs(np.diff(pbg)), mode='full')
            shift = len(pim) - np.argmax(cnv) - 1
            offset = np.array([0, -shift])
            if image_coord:
                data[i] = image - ir.shift_image(
                        bg, offset, borderValue=np.nan)
            else:
                data[i] = ir.shift_image(
                        image, -offset, borderValue=np.nan) - bg
            offsets[i] = offset

        if reflatten:
            data += 1
            fdata = rmbg.polyfit2d(data, 2, mask=maskbg)
            infos['image_intensity_reflatten'] = fdata
            data /= fdata
            data -= 1

        if squeeze:
            data = np.squeeze(data)
            offsets = np.squeeze(offsets)

        infos['offset'] = offsets
        return data, infos

    def remove_bg(self, im, bg, infos, centersOut=None):
        """
        Flatten and background subtract images

        Parameters
        ----------
        im:  2d/3d array
            list of images containning the 4 channels
        bg: 2d array
            Background corresponding to the list
        chwidth: float
            The channel width in [m]
        wallwidth: float
            The wall width in [m]
        Nprofs: integer
            the numbers of channels
        edgesOut: 1d array
            output for the edges

        Returns
        -------
        flatIm: 2d array
            Flattened image

        """
        # Get settings
        goodFeatures = self.settings["KEY_STG_GOODFEATURES"]
        image_coord = self.settings["KEY_STG_IMAGE_COORD"]
        align_background = self.settings["KEY_ALIGN_BACKGROUND"]
        channel_width = self.metadata["KEY_MD_WY"]

        # Get brightest image if stack
        if len(np.shape(im)) == 3:
            im_tmp = im[np.argmax(np.nanmean(im, axis=(1, 2)))]
        else:
            im_tmp = im

        # Get first flattened image
        if not align_background:
            data_tmp = im_tmp - bg
            infos['offset'] = np.zeros(2)
            infos['diffAngle'] = 0
            infos['diffScale'] = 1
        elif goodFeatures:
            data_tmp = rmbg.remove_curve_background(
                im_tmp, bg, infoDict=infos, bgCoord=not image_coord)
        else:
            data_tmp, infos = self.remove_curve_background_alt(
                im_tmp, bg, infos, image_coord=image_coord)

        # Get angle
        angle = dp.image_angle(data_tmp)
        infos["image_angle"] = angle

        # rotate
        bg = self.rotate_image(bg, -angle)
        im = self.rotate_image(im, -angle)
        data_tmp = self.rotate_image(data_tmp, -angle)

        # Get current centers
        bright_infos = self.image_infos(data_tmp)
        pixel_size = bright_infos['pixel_size']
        centers = bright_infos['centers']

        # Get data mask
        mask_data = np.ones(data_tmp.shape)
        X = np.arange(np.shape(data_tmp)[1])
        for c in centers:
            mask_data[:, np.abs(X - c) < channel_width / pixel_size/2] = 0

        if image_coord:
            mask_im = mask_data
            offset = -infos['offset']
            if goodFeatures:
                c, s = np.cos(angle), np.sin(angle)
                R = np.array(((c, -s), (s, c)))
                offset = R @ offset
                mask_bg = ir.rotate_scale_shift(
                    mask_im, -infos['diffAngle'],
                    1/infos['diffScale'], offset,
                    borderValue=0) > .5
            else:
                mask_bg = ir.shift_image(mask_im, offset,
                                         borderValue=0) > .5
        else:
            mask_bg = mask_data
            offset = infos['offset']
            if goodFeatures:
                c, s = np.cos(angle), np.sin(angle)
                R = np.array(((c, -s), (s, c)))
                offset = R @ offset
                mask_im = ir.rotate_scale_shift(
                    mask_bg, infos['diffAngle'],
                    infos['diffScale'], offset,
                    borderValue=0) > .5
            else:
                mask_im = ir.shift_image(mask_bg, offset,
                                         borderValue=0) > .5
            # give infos to remove_curve_background to save transformations

        if not align_background:
            ret = im - bg
        elif goodFeatures:
            # Get Intensity
            ret = rmbg.remove_curve_background(
                    im, bg, maskbg=mask_bg, maskim=mask_im,
                    bgCoord=not image_coord, reflatten=True,
                    infoDict=infos)
        else:
            ret, infos = self.remove_curve_background_alt(
                im, bg, infos, mask_im, mask_bg, reflatten=True,
                image_coord=image_coord)

        if centersOut is not None:
            im = ret
            if len(np.shape(im)) == 3:
                im = im[np.argmax(np.nanmean(im, axis=(1, 2)))]
            centersOut[:] = self.image_infos(im)['centers']
        return ret

    def bg_extract_data(self, im, bg, infos):
        """
        Extract diffusion profiles


        Parameters
        ----------
        im:  2d array
            image containning the 4 channels
        bg: 2d array
            Background image
        Nprofs: integer
            the numbers of channels
        chwidth: float
            The channel width in [m]
        wallwidth: float
            The wall width in [m]

        Returns
        -------
        profiles: 2d array
            list of profiles
        """
        channel_width = self.metadata["KEY_MD_WY"]
        nchannels = self.metadata["KEY_MD_NCHANNELS"]
        wall_width = self.metadata["KEY_MD_WALLWIDTH"]
        # get edges
        centers = np.empty(nchannels, dtype=int)
        # Get flattened image
        flat_im = self.remove_bg(im, bg, infos, centersOut=centers)
        # Get channel width
        pixel_size = (channel_width + wall_width) / np.mean(np.diff(centers))

        infos["Pixel size"] = pixel_size
        infos["Centers"] = centers

        return flat_im, infos
