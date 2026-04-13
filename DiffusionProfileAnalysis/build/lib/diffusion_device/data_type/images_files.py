# -*- coding: utf-8 -*-
"""
Useful functions to work on images

Created on Wed Sep 13 07:30:56 2017

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
import registrator.image as ir
from . import DataType


class ImagesFile(DataType):

    def __init__(self, metadata, settings, outpath):
        super().__init__(metadata, settings, outpath)
        self.files = {}

    def process_background(self, data):
        """Remove optical background from data and background, and clip images.

        Parameters
        ----------
        data: array of floats
            The data to process
        metadata: dict
            The metadata informations

        Returns
        -------
        data: array
            data
        backgrounds: array of float
            backgrounds if there is any

        """
        data = self.remove_optics_background(data)
        background = self.remove_optics_background(self.get_background())
        return data, background

    def load_images(self, filename):
        """ Load image or list of images

        Parameters
        ----------
        filename: dict
            The image filename

        Returns
        -------
        ims: array
            image
        """

        # load data
        if isinstance(filename, (list, tuple)):
            data = np.asarray([self.load_image(fn) for fn in filename])
        else:
            data = self.load_image(filename)
        return data

    def load_image(self, fn):
        """ Load single image

        Parameters
        ----------
        filename: dict
            The image filename

        Returns
        -------
        im: array
            image
        """
        if fn in self.files:
            return self.files[fn]

        data = imread(fn)
        if len(data.shape) == 3:
            data = np.squeeze(data[np.logical_not(np.all(data == 0, (1, 2)))])
        data = self.clip_border(data)
        self.files[fn] = data
        return data

    def get_background(self):
        """Load background

        Parameters
        ----------
        metadata: dict
            The metadata informations

        Returns
        -------
        backgrounds: array of float
            backgrounds if there is any

        """
        backgrounds = None
        background_fn = self.metadata["KEY_MD_BGFN"]
        if background_fn is not None:
            backgrounds = self.load_images(background_fn)
        return backgrounds

    def remove_optics_background(self, images):
        """Remove optical background from data and background

        Parameters
        ----------
        images: array of floats
            The data to process
        backgrounds: array of floats
            The data to process
        metadata: dict
            The metadata informations

        Returns
        -------
        images: array
            images
        backgrounds: array of float
            backgrounds if there is any

        """
        if images is None:
            return None

        images = np.asarray(images, "float32")
        # Remove background from optics
        optics_bgfn = self.metadata["KEY_MD_OPBGFN"]
        if optics_bgfn is None:
            return images

        optics = np.asarray(self.load_images(optics_bgfn), "float32")

        factor = np.sum(optics**2) / np.sum(optics * images, axis=(-2, -1))

        if len(np.shape(factor)) == 1:
            factor = factor[:, np.newaxis, np.newaxis]

        if self.metadata["KEY_MD_OPTIC_SUBTRACT"]:
            images = images - (optics - np.median(optics)) / factor
        else:
            images /= optics
        return images

    def clip_border(self, images):
        """Remove border from data and background

        Parameters
        ----------
        images: array of floats
            The data to process
        backgrounds: array of floats
            The data to process
        metadata: dict
            The metadata informations

        Returns
        -------
        images: array
            images
        backgrounds: array of float
            backgrounds if there is any

        """
        if images is None:
            return None
        imborder = self.metadata["KEY_MD_BORDER"]
        if imborder is not None:
            # Remove Border
            images = images[..., imborder[0]:imborder[1],
                            imborder[2]:imborder[3]]
        return images

    def rotate_image(self, im, angle, borderValue=np.nan):
        """Rotate the image or stack of images
        """
        if len(np.shape(im)) == 3:
            for image in im:
                image[:] = ir.rotate_scale(
                    image, angle, 1, borderValue=borderValue)
        else:
            im = ir.rotate_scale(im, angle, 1, borderValue=borderValue)
        return im
