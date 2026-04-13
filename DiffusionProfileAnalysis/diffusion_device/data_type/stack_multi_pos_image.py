# -*- coding: utf-8 -*-
"""
Module for movies of 4 pos channels

Created on Wed Sep 13 10:15:41 2017

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

from registrator.image import is_overexposed
import numpy as np
import tifffile
import sys
import pandas as pd
import os
import shutil

from .multi_pos_image import MultiPosImage
from .. import display_data
from . import images_files
from .. import profile as dp


class StackMultiPosImage(MultiPosImage):

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
        Nchannel = self.metadata['KEY_MD_NCHANNELS']
        framesslices = slice(*self.settings["KEY_STG_STACK_FRAMESSLICES"])
        rebin = self.settings["KEY_STG_STACK_REBIN"]
        index = np.arange(len(data))[framesslices]
        index = index[:(len(index)//rebin)*rebin:rebin]

        overexposed = [is_overexposed(d) for d in data[framesslices]]
        data = np.asarray(data[framesslices], dtype="float32")

        if rebin > 1:
            new_data = np.zeros(
                (np.shape(data)[0]//rebin, *np.shape(data)[1:]))
            for i in range(len(new_data)):
                new_data[i] = np.mean(data[i * rebin:(i + 1) * rebin], 0)
            data = new_data

            new_overexposed = np.zeros(len(overexposed)//rebin, bool)
            for i in range(len(new_overexposed)):
                new_overexposed[i] = np.any(
                    overexposed[i * rebin:(i + 1) * rebin])

            overexposed = new_overexposed
            for i, val in enumerate(self.settings["KEY_STG_STACK_FRAMESSLICES"]):
                if val is not None:
                    self.settings["KEY_STG_STACK_FRAMESSLICES"][i] = val // rebin
            framesslices = slice(*self.settings["KEY_STG_STACK_FRAMESSLICES"])

        infos = pd.DataFrame(index=index)
        infos.loc[:, "Overexposed"] = overexposed

        if self.settings["KEY_STG_STAT_STACK"]:
            # Check KEY_MD_EXP are all the same
            if isinstance(self.metadata["KEY_MD_EXP"], list):
                if np.all(np.equal(self.metadata["KEY_MD_EXP"],
                                   self.metadata["KEY_MD_EXP"][0])):
                    raise RuntimeError(
                        "Can not have different exposure times"
                        " when using stationary option.")
                else:
                    self.metadata["KEY_MD_EXP"] = self.metadata["KEY_MD_EXP"][0]

            super_infos = super().process_data(data)
            super_infos['Error'] = False
            for data_idx, i in enumerate(infos.index):
                infos_i = super_infos.copy()
                key_break =['Data', 'image_intensity',
                            'Overexposed', 'noise_var']
                for key in key_break:
                    infos_i[key] = infos_i[key][data_idx]
                infos = self.add_to_line(infos, i, infos_i)

            return infos

        metadata_stack = self.metadata.copy()
        for i, frame in zip(index, data):
            try:
                # Get KEY_MD_EXP correct in the metadata
                if isinstance(metadata_stack["KEY_MD_EXP"], list):
                    self.metadata["KEY_MD_EXP"] = (
                        np.asarray(metadata_stack["KEY_MD_EXP"])[framesslices][i])
                infos_i = super().process_data(frame)
                infos_i['Error'] = False
                infos = self.add_to_line(infos, i, infos_i)

            except BaseException:
                if self.settings["KEY_STG_IGNORE_ERROR"]:
                    print('Frame', i, sys.exc_info()[1])
                    infos.at[i, 'Error'] = True
                else:
                    raise

        # Fix metadata
        metadata_stack["KEY_MD_FLOWDIR"] = self.metadata["KEY_MD_FLOWDIR"]
        self.metadata = metadata_stack
        return infos

    def add_to_line(self, infos, index, adict):
        intersection = infos.columns.intersection(adict.keys())
        missing = [key for key in adict.keys() if key not in infos.columns]
        adict = pd.DataFrame({k: [adict[k]] for k in adict.keys()},
                              index=[index])
        if len(missing) > 0:
            # add columns
            infos = infos.join(adict.loc[:, missing])
        infos.loc[index, intersection] = adict.loc[index, intersection]
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

        for i in infos.index[np.logical_not(infos.loc[:, 'Error'])]:
            try:
                infos = self.add_to_line(
                        infos, i, super().get_profiles(infos.loc[i].to_dict()))
            except BaseException:
                if self.settings["KEY_STG_IGNORE_ERROR"]:
                    print('Frame', i, sys.exc_info()[1])
                    infos.at[i, 'Error'] = True
                else:
                    raise
        return infos

    def size_profiles(self, infos):
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
        radius:
            if nspecies==1:
                radii: float
                    The best radius fit
            else:
                Rs, spectrum, the radii and corresponding spectrum
        fits: 2d array
            The fits
        """
        for i in infos.index[np.logical_not(infos.loc[:, 'Error'])]:
            try:
                if len(np.shape(self.metadata["KEY_MD_Q"])) > 0:
                    metadata_i = self.metadata.copy()
                    metadata_i["KEY_MD_Q"] = metadata_i["KEY_MD_Q"][i]
                else:
                    metadata_i = self.metadata
                infos_i = dp.size_profiles(
                        infos.loc[i].to_dict(), metadata_i, self.settings)
                infos = self.add_to_line(infos, i, infos_i)
            except BaseException:
                if self.settings["KEY_STG_IGNORE_ERROR"]:
                    print('Frame', i, sys.exc_info()[1])
                    infos.at[i, 'Error'] = True
                else:
                    raise

        return infos

    def savedata(self, infos):
        """Save the data"""
        data = infos.loc[infos.loc[:, 'Data'].notna(), 'Data']
        nchar = int(np.ceil(np.log10(np.max(data.index.to_numpy()))))
        fn = self.outpath + '_image'
        if os.path.exists(fn):
            shutil.rmtree(fn)
        os.makedirs(fn)
        for i in data.index:
            tifffile.imsave(self.outpath + f'_image/{i:0{nchar}d}.tif',
                        np.asarray(data.at[i], 'float32'))

    def plot_and_save(self, infos):
        """Plot the sizing data"""
        return display_data.plot_and_save_stack(
            infos, self.settings, self.outpath)

    def plot_wide_profiles(self, infos):
        """Print the profiles."""
        display_data.plot_wide_profiles_stack(
            infos, self.metadata, self.settings, self.outpath)

    def process_profiles(self, infos):
        for i in infos.index[np.logical_not(infos.loc[:, 'Error'])]:
            infos_i = dp.process_profiles(
                infos.loc[i].to_dict(),
                self.metadata, self.settings, self.outpath)
            infos = self.add_to_line(infos, i, infos_i)

        return infos
