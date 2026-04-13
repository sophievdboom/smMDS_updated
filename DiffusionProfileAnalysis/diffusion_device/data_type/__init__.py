# -*- coding: utf-8 -*-
"""
Folder of modules defining:

data = load_data(metadata, infos)
data = process_data(data, metadata, settings, infos)
profiles = get_profiles(data, metadata, settings, infos)
radius, fits = size_profiles(profiles, metadata, settings, infos)
plot_and_save(radius, profiles, fits, outpath, settings, infos)
savedata(data, outpath)

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
from .. import profile as dp
from .. import display_data


class DataType():
    def __init__(self, metadata, settings, outpath):
        """Init"""
        super().__init__()
        self.metadata = metadata
        self.settings = settings
        self.outpath = outpath

    def full_fit(self):
        """Perform a fit with the imformations found in the settings file"""
        raw_data = self.load_data()
        infos = self.process_data(raw_data)

        if self.outpath is not None:
            self.savedata(infos)

        infos = self.get_profiles(infos)
        self.plot_wide_profiles(infos)
        try:
            infos = self.process_profiles(infos)
            infos = self.size_profiles(infos)
        except RuntimeError:
            import matplotlib.pyplot as plt
            import numpy as np
            plt.figure()
            plt.plot(np.ravel(infos["Profiles"]))
            plt.xlabel("Diffusion direction --->")
            raise

        if self.outpath is not None:
            infos = self.plot_and_save(infos)

        return infos

    def load_data(self):
        """load data from metadata

        Returns
        -------
        data: array
            the image
        """
        raise NotImplementedError

    def process_data(self, data):
        """Do some data processing

        Parameters
        ----------
        data: array
            The data to process

        Returns
        -------
        data: array
            The processed data
        """
        infos = {}
        infos['Data'] = data
        return infos

    def savedata(self, infos):
        """Save the data"""
        pass

    def get_profiles(self, infos):
        """Do some data processing

        Parameters
        ----------
        data: array
            The data to process

        Returns
        -------
        profiles: array
            The profiles
        """
        raise NotImplementedError

    def process_profiles(self, infos):
        """process_profiles"""
        return infos

    def size_profiles(self, infos):
        """size_profiles"""
        return dp.size_profiles(
            infos, self.metadata, self.settings)

    def plot_and_save(self, infos):
        """plot_and_save"""
        return display_data.plot_and_save(infos, self.settings, self.outpath)

    def plot_wide_profiles(self, infos):
        """Print the profiles."""
        display_data.plot_wide_profiles(infos, self.metadata,
                                        self.settings, self.outpath,
                                        new_figure=True)
