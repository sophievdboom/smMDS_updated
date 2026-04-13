# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 12:27:04 2017

@author: quentinpeter
"""

from diffusion_device import process_data
from glob import glob
import matplotlib.pyplot as plt

# Where should the results be saved
outpath = './MDS_confocal_try_1/output'

# json files location (use glob!)
settingsfn = './MDS_confocal_try_1\settings.json'
metadatafn = './MDS_confocal_try_1\PhotonData_1_metadata.json'

settings = glob(settingsfn)
if len(settings) == 0:
    raise RuntimeError(f"Can't find {settingsfn}")
metadatas = glob(metadatafn)
if len(metadatas) == 0:
    raise RuntimeError(f"Can't find {metadatafn}")

for sfn in settings:
    for mfn in metadatas:
        print('Processing new file:')
        print('Metadata file: ', mfn)
        print('Settings file: ', sfn)
        # Call function
        process_data.full_fit(sfn, mfn, outpath)
        try:
            plt.show(blocking=False)
        except TypeError:
            plt.show()