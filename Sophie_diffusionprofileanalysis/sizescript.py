# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 12:27:04 2017

@author: quentinpeter
"""

from diffusion_device import process_data
from glob import glob
import matplotlib.pyplot as plt

# Where should the results be saved

data_folder = './Test_Data/2026-04-10_GOOD_Atto488_T2_smMDS/conc-1nM_laser-485nm_640nm_Mode-T2_ch-2_T-23_buffer-1xsb_fr-100mlph_power-con100_chipname-master3.3_fabdate-200226_h-28.1__StepScan_stepnumber-200_time-2s/3/'

outpath = data_folder + 'results'
# json files location (use glob!)
settingsfn = data_folder + 'out/settings.json'
metadatafn = data_folder + 'out/Alldata_metadata.json'

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
