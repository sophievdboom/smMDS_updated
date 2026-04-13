# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 12:27:04 2017

@author: quentinpeter
"""

from pathlib import Path
from glob import glob
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
from diffusion_device import process_data

# Running the script will allow you to choose the folder containing the data

def choose_data_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select folder containing metadata and settings")
    root.update()
    root.destroy()
    return folder


selected_folder = choose_data_folder()
if not selected_folder:
    raise RuntimeError("No folder selected.")

folder = Path(selected_folder)
out_folder = folder / 'out'
results_folder = folder / 'results'
results_folder.mkdir(parents=True, exist_ok=True)
outpath = str(results_folder)

settings_patterns = [
    str(out_folder / 'settings.json'),
    str(folder / 'settings.json'),
]
metadata_patterns = [
    str(out_folder / '*_metadata.json'),
    str(folder / '*_metadata.json'),
]

settings_files = []
for pattern in settings_patterns:
    settings_files.extend(glob(pattern))
settings_files = sorted(set(settings_files))

metadata_files = []
for pattern in metadata_patterns:
    metadata_files.extend(glob(pattern))
metadata_files = sorted(set(metadata_files))

if len(settings_files) == 0:
    raise RuntimeError(f"Can't find settings.json in {folder} or {out_folder}")
if len(metadata_files) == 0:
    raise RuntimeError(f"Can't find *_metadata.json in {folder} or {out_folder}")

for sfn in settings_files:
    for mfn in metadata_files:
        print('Processing new file:')
        print('Metadata file: ', mfn)
        print('Settings file: ', sfn)
        process_data.full_fit(sfn, mfn, outpath)
        try:
            plt.show(blocking=False)
        except TypeError:
            plt.show()
