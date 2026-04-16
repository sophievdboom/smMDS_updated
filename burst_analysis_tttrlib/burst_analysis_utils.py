import os
import re
import glob
import copy
import tkinter as tk
from tkinter import filedialog

CHANNEL_MODE_CONFIGS = {
    "sum": [("", "sum")],
    "ch1": [("_ch1", "ch1")],
    "ch2": [("_ch2", "ch2")],
    "separate": [("_ch1", "ch1"), ("_ch2", "ch2")],
}

def extract_position_um(filename):
    match = re.search(r'([0-9]{1,6}(?:\.[0-9]{0,2})?)um_', filename)
    if match is None:
        raise ValueError(f"Could not extract position from filename: {filename}")
    return float(match.group(1))


def build_user_setting(base_user_setting, output_folder):
    user_setting = copy.deepcopy(base_user_setting)
    user_setting["output_folder"] = output_folder
    return user_setting

def save_inputparameter_txt(output_folder, user_setting, suffix=""):
    filepath = os.path.join(output_folder, f"Analysed_data_inputparameter{suffix}.txt")
    with open(filepath, "w") as f:
        f.write(f"{user_setting['set_lee_filter']}\n")
        f.write(f"{user_setting['threshold_iT_signal']}\n")
        f.write(f"{user_setting['threshold_iT_noise']}\n")
        f.write(f"{user_setting['min_phs_burst']}\n")
        f.write(f"{user_setting['min_phs_noise']}\n")
        f.write(f"{user_setting['filter_name']}\n")
        f.write(f"{user_setting['show_plot']}\n")
        f.write(f"{user_setting['output_folder']}\n")
        f.write(f"{user_setting['burst_channel_mode']}\n")

def save_positions_txt(output_folder, positions, suffix=""):
    filepath = os.path.join(output_folder, f"Analysed_data_positions{suffix}.txt")
    with open(filepath, "w") as f:
        for value in positions:
            f.write(f"{value:.18e}\n")



def save_numb_molecules_txt(output_folder, numb_molecules, suffix=""):
    filepath = os.path.join(output_folder, f"Analysed_data_numb_molecules{suffix}.txt")
    with open(filepath, "w") as f:
        for value in numb_molecules:
            f.write(f"{value:.18e}\n")



def save_sumphoton_txt(output_folder, sum_photons, suffix=""):
    filepath = os.path.join(output_folder, f"Analysed_data_SumPhoton{suffix}.txt")
    with open(filepath, "w") as f:
        for value in sum_photons:
            f.write(f"{value:.18e}\n")



def save_alldata_txt(output_folder, alldata_rows, suffix=""):
    filepath = os.path.join(output_folder, f"Alldata{suffix}.txt")
    with open(filepath, "w") as f:
        for row in alldata_rows:
            row_str = ",".join(f"{v:.18e}" for v in row)
            f.write(row_str + "\n")



def initialise_results_dict():
    return {
        suffix: {
            "positions": [],
            "numb_molecules": [],
            "sum_photons": [],
            "alldata_rows": [],
            "mode": mode,
        }
        for suffix, mode in CHANNEL_MODE_CONFIGS["sum"]
    }



def build_mode_results(channel_output_mode):
    if channel_output_mode not in CHANNEL_MODE_CONFIGS:
        raise ValueError(
            "channel_output_mode must be one of: 'sum', 'ch1', 'ch2', 'separate'."
        )
    return {
        suffix: {
            "positions": [],
            "numb_molecules": [],
            "sum_photons": [],
            "alldata_rows": [],
            "mode": mode,
        }
        for suffix, mode in CHANNEL_MODE_CONFIGS[channel_output_mode]
    }