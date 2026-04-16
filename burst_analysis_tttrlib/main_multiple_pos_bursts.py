'''
This is the main script to perform a burst search for step scan data.
At the end of the script you can alter the burst searh settings.
Run the script and a pop up will appear, allowing you to select a ptu file. All ptu files with x.xxum_steps in the name will be analysed.

'''

import os
import glob
import copy
import tkinter as tk
from tkinter import filedialog

from get_bursts_tttrlib import get_bursts

from burst_analysis_utils import (
    extract_position_um,
    build_user_setting,
    save_inputparameter_txt,
    save_positions_txt,
    save_numb_molecules_txt,
    save_sumphoton_txt,
    save_alldata_txt,
    initialise_results_dict,
    build_mode_results
)

def get_ptufilename():
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(
        filetypes=(("ptu files", "*.ptu"), ("all files", "*.*"))
    )
    root.update()
    root.destroy()
    return filename


def analyse_folder(selected_file, base_user_setting):
    folder = os.path.dirname(selected_file)
    filenames = sorted(glob.glob(os.path.join(folder, "*.ptu")))
    if not filenames:
        raise FileNotFoundError(f"No .ptu files found in folder: {folder}")

    output_folder = os.path.join(folder, "out")
    os.makedirs(output_folder, exist_ok=True)

    user_setting = build_user_setting(base_user_setting, output_folder)
    channel_output_mode = str(user_setting.get("channel_output_mode", "sum")).lower()
    mode_results = build_mode_results(channel_output_mode)

    total_files = len(filenames)

    for i, ptufilename in enumerate(filenames, start=1):
        position_um = extract_position_um(ptufilename)
        progress_pct = 100.0 * i / total_files
        print(f"step {position_um:.2f} um, {progress_pct:.0f}%")

        for suffix, result in mode_results.items():
            mode_user_setting = copy.deepcopy(user_setting)
            mode_user_setting["burst_channel_mode"] = result["mode"]

            df_photons, df_bursts = get_bursts(ptufilename, user_setting=mode_user_setting)

            if len(df_bursts) > 0:
                number_of_bursts = float(len(df_bursts))
                mean_intensity = float(df_bursts["Burst intensity"].mean())
                median_intensity = float(df_bursts["Burst intensity"].median())
                max_intensity = float(df_bursts["Burst intensity"].max())
            else:
                number_of_bursts = 0.0
                mean_intensity = 0.0
                median_intensity = 0.0
                max_intensity = 0.0

            total_photons_in_file = float(df_photons.attrs.get("total_photons_in_file", 0))

            alldata_row = [
                position_um,
                number_of_bursts,
                mean_intensity,
                median_intensity,
                max_intensity,
                total_photons_in_file,
            ]

            result["alldata_rows"].append(alldata_row)
            result["positions"].append(position_um)
            result["numb_molecules"].append(number_of_bursts)
            result["sum_photons"].append(total_photons_in_file)

            base_name = os.path.splitext(os.path.basename(ptufilename))[0]
            bursts_folder = os.path.join(output_folder, "bursts_csv")
            os.makedirs(bursts_folder, exist_ok=True)

            # define file path inside that folder
            bursts_csv = os.path.join(
                bursts_folder,
                f"{base_name}_bursts_tttrlib{suffix}.csv"
            )

            # save file
            df_bursts.to_csv(bursts_csv, index=False)

    for suffix, result in mode_results.items():
        combined = list(zip(
            result["positions"],
            result["numb_molecules"],
            result["sum_photons"],
            result["alldata_rows"],
        ))
        combined.sort(key=lambda x: x[0])

        positions = [x[0] for x in combined]
        numb_molecules = [x[1] for x in combined]
        sum_photons = [x[2] for x in combined]
        alldata_rows = [x[3] for x in combined]

        save_alldata_txt(output_folder, alldata_rows, suffix=suffix)
        save_inputparameter_txt(output_folder, {**user_setting, "burst_channel_mode": result["mode"]}, suffix=suffix)
        save_positions_txt(output_folder, positions, suffix=suffix)
        save_numb_molecules_txt(output_folder, numb_molecules, suffix=suffix)
        save_sumphoton_txt(output_folder, sum_photons, suffix=suffix)


if __name__ == "__main__":
    base_user_setting = {
        "set_lee_filter": 2,
        "threshold_iT_signal": 0.05, # interphoton time threshold in ms
        "threshold_iT_lower":  0.4e-6, # lower threshold
        "threshold_iT_noise": 0.1,
        "min_phs_burst": 10,
        "min_phs_noise": 160,
        "filter_name": "addLeefilter",
        "show_plot": True, # saves the interphoton time plots for filter optimalisation, does take a long time
        "output_folder": "out",
        "use_noise_regions": False, # does nothing yet when
        "tttr_mode": "T2",
        "allowed_routing_channels": None,
        "pie_microtime_gate": None,
        "diff_chunk_size": 5_000_000,
        "debug_photons_n": 0, # creates csv files for debugging, set to 0 if no csv files should be saved
        "channel_output_mode": "ch1",   # 'sum', 'ch1', 'ch2', or 'separate'
        "burst_channel_mode": "ch1",
        "plot_max_points": 200000
    }

    selected_file = get_ptufilename()
    if not selected_file:
        raise RuntimeError("No file selected.")

    analyse_folder(selected_file, base_user_setting)
