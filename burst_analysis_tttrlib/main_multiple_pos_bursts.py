import os
import re
import glob
import copy
import tkinter as tk
from tkinter import filedialog

from get_bursts_tttrlib import get_bursts


CHANNEL_MODE_CONFIGS = {
    "sum": [("", "sum")],
    "ch1": [("_ch1", "ch1")],
    "ch2": [("_ch2", "ch2")],
    "separate": [("_ch1", "ch1"), ("_ch2", "ch2")],
}


def get_ptufilename():
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(
        filetypes=(("ptu files", "*.ptu"), ("all files", "*.*"))
    )
    root.update()
    root.destroy()
    return filename


def extract_position_um(filename):
    match = re.search(r'([0-9]{1,6}(?:\.[0-9]{0,2})?)um_', filename)
    if match is None:
        raise ValueError(f"Could not extract position from filename: {filename}")
    return float(match.group(1))


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



def build_user_setting(base_user_setting, output_folder):
    user_setting = copy.deepcopy(base_user_setting)
    user_setting["output_folder"] = output_folder
    return user_setting



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
            bursts_csv = os.path.join(output_folder, f"{base_name}_bursts_tttrlib{suffix}.csv")
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
        "threshold_iT_signal": 0.05,
        "threshold_iT_noise": 0.1,
        "min_phs_burst": 10,
        "min_phs_noise": 160,
        "filter_name": "addLeefilter",
        "show_plot": False,
        "output_folder": "out",
        "use_noise_regions": False,
        "tttr_mode": "T2",
        "allowed_routing_channels": None,
        "pie_microtime_gate": None,
        "diff_chunk_size": 5_000_000,
        "debug_photons_n": 0,
        "channel_output_mode": "ch1",   # 'sum', 'ch1', 'ch2', or 'separate'
        "burst_channel_mode": "ch1",
    }

    selected_file = get_ptufilename()
    if not selected_file:
        raise RuntimeError("No file selected.")

    analyse_folder(selected_file, base_user_setting)
