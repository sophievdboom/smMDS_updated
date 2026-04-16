'''
This script allows you to choose a few ptu files to analyse with a bursts search.
At the end of the script you can alter the burst search settings to find correct ones.
It will generate png files with the inter-photon times and burst lengths.

'''

import os
import tkinter as tk
from tkinter import filedialog

from get_bursts_tttrlib import get_bursts
from burst_analysis_utils import (
    extract_position_um,
    build_user_setting
)


def get_ptufilenames():
    """
    Open a file dialog and let the user select multiple .ptu files.
    """
    root = tk.Tk()
    root.withdraw()
    filenames = filedialog.askopenfilenames(
        title="Select PTU files for burst-search testing",
        filetypes=(("ptu files", "*.ptu"), ("all files", "*.*"))
    )
    root.update()
    root.destroy()
    return list(filenames)


def save_summary_txt(output_folder, summary_rows):
    filepath = os.path.join(output_folder, "selected_steps_summary.txt")
    with open(filepath, "w") as f:
        f.write("filename\tposition_um\tnumber_of_bursts\tmean_intensity\tmedian_intensity\tmax_intensity\ttotal_photons\n")
        for row in summary_rows:
            f.write(
                f"{row['filename']}\t"
                f"{row['position_um']:.6f}\t"
                f"{row['number_of_bursts']:.0f}\t"
                f"{row['mean_intensity']:.6f}\t"
                f"{row['median_intensity']:.6f}\t"
                f"{row['max_intensity']:.6f}\t"
                f"{row['total_photons']:.0f}\n"
            )


def analyse_selected_files(selected_files, base_user_setting):
    if not selected_files:
        raise RuntimeError("No PTU files selected.")

    # Use the folder of the first selected file
    base_folder = os.path.dirname(selected_files[0])
    output_folder = os.path.join(base_folder, "out_selected_steps")
    os.makedirs(output_folder, exist_ok=True)

    user_setting = build_user_setting(base_user_setting, output_folder)

    summary_rows = []

    total_files = len(selected_files)
    print(f"Selected {total_files} PTU files.\n")

    for i, ptufilename in enumerate(selected_files, start=1):
        base_name = os.path.splitext(os.path.basename(ptufilename))[0]

        try:
            position_um = extract_position_um(ptufilename)
        except ValueError:
            position_um = float("nan")

        print(f"[{i}/{total_files}] Processing: {base_name}")
        if position_um == position_um:
            print(f"    Position: {position_um:.2f} um")

        df_photons, df_bursts = get_bursts(ptufilename, user_setting=user_setting)

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

        total_photons = float(df_photons.attrs.get("total_photons_in_file", 0))

        bursts_folder = os.path.join(output_folder, "bursts_csv")
        os.makedirs(bursts_folder, exist_ok=True)

        bursts_csv = os.path.join(
            bursts_folder,
            f"{base_name}_bursts_tttrlib.csv"
        )
        df_bursts.to_csv(bursts_csv, index=False)

        summary_rows.append({
            "filename": base_name,
            "position_um": position_um,
            "number_of_bursts": number_of_bursts,
            "mean_intensity": mean_intensity,
            "median_intensity": median_intensity,
            "max_intensity": max_intensity,
            "total_photons": total_photons,
        })

        print(f"    Bursts: {number_of_bursts:.0f}")
        print(f"    Saved bursts CSV: {bursts_csv}")
        print()

    save_summary_txt(output_folder, summary_rows)

    print("Done.")
    print(f"Results saved in: {output_folder}")
    print(f"Summary file: {os.path.join(output_folder, 'selected_steps_summary.txt')}")


if __name__ == "__main__":
    base_user_setting = {
        "set_lee_filter": 2,
        "threshold_iT_signal": 25e-06, # interphoton time threshold in ms
        "threshold_iT_lower": 0.1e-6,
        "threshold_iT_noise": 0.1,
        "min_phs_burst": 10e3,
        "min_phs_noise": 1600,
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

    selected_files = get_ptufilenames()
    analyse_selected_files(selected_files, base_user_setting)