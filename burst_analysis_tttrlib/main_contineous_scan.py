import os
import json
import copy
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

from get_bursts_tttrlib import get_bursts


def get_ptufilename():
    """
    Open a file dialog and let the user select one .ptu file.
    """
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(
        filetypes=(("ptu files", "*.ptu"), ("all files", "*.*"))
    )
    root.update()
    root.destroy()
    return filename


def save_inputparameter_txt(output_folder, user_setting):
    filepath = os.path.join(output_folder, "Analysed_data_inputparameter.txt")
    print(f"\nSaving input parameters to: {filepath}")

    with open(filepath, "w") as f:
        f.write(f"{user_setting['set_lee_filter']}\n")
        f.write(f"{user_setting['threshold_iT_signal']}\n")
        f.write(f"{user_setting['threshold_iT_noise']}\n")
        f.write(f"{user_setting['min_phs_burst']}\n")
        f.write(f"{user_setting['min_phs_noise']}\n")
        f.write(f"{user_setting['filter_name']}\n")
        f.write(f"{user_setting['show_plot']}\n")
        f.write(f"{user_setting['output_folder']}\n")


def save_positions_txt(output_folder, positions_um):
    filepath = os.path.join(output_folder, "Analysed_data_positions.txt")
    print(f"Saving positions to: {filepath}")

    with open(filepath, "w") as f:
        for value in positions_um:
            f.write(f"{value:.18e}\n")


def save_numb_molecules_txt(output_folder, numb_molecules):
    filepath = os.path.join(output_folder, "Analysed_data_numb_molecules.txt")
    print(f"Saving number of molecules to: {filepath}")

    with open(filepath, "w") as f:
        for value in numb_molecules:
            f.write(f"{value:.18e}\n")


def save_sumphoton_txt(output_folder, sum_photons):
    filepath = os.path.join(output_folder, "Analysed_data_SumPhoton.txt")
    print(f"Saving sum photons to: {filepath}")

    with open(filepath, "w") as f:
        for value in sum_photons:
            f.write(f"{value:.18e}\n")


def save_alldata_txt(output_folder, alldata_rows):
    filepath = os.path.join(output_folder, "Alldata.txt")
    print(f"Saving Alldata to: {filepath}")

    with open(filepath, "w") as f:
        for row in alldata_rows:
            row_str = ",".join(f"{v:.18e}" for v in row)
            f.write(row_str + "\n")


def build_user_setting(base_user_setting, output_folder):
    user_setting = copy.deepcopy(base_user_setting)
    user_setting["output_folder"] = output_folder

    print("\nFinal user_setting:")
    for key, value in user_setting.items():
        print(f"  {key}: {value}")

    return user_setting


def find_metadata_json(ptufilename):
    """
    Look for metadata.json in the same folder as the PTU file.
    """
    folder = os.path.dirname(ptufilename)
    metadata_path = os.path.join(folder, "metadata.json")

    print(f"\nLooking for metadata.json at: {metadata_path}")

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"metadata.json not found in the PTU folder:\n{metadata_path}"
        )

    return metadata_path


def load_scan_metadata(ptufilename):
    """
    Load metadata.json from the same folder as the PTU file.

    Required:
        speed : scan speed in um/s

    Optional:
        start_um : scan start position in um
        time_offset_s : shift applied before converting time -> position
        n_position_bins : number of output position bins
        photon_hist_chunk_size : chunk size for photon binning
    """
    metadata_path = find_metadata_json(ptufilename)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print("\nLoaded metadata.json:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")

    if "speed" not in metadata:
        raise KeyError("metadata.json must contain key: 'speed'")

    speed_um_per_s = float(metadata["speed"])
    start_um = float(metadata.get("start_um", 0.0))
    time_offset_s = float(metadata.get("time_offset_s", 0.0))
    n_position_bins = int(metadata.get("n_position_bins", 400))
    photon_hist_chunk_size = int(metadata.get("photon_hist_chunk_size", 5_000_000))

    print("\nParsed scan metadata:")
    print(f"  speed_um_per_s       = {speed_um_per_s}")
    print(f"  start_um             = {start_um}")
    print(f"  time_offset_s        = {time_offset_s}")
    print(f"  n_position_bins      = {n_position_bins}")
    print(f"  photon_hist_chunk_size = {photon_hist_chunk_size}")

    return {
        "metadata_path": metadata_path,
        "speed_um_per_s": speed_um_per_s,
        "start_um": start_um,
        "time_offset_s": time_offset_s,
        "n_position_bins": n_position_bins,
        "photon_hist_chunk_size": photon_hist_chunk_size,
        "raw_metadata": metadata,
    }


def times_to_positions_with_speed(
    times_ns,
    speed_um_per_s,
    start_um=0.0,
    time_offset_s=0.0
):
    """
    Convert times to positions using:
        position_um = start_um + speed_um_per_s * (time_s - time_offset_s)
    """
    times_s = np.asarray(times_ns, dtype=np.float64) * 1e-9
    times_s = times_s - time_offset_s
    positions_um = start_um + speed_um_per_s * times_s
    return positions_um


def get_position_range_from_macro_times(
    macro_times_raw_full,
    macro_resolution_s,
    speed_um_per_s,
    start_um=0.0,
    time_offset_s=0.0
):
    """
    Compute min/max scan positions from first and last macro times.
    Works because macro times are monotonic in the event stream.
    """
    if len(macro_times_raw_full) == 0:
        raise ValueError("macro_times_raw_full is empty.")

    first_time_ns = float(macro_times_raw_full[0]) * macro_resolution_s * 1e9
    last_time_ns = float(macro_times_raw_full[-1]) * macro_resolution_s * 1e9

    first_pos = times_to_positions_with_speed(
        np.array([first_time_ns]),
        speed_um_per_s=speed_um_per_s,
        start_um=start_um,
        time_offset_s=time_offset_s,
    )[0]

    last_pos = times_to_positions_with_speed(
        np.array([last_time_ns]),
        speed_um_per_s=speed_um_per_s,
        start_um=start_um,
        time_offset_s=time_offset_s,
    )[0]

    pos_min = float(min(first_pos, last_pos))
    pos_max = float(max(first_pos, last_pos))

    return pos_min, pos_max


def histogram_photon_positions_chunked(
    macro_times_raw_full,
    macro_resolution_s,
    speed_um_per_s,
    start_um,
    time_offset_s,
    bin_edges_um,
    chunk_size=5_000_000,
):
    """
    Histogram photon positions in chunks to avoid building a giant full position array.
    """
    n = len(macro_times_raw_full)
    hist = np.zeros(len(bin_edges_um) - 1, dtype=np.int64)

    print("\nHistogramming photon positions in chunks...")
    print(f"  total photons = {n:,}")
    print(f"  chunk_size    = {chunk_size:,}")

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_raw = macro_times_raw_full[start:end]

        chunk_times_ns = chunk_raw.astype(np.float64) * macro_resolution_s * 1e9
        chunk_positions_um = times_to_positions_with_speed(
            times_ns=chunk_times_ns,
            speed_um_per_s=speed_um_per_s,
            start_um=start_um,
            time_offset_s=time_offset_s,
        )

        h, _ = np.histogram(chunk_positions_um, bins=bin_edges_um)
        hist += h

        if start == 0 or ((start // chunk_size) + 1) % 10 == 0 or end == n:
            print(f"  processed photons {start:,} to {end:,} of {n:,}")

    return hist.astype(np.float64)


def bin_bursts_by_position(
    burst_positions_um,
    burst_intensities,
    photon_counts_per_bin,
    bin_edges_um
):
    """
    Bin burst data into existing position bins.

    Parameters
    ----------
    burst_positions_um : 1D array
        Position of each burst
    burst_intensities : 1D array
        Burst intensity of each burst
    photon_counts_per_bin : 1D array
        Precomputed photon histogram per bin
    bin_edges_um : 1D array
        Position bin edges

    Returns
    -------
    bin_centers_um
    number_of_bursts
    mean_intensity
    median_intensity
    max_intensity
    total_photons_per_bin
    """
    burst_positions_um = np.asarray(burst_positions_um, dtype=np.float64)
    burst_intensities = np.asarray(burst_intensities, dtype=np.float64)
    total_photons_per_bin = np.asarray(photon_counts_per_bin, dtype=np.float64)

    n_position_bins = len(bin_edges_um) - 1
    bin_centers_um = 0.5 * (bin_edges_um[:-1] + bin_edges_um[1:])

    number_of_bursts = np.zeros(n_position_bins, dtype=np.float64)
    mean_intensity = np.zeros(n_position_bins, dtype=np.float64)
    median_intensity = np.zeros(n_position_bins, dtype=np.float64)
    max_intensity = np.zeros(n_position_bins, dtype=np.float64)

    if burst_positions_um.size > 0:
        burst_bin_idx = np.digitize(burst_positions_um, bin_edges_um) - 1
        valid = (burst_bin_idx >= 0) & (burst_bin_idx < n_position_bins)

        burst_bin_idx = burst_bin_idx[valid]
        burst_intensities = burst_intensities[valid]

        for i in range(n_position_bins):
            values = burst_intensities[burst_bin_idx == i]
            if values.size > 0:
                number_of_bursts[i] = float(values.size)
                mean_intensity[i] = float(np.mean(values))
                median_intensity[i] = float(np.median(values))
                max_intensity[i] = float(np.max(values))

    return (
        bin_centers_um,
        number_of_bursts,
        mean_intensity,
        median_intensity,
        max_intensity,
        total_photons_per_bin,
    )


def main_continuous_scan(ptufilename, base_user_setting):
    folder = os.path.dirname(ptufilename)
    output_folder = os.path.join(folder, "out")

    print(f"Creating output folder if needed: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    user_setting = build_user_setting(base_user_setting, output_folder)

    print("\nRunning continuous-scan burst analysis on:")
    print(ptufilename)

    scan_meta = load_scan_metadata(ptufilename)
    speed_um_per_s = scan_meta["speed_um_per_s"]
    start_um = scan_meta["start_um"]
    time_offset_s = scan_meta["time_offset_s"]
    n_position_bins = scan_meta["n_position_bins"]
    photon_hist_chunk_size = scan_meta["photon_hist_chunk_size"]

    df_photons, df_bursts = get_bursts(ptufilename, user_setting=user_setting)

    total_photons_in_file = float(
        df_photons.attrs.get("total_photons_in_file", len(df_photons))
    )

    print("\nPhotons dataframe head:")
    print(df_photons.head())

    print("\nBursts dataframe head:")
    print(df_bursts.head())

    print(f"\nTotal photons in file: {total_photons_in_file}")

    macro_times_raw_full = df_photons.attrs.get("macro_times_raw_full", None)
    macro_resolution_s = df_photons.attrs.get("macro_resolution_s", None)

    if macro_times_raw_full is None or macro_resolution_s is None:
        raise RuntimeError(
            "Full photon timing information is missing from df_photons.attrs. "
            "Please update get_bursts_tttrlib.py to store macro_times_raw_full "
            "and macro_resolution_s in df_photons.attrs."
        )

    print("\nComputing scan position range from first/last photon times...")
    pos_min, pos_max = get_position_range_from_macro_times(
        macro_times_raw_full=macro_times_raw_full,
        macro_resolution_s=macro_resolution_s,
        speed_um_per_s=speed_um_per_s,
        start_um=start_um,
        time_offset_s=time_offset_s,
    )

    print("Photon position sanity check:")
    print(f"  min photon position = {pos_min} um")
    print(f"  max photon position = {pos_max} um")

    if pos_max == pos_min:
        pos_max = pos_min + 1e-9

    bin_edges_um = np.linspace(pos_min, pos_max, n_position_bins + 1)

    total_photons_per_bin = histogram_photon_positions_chunked(
        macro_times_raw_full=macro_times_raw_full,
        macro_resolution_s=macro_resolution_s,
        speed_um_per_s=speed_um_per_s,
        start_um=start_um,
        time_offset_s=time_offset_s,
        bin_edges_um=bin_edges_um,
        chunk_size=photon_hist_chunk_size,
    )

    print(f"  Sum of photons across bins = {np.sum(total_photons_per_bin)}")

    if len(df_bursts) > 0:
        print("\nConverting burst start times to positions using scan speed...")
        burst_positions_um = times_to_positions_with_speed(
            times_ns=df_bursts["Burst timestart"].to_numpy(),
            speed_um_per_s=speed_um_per_s,
            start_um=start_um,
            time_offset_s=time_offset_s,
        )
        burst_intensities = df_bursts["Burst intensity"].to_numpy()

        print("Burst position sanity check:")
        print(f"  min burst position = {np.min(burst_positions_um)} um")
        print(f"  max burst position = {np.max(burst_positions_um)} um")
    else:
        burst_positions_um = np.array([], dtype=np.float64)
        burst_intensities = np.array([], dtype=np.float64)
        print("\nNo bursts found, continuing with empty burst arrays.")

    (
        positions_um,
        numb_molecules,
        mean_intensity,
        median_intensity,
        max_intensity,
        sum_photons,
    ) = bin_bursts_by_position(
        burst_positions_um=burst_positions_um,
        burst_intensities=burst_intensities,
        photon_counts_per_bin=total_photons_per_bin,
        bin_edges_um=bin_edges_um,
    )

    print("\nBinned profile summary:")
    print(f"  Number of bins = {len(positions_um)}")
    print(f"  First 10 positions = {positions_um[:10]}")
    print(f"  First 10 numb_molecules = {numb_molecules[:10]}")
    print(f"  First 10 sum_photons = {sum_photons[:10]}")

    alldata_rows = []
    for i in range(len(positions_um)):
        alldata_rows.append([
            float(positions_um[i]),
            float(numb_molecules[i]),
            float(mean_intensity[i]),
            float(median_intensity[i]),
            float(max_intensity[i]),
            float(sum_photons[i]),
        ])

    save_alldata_txt(output_folder, alldata_rows)
    save_inputparameter_txt(output_folder, user_setting)
    save_positions_txt(output_folder, positions_um)
    save_numb_molecules_txt(output_folder, numb_molecules)
    save_sumphoton_txt(output_folder, sum_photons)

    bursts_csv = os.path.join(output_folder, "bursts_tttrlib.csv")
    binned_csv = os.path.join(output_folder, "continuous_scan_binned_profile.csv")

    print(f"\nSaving bursts CSV to: {bursts_csv}")
    df_bursts.to_csv(bursts_csv, index=False)

    print(f"Saving binned profile CSV to: {binned_csv}")
    pd.DataFrame({
        "position_um": positions_um,
        "number_of_bursts": numb_molecules,
        "mean_intensity": mean_intensity,
        "median_intensity": median_intensity,
        "max_intensity": max_intensity,
        "sum_photons": sum_photons,
    }).to_csv(binned_csv, index=False)

    print("\nDone. Saved files:")
    print(os.path.join(output_folder, "Alldata.txt"))
    print(os.path.join(output_folder, "Analysed_data_inputparameter.txt"))
    print(os.path.join(output_folder, "Analysed_data_positions.txt"))
    print(os.path.join(output_folder, "Analysed_data_numb_molecules.txt"))
    print(os.path.join(output_folder, "Analysed_data_SumPhoton.txt"))
    print(bursts_csv)
    print(binned_csv)


if __name__ == "__main__":
    ptufilename = get_ptufilename()
    if not ptufilename:
        raise RuntimeError("No file selected.")

    user_setting = {
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
    }

    main_continuous_scan(ptufilename, user_setting)