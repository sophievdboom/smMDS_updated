import os
import re
import glob
import copy
import tkinter as tk
from tkinter import filedialog

from get_bursts_tttrlib import get_bursts


def get_ptufilename():
    """
    Open a file dialog and let the user select one .ptu file.
    Then all .ptu files in the same folder will be analysed.
    """
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(
        filetypes=(("ptu files", "*.ptu"), ("all files", "*.*"))
    )
    root.update()
    root.destroy()
    return filename


def extract_position_um(filename):
    print(f"\nExtracting position from filename: {filename}")
    match = re.search(r'([0-9]{1,6}(?:\.[0-9]{0,2})?)um_', filename)
    if match is None:
        raise ValueError(f"Could not extract position from filename: {filename}")

    position = float(match.group(1))
    print(f"Extracted position (um): {position}")
    return position


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


def save_positions_txt(output_folder, positions):
    filepath = os.path.join(output_folder, "Analysed_data_positions.txt")
    print(f"Saving positions to: {filepath}")

    with open(filepath, "w") as f:
        for value in positions:
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
    """
    Build final user_setting for this folder.
    """
    user_setting = copy.deepcopy(base_user_setting)
    user_setting["output_folder"] = output_folder

    print("\nFinal user_setting:")
    for key, value in user_setting.items():
        print(f"  {key}: {value}")

    return user_setting


def analyse_folder(selected_file, base_user_setting):
    folder = os.path.dirname(selected_file)

    print(f"\nSelected file: {selected_file}")
    print(f"Analysing all PTU files in folder: {folder}")

    filenames = sorted(glob.glob(os.path.join(folder, "*.ptu")))
    if not filenames:
        raise FileNotFoundError(f"No .ptu files found in folder: {folder}")

    print(f"Found {len(filenames)} PTU files.")

    # Save everything in a subfolder called 'out' inside the PTU folder
    output_folder = os.path.join(folder, "out")
    print(f"Creating output folder if needed: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    user_setting = build_user_setting(base_user_setting, output_folder)

    alldata_rows = []
    positions = []
    numb_molecules = []
    sum_photons = []

    for i, ptufilename in enumerate(filenames, start=1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(filenames)}] Analysing file:")
        print(ptufilename)

        position_um = extract_position_um(ptufilename)

        df_photons, df_bursts = get_bursts(ptufilename, user_setting=user_setting)

        print("\nPhotons dataframe head:")
        print(df_photons.head())

        print("\nBursts dataframe head:")
        print(df_bursts.head())

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

        total_photons_in_file = float(len(df_photons))

        print("\nComputed summary values:")
        print(f"  position_um           = {position_um}")
        print(f"  number_of_bursts      = {number_of_bursts}")
        print(f"  mean_intensity        = {mean_intensity}")
        print(f"  median_intensity      = {median_intensity}")
        print(f"  max_intensity         = {max_intensity}")
        print(f"  total_photons_in_file = {total_photons_in_file}")

        alldata_row = [
            position_um,
            number_of_bursts,
            mean_intensity,
            median_intensity,
            max_intensity,
            total_photons_in_file
        ]

        alldata_rows.append(alldata_row)
        positions.append(position_um)
        numb_molecules.append(number_of_bursts)
        sum_photons.append(total_photons_in_file)

        # Per-file debug saves
        base_name = os.path.splitext(os.path.basename(ptufilename))[0]
        debug_photons_csv = os.path.join(output_folder, f"{base_name}_debug_photons.csv")
        photons_csv = os.path.join(output_folder, f"{base_name}_photons_tttrlib.csv")
        bursts_csv = os.path.join(output_folder, f"{base_name}_bursts_tttrlib.csv")

        print(f"\nSaving first 100 photons to: {debug_photons_csv}")
        df_photons.head(100).to_csv(debug_photons_csv, index=False)

        print(f"Saving first 100 photons to: {photons_csv}")
        df_photons.head(100).to_csv(photons_csv, index=False)

        print(f"Saving bursts CSV to: {bursts_csv}")
        df_bursts.to_csv(bursts_csv, index=False)

    # Sort final outputs by position
    combined = list(zip(positions, numb_molecules, sum_photons, alldata_rows))
    combined.sort(key=lambda x: x[0])

    positions = [x[0] for x in combined]
    numb_molecules = [x[1] for x in combined]
    sum_photons = [x[2] for x in combined]
    alldata_rows = [x[3] for x in combined]

    print("\n" + "=" * 80)
    print("Saving final batch output files...")

    save_alldata_txt(output_folder, alldata_rows)
    save_inputparameter_txt(output_folder, user_setting)
    save_positions_txt(output_folder, positions)
    save_numb_molecules_txt(output_folder, numb_molecules)
    save_sumphoton_txt(output_folder, sum_photons)

    print("\nDone. Saved files:")
    print(os.path.join(output_folder, "Alldata.txt"))
    print(os.path.join(output_folder, "Analysed_data_inputparameter.txt"))
    print(os.path.join(output_folder, "Analysed_data_positions.txt"))
    print(os.path.join(output_folder, "Analysed_data_numb_molecules.txt"))
    print(os.path.join(output_folder, "Analysed_data_SumPhoton.txt"))


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
        "use_noise_regions": False
    }

    selected_file = get_ptufilename()
    if not selected_file:
        raise RuntimeError("No file selected.")

    analyse_folder(selected_file, base_user_setting)