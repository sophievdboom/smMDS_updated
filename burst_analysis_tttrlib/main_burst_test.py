import os
import re
import copy

from get_bursts_tttrlib import get_bursts


def save_inputparameter_txt(output_folder, user_setting):
    filepath = os.path.join(output_folder, "inputparameter.txt")
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


def extract_position_um(filename):
    print(f"\nExtracting position from filename: {filename}")
    match = re.search(r'([0-9]{1,6}(?:\.[0-9]{0,2})?)um_', filename)
    if match is None:
        raise ValueError(f"Could not extract position from filename: {filename}")

    position = float(match.group(1))
    print(f"Extracted position (um): {position}")
    return position


def save_alldata_txt(output_folder, row_values):
    filepath = os.path.join(output_folder, "Alldata.txt")
    print(f"\nSaving Alldata row to: {filepath}")

    row_str = ",".join(f"{v:.18e}" for v in row_values)

    with open(filepath, "w") as f:
        f.write(row_str + "\n")


def build_user_setting(base_user_setting, per_file_settings, ptufilename):
    """
    Build the final user_setting for the selected file by combining:
    - base settings
    - per-file overrides
    """
    if ptufilename not in per_file_settings:
        raise KeyError(f"No per-file settings found for: {ptufilename}")

    user_setting = copy.deepcopy(base_user_setting)
    user_setting.update(per_file_settings[ptufilename])

    print("\nFinal user_setting for this file:")
    for key, value in user_setting.items():
        print(f"  {key}: {value}")

    return user_setting


def main(ptufilename, user_setting):
    output_folder = user_setting["output_folder"]
    print(f"\nCreating output folder if needed: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    print("\nRunning burst analysis on:")
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

    save_alldata_txt(output_folder, alldata_row)
    save_inputparameter_txt(output_folder, user_setting)

    debug_photons_csv = os.path.join(output_folder, "debug_photons.csv")
    photons_csv = os.path.join(output_folder, "photons_tttrlib.csv")
    bursts_csv = os.path.join(output_folder, "bursts_tttrlib.csv")

    print(f"\nSaving first 100 photons to: {debug_photons_csv}")
    df_photons.head(100).to_csv(debug_photons_csv, index=False)

    print(f"Saving first 100 photons to: {photons_csv}")
    df_photons.head(100).to_csv(photons_csv, index=False)

    print(f"Saving bursts CSV to: {bursts_csv}")
    df_bursts.to_csv(bursts_csv, index=False)

    print("\nSaved files:")
    print(os.path.join(output_folder, "Alldata.txt"))
    print(os.path.join(output_folder, "inputparameter.txt"))
    print(debug_photons_csv)
    print(photons_csv)
    print(bursts_csv)


if __name__ == "__main__":

    filenames = [
        "Test_Data/2026-04-01_test_data/2026-04-01_Final_Test_Atto488_T2_All_Modes/conc-10pM_laser-485nm_640nm_Mode-T2_ch-2__StepScan_stepnumber-20_time-5s/3/216.34um_steps_PhotonData_T2.ptu",
        "Test_Data/2026-04-01_test_data/2026-04-01_Final_Test_Atto488_T3_All_Modes/conc-10pM_laser-485nm_640nm_Mode-T3_ch-2__StepScan_stepnumber-20_time-5s/3/239.66um_steps_PhotonData.ptu",
        "Example/HSA_20pM_PBS_0p01tween_100ulph_400steps_2sec_27/HSA_20pM_PBS_0p01tween_100ulph_400steps_2sec_27_594.85um_steps.ptu"
    ]

    base_user_setting = {
        "set_lee_filter": 2,
        "threshold_iT_signal": 0.05,
        "threshold_iT_noise": 0.1,
        "min_phs_burst": 10,
        "min_phs_noise": 160,
        "filter_name": "addLeefilter",
        "show_plot": False,
        "output_folder": "",
        "use_noise_regions": False
    }

    per_file_settings = {
        filenames[0]: {
            "output_folder": r"Sophie_burst_analysis/output_T2",
            "threshold_iT_signal": 0.05,
            "threshold_iT_noise": 0.1,
            "min_phs_burst": 10,
            "min_phs_noise": 160
        },
        filenames[1]: {
            "output_folder": r"Sophie_burst_analysis/output_T3",
            "threshold_iT_signal": 0.05,
            "threshold_iT_noise": 0.1,
            "min_phs_burst": 10,
            "min_phs_noise": 160
        },
        filenames[2]: {
            "output_folder": r"Sophie_burst_analysis/output_smMDS",
            "threshold_iT_signal": 0.05,
            "threshold_iT_noise": 0.1,
            "min_phs_burst": 10,
            "min_phs_noise": 160
        }
    }

    file_to_analyze = filenames[2]

    user_setting = build_user_setting(
        base_user_setting=base_user_setting,
        per_file_settings=per_file_settings,
        ptufilename=file_to_analyze
    )

    main(file_to_analyze, user_setting)