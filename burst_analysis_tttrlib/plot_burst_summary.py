'''
This script is written to see how the burst search went. It will show you position vs number of bursts, mean intensity, and sum photons.

To use just run and select in the pop up the Alldata.txt file you want to analyse. It will save a png of the results.

'''



import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


def select_data_file():
    """
    Open a file dialog to select Alldata.txt.
    """
    root = tk.Tk()
    root.withdraw()

    filename = filedialog.askopenfilename(
        title="Select Alldata.txt",
        filetypes=(
            ("Text files", "*.txt"),
            ("CSV files", "*.csv"),
            ("All files", "*.*"),
        ),
    )

    root.update()
    root.destroy()
    return filename


def load_alldata(filepath):
    """
    Load Alldata.txt with columns:
    0 = position_um
    1 = number_of_bursts
    2 = mean_intensity
    3 = median_intensity
    4 = max_intensity
    5 = sum_photons
    """
    data = np.loadtxt(filepath, delimiter=",")

    if data.ndim == 1:
        data = data[np.newaxis, :]

    if data.shape[1] < 6:
        raise ValueError(
            f"Expected at least 6 columns in Alldata.txt, but found {data.shape[1]}."
        )

    position_um = data[:, 0]
    number_of_bursts = data[:, 1]
    mean_intensity = data[:, 2]
    sum_photons = data[:, 5]

    return position_um, number_of_bursts, mean_intensity, sum_photons


def make_plot(position_um, number_of_bursts, mean_intensity, sum_photons, savepath):
    """
    Make one figure with three plots and save it.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(position_um, number_of_bursts)
    axes[0].set_ylabel("Number of bursts")
    axes[0].set_title("Burst detection summary")

    axes[1].plot(position_um, mean_intensity)
    axes[1].set_ylabel("Mean intensity")

    axes[2].plot(position_um, sum_photons)
    axes[2].set_ylabel("Sum photons")
    axes[2].set_xlabel("Position [µm]")

    plt.tight_layout()
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    filepath = select_data_file()
    if not filepath:
        raise RuntimeError("No file selected.")

    print(f"Selected file: {filepath}")

    position_um, number_of_bursts, mean_intensity, sum_photons = load_alldata(filepath)

    output_folder = os.path.dirname(filepath)
    savepath = os.path.join(output_folder, "burst_detection.png")

    print(f"Saving plot to: {savepath}")

    make_plot(
        position_um=position_um,
        number_of_bursts=number_of_bursts,
        mean_intensity=mean_intensity,
        sum_photons=sum_photons,
        savepath=savepath,
    )

    print("Done.")


if __name__ == "__main__":
    main()