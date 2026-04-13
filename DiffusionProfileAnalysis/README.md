# Installation

This package works with python 3.6.
Please make sure you have the correct version.
If you need to install python, I would recommend [Anaconda](https://www.anaconda.com).
All the following commands should be typed in the `Anaconda prompt` application.

## Install dependencies:
First, install opencv3 and tifffile with pip or conda:

with pip:
- `pip install tifffile`
- `pip install opencv-python`

with conda:
- `conda install -c menpo opencv3`
- `conda install -c conda-forge tifffile`

You should now be able to import the following from python:
- `import cv2`
- `import tifffile`

If you have windows and cv2 is not working, you might need to download the wheel yourself:
- https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv

you can then install the package with pip:
- `pip install opencv_python‑3.3.1‑cp36‑cp36m‑win_amd64.whl`

## Install package
Use pip to install the package:
`pip install path/to/package`

## Update
Use pip to uninstall the package, then install it:
`pip uninstall diffusion_device`
`pip install path/to/package`

## Check installation is successful
Open the Samples folder and run in that order:
`generate_metadata.py`
`generate_settings.py`
`sizescript.py`
If everything is running fine, you can keep the Samples folder and
The README and delete the rest.

# Usage

## JSON Files
There is two JSON files you need to run the script: metadata and settings.

### Metadata
The metadata file stores all the informations about your data.
This file is valuable in itself as it contains all the informations about your experiments.
If you give your data with this file to someone,
this person should know all there is to know
about your experiment without asking you anything.
If you feel one filed is missing, I will gladly add it.

This implies that if you feel the need to modify the metadata to get a different fit,
you are probably doing something wrong.

It is important to give good values for the metadata.
Otherwise the results will not be quantitative.

The data contained in this file are:
- The data files names, which contain the data.
- The background file name, which contain the exact same image,
  except without analyte.
- The optics background file name, which contain an image of an empty region
  of the chip taken with the same conditions.
- The exposure time for all these images
- The width and height of the channels
- For images with multiple channels, the width of the walls between the channels.
- The number of channels in a file
- The flow rate
- The ROI. If you want to analyse a slice, change the settings file!
- The reading position of the channels. In the **center** of the ROI.
- The image pixel size
- The type of data (single channel, multi channels, scans, stack, ...)
- The date, informations about the analyte, the buffer, the device, the operator.
- The temperature and viscosity.
- The Z position of scans (for confocal scans)
- The flow direction for each channel.

### Settings
The setting file contains everything you can change during the fitting.

- The test radius
- The distance from the edges of the channel to ignore.
- The number of species to fit (more = slower)
- The positions to fit (!!!Don't change the metadata!!!)
- The number of z slices for the simulation (11 is plenty)
- Should the profiles be normalized? (Probably)
- The slice to analyze. Distance from the center of the image.
- Flatten bright field? If you are unfortunate enough to not have a background,
  Do you want to try to save your data by flattening?
- What frames should be plotted (to check everything is fine)
- What frames should be analyzed?
- Should the errors be ignored? (For a stack, probably)
- Do you want to apply a Savitzky-Golay filter? (Not unless the only noise is gaussian)
- Does your stack move?
- Does your background has good features? (Corners or random blobs) (carbon powder)
- Should the bas fits be ignored?

# Improve your results
## Background
You need to have a good background if you want to extract meaningful data.

Don't:
- Don't flow any proteins before taking the Background (sticking kills the fit)
- Don't take a background without buffer in the channels

Do:
- Take a background before and after the experiment (use the latter)
- Take an image in an empty region of the chip
- Take the background with conditions as similar as possible to the data
  - same flow rate
  - same exposure
  - ...

## Flow rate
If the flow rate is too low, all the profiles will be flat.
If the flow rate is too high, all the profiles will be similar.
A good rule of thumbs is that the profiles should at least reach the walls
by the last channel, but the last channel should clearly show a slope.

# Errors
Any JSONDecodeError:
You modified a json file and now it is not valid. Look for single quotes or extra ','.

"Can't work with negative radii!":
The fit gave a negative radii for some reason.

"Can't normalise profiles"
One of the profiles is too negative. Probably caused by a bad background.

"The test radius are incorrectly specified."
Check the settings JSON file

"Number of profiles and reading positions mismatching."
Check the metadata JSON file

"signal to noise too low"
The signal is smaller than the noise!

"Least square error too large"
The fit is not good enough

'The test radius are too big!'
Check the settings JSON file. Decrease the lower bound.

'The test radius are too small!'
Check the settings JSON file. Increase the higher bound.

'The signal is too noisy!'
Was not able to detect the angle. Check your signal stand out of the noise.

"Can't find {}"
Does {} exist?

"Missing Key: '{}' not in {}"
Check you JSON files.

'Flow direction must be up or down for scans.'
Check your metadata.

'Unknown orientation'
Check your metadata.

"image too small to get entire channel."
Are the channels fully enclosed in your image?

'Channel not fully contained in the image'
Are the channels fully enclosed in your image?

"Large background. Probably incorrect."
How good is your background?

"Poorly defined slice"
Check settings.

"Can not have different exposure times when using stationary option."
Remove stack is stationary from settings.

"Can't find a single good frame"
Maybe your stack is not that good?

"Image/Background mask too small"
Send me the data if you feel like the background and intensity are fine.

"Edges incorrectly detected."
Send me the data if you feel like the background and intensity are fine.

"Can't get image infos"
Check The output folder. Send me the data if the processed image looks fine.

ValueError('A value in x_new is above the interpolation range.')
Send me the data if you feel like the background and intensity are fine.
