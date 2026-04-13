from diffusion_device.keys import metadata

# path to Alldata.txt
datapath = './Test_Data/2026-04-10_GOOD_Atto488_T2_smMDS/conc-1nM_laser-485nm_640nm_Mode-T2_ch-2_T-23_buffer-1xsb_fr-100mlph_power-con100_chipname-master3.3_fabdate-200226_h-28.1__StepScan_stepnumber-200_time-2s/3/out/Alldata.txt'
json_infos = {}

# Data Type
# One of "single_pos_scan", "multi_pos_scan", "single_pos_image",
# multi_pos_image", "stack_multi_pos_image"
json_infos['Type of data'] = 'multi_pos_scan'

# delimiter is the delimiter to separate values in file (None for space)
# skiprows is the number of rows to skip
# index are the index to select. List of tuple given to slice().
# transpose is only needed for 'single_pos_scan'. Should transpose?
json_infos['Scan file structure'] = {'delimiter': ",",
                                     'skiprows': 0,
                                     'index': ((0, 400), (1, 2)),
                                     'transpose': False}

# Location of the data file(s).
# If None, will take the last path part of datapath.
json_infos['Image file name'] = None

# Background file, relative to the image file. None means no background
json_infos['Background file name'] = None

# image file to remove the background coming from the optics (Not chip related)
json_infos['Optics background file name'] = None

# Should the optics background be subtracted instead of divided.
json_infos['Subtract optics background'] = False

# Image exposition time [s]
json_infos['Image exposition time [s]'] = 1

# Background image exposition time [s]
json_infos['Background image exposition time [s]'] = 1

# Optics background image exposition time [s]
json_infos['Optics background image exposition time [s]'] = None

# Height of the channel [m]
json_infos['Wz[m]'] = 27e-6

# Width of the channel [m]
json_infos['Wy[m]'] = 225e-6

# Width of the walls [m] (Only for multiple channel in an image)
json_infos['Wall Width [m]'] = 50e-6

# Number of channels in a file
json_infos['Number of channels in a file'] = 4

# Flow [ulph].
json_infos['Q[ulph]'] = 100

# Reading position at the middle of the image [m]
json_infos['Read Positions [m]'] = [
    1000e-6,
    16000.007e-6,
    40000.014e-6,
    80000.005e-6]

# Pixel Size [m]
json_infos['Pixel Size [m]'] = 3.516e-6

# order of the region of interest [px]
json_infos['Image border[px] (t, d, l, r)'] = [
    None,
    None,
    None,
    None]

# Date [YYYYMMDD]
json_infos['Date [YYYYMMDD]'] = '20201115'

# Analyte informations
json_infos['Analyte informations'] = '20pM HSA'

# Buffer informations
json_infos['Buffer informations'] = '1xPBS pH7'

# Device informations
json_infos['Device informations'] = 'Kadi100x50 model 1'

# profile position. None means the mean over Z is used.
json_infos['Z position of scans [m]'] = None

# Where is the flow going for each reading position?
# For scans, only use 'u' up or 'd' down
json_infos['Flow direction (u, d, l, r)'] = [
    'd',
    'u',
    'd',
    'u']

# Operator
json_infos['Operator'] = 'Raphaël'

# Success [1-3]
json_infos['Success [1-3]'] = 3

# Temperature [K]
json_infos['Temperature [K]'] = 295

# Viscosity [Pa s]
json_infos['Viscosity [Pa s]'] = 1e-3

# Frame rate for movies
json_infos['Frame rate [1/s]'] = None

# For non constant frames rate
json_infos['Frames times [s]'] = None

# Inlet channel location in multiple pos data.
# (top, bottom, left, right)
# For scan, use only left and right.
json_infos['Inlet location'] = None

# For scans: Is the wall brighter than the channel background?
json_infos['Bright wall?'] = False

metadata.generate_json(datapath, json_infos)
