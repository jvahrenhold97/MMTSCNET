import logging
import argparse
import sys
import os
import laspy as lp

def setup_logging(log_level):
    """
    Configures logging settings for the application.

    This function:
    - Sets the logging verbosity level.
    - Defines a standardized log message format.
    - Includes timestamps, logger names, and log levels in the output.

    Args:
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')

def parse_arguments():
    """
    Parses command-line arguments for configuring MMTSCNet.

    This function:
    - Enables user input to specify data directories, model settings, and hyperparameters.
    - Defines options for data selection, augmentation, and training configurations.
    - Supports both training and inference modes.

    Returns:
        argparse.Namespace: Parsed user-input arguments.
    """
    parser = argparse.ArgumentParser(description="MMTSCNet - Multi-Modal Tree Species Classification")
    parser.add_argument('--datadir',
                        help='Enter file path to base data. For formatting help check documentation.',
                        type=str, required=True)
    parser.add_argument('--workdir',
                        help='Enter file path to the desired working directory.',
                        type=str, required=True)
    parser.add_argument('--modeldir',
                        help='Enter file path to the desired model directory.',
                        type=str, required=True)
    parser.add_argument('--elimper',
                        help='Threshold percentage which defines which tree species should not be included based on their representation percentage. Range: [0 - 99]',
                        type=float, default=5.0)
    parser.add_argument('--maxpcscale',
                        help='Maximum scaling to apply when augmenting pointclouds. Range: [0.001 - 0.15]',
                        type=float, default=0.20)
    parser.add_argument('--ssstest',
                        help='Ratio for validation data. Range: [0.05 - 0.5]',
                        type=float, default=0.20)
    parser.add_argument('--capsel',
                        help='[ALS | TLS | ULS] - Which capture method should be used for training.',
                        type=str, default="ULS")
    parser.add_argument('--growsel',
                        help='[LEAF-ON | LEAF-OFF] - Which growth period should be used for training.',
                        type=str, default="LEAF-ON")
    parser.add_argument('--batchsize',
                        help='4, 8, 16, 32, ... always use Power of Two!',
                        type=int, default=8)
    parser.add_argument('--numpoints',
                        help='1024, 2048, ... always double!',
                        type=int, default=2048)
    parser.add_argument('--verbose',
                        help="Display debug logging if used.",
                        action='store_true')
    parser.add_argument('--inference',
                        help="Load optimal hyperparameters and start training immediately.",
                        action='store_true')
    args = parser.parse_args()
    return args

def validate_inputs(datadir, workdir, modeldir, elimper, maxpcscale, ssstest, capsel, growsel, batchsize, numpoints):
    """
    Validates user-input arguments for MMTSCNet's configuration.

    This function:
    - Checks if input values are within acceptable ranges.
    - Ensures valid selections for capture method, growth condition, batch size, and point cloud size.
    - Logs errors and exits if invalid values are provided.

    Args:
        datadir (str): Path to the base data directory.
        workdir (str): Path to the working directory.
        modeldir (str): Path to the model directory.
        elimper (float): Percentage threshold for species elimination (0-99).
        maxpcscale (float): Maximum scaling factor for point cloud augmentation (0.01-0.3).
        ssstest (float): Train-test split ratio (0.05-0.5).
        capsel (str): Capture selection method [ALS | TLS | ULS.
        growsel (str): Growth condition selection [LEAF-ON | LEAF-OFF].
        batchsize (int): Training batch size [4 | 8 | 16 | 32].
        numpoints (int): Number of points per point cloud [512 | 1024 | 2048 | 4096].

    Returns:
        tuple: Validated configuration parameters:
            - data_dir (str): Validated data directory.
            - work_dir (str): Validated working directory.
            - model_dir (str): Validated model directory.
            - elim_per (float): Validated species elimination threshold.
            - max_pcscale (float): Validated point cloud scaling factor.
            - sss_test (float): Validated train-test split ratio.
            - cap_sel (str): Validated capture selection method.
            - grow_sel (str): Validated growth selection method.
            - bsize (int): Validated batch size.
            - img_size (int): Image size (fixed at 224).
            - pc_size (int): Validated point cloud size.
    """
    data_dir = datadir
    work_dir = workdir
    model_dir = modeldir
    if elimper <= 99 and elimper > 0:
        elim_per = elimper
    elif elimper == 0:
        elim_per = elimper
    else:
        logging.error("Elimination percentage can not be 100 or higher! Exiting now!")
        sys.exit(1)
    if maxpcscale > 0.01 and maxpcscale < 0.3:
        max_pcscale = maxpcscale
    elif maxpcscale < 0.01 or maxpcscale > 0.3:
        logging.error("Scaling factor is too small/large! Exiting now!")
        sys.exit(1)
    if ssstest > 0.05 and ssstest < 0.5:
        sss_test = ssstest
    else:
        logging.error("Train-Test ratio is too large/small! Exiting now!")
        sys.exit(1)
    if capsel == "ALS" or capsel == "ULS" or capsel == "TLS":
        cap_sel = capsel
    else:
        logging.error("Capture selection can only be [ALS | TLS | ULS]! Exiting now!")
        sys.exit(1)
    if growsel == "LEAF-ON" or growsel == "LEAF-OFF":
        grow_sel = growsel
    else:
        logging.error("Growth selection can only be [LEAF-ON | LEAF-OFF]! Exiting now!")
        sys.exit(1)
    if batchsize == 4 or batchsize == 8 or batchsize == 10 or batchsize == 12 or batchsize == 16 or batchsize == 32:
        bsize = batchsize
    else:
        logging.error("Batch size can only be [4 | 8 | 16 | 32]! Exiting now!")
        sys.exit(1)
    if numpoints == 512 or numpoints == 1024 or numpoints == 2048 or numpoints == 4096:
        pc_size = numpoints
    else:
        logging.error("Point cloud sampling size can only be [512 | 1024 | 2048 | 4096]! Exiting now!")
        sys.exit(1)
    img_size = 224
    return data_dir, work_dir, model_dir, elim_per, max_pcscale, sss_test, cap_sel, grow_sel, bsize, img_size, pc_size

def are_fwf_pointclouds_available(data_dir):
    """
    Determines whether Full Waveform (FWF) point cloud data is available.

    This function:
    - Scans the given directory for subfolders containing "fwf" or "FWF" in their names.
    - Returns `True` if at least one such folder is found, otherwise returns `False`.

    Args:
        data_dir (str): Directory containing the source data.

    Returns:
        bool: `True` if FWF data is available, `False` otherwise.
    """
    fwf_folders = []
    for subfolder in os.listdir(data_dir):
        if "fwf" in subfolder or "FWF" in subfolder:
            fwf_folders.append(subfolder)
        else:
            pass
    if len(fwf_folders) > 0:
        return True
    else:
        return False
    
def join_paths(path, folder_name):
    """
    Constructs a full file path by appending a folder name to a given directory path.

    This function:
    - Joins the specified directory path and folder name.
    - Returns the full path for further use.

    Args:
        path (str): Directory where the folder is located or should be created.
        folder_name (str): Name of the folder to be appended to the path.

    Returns:
        str: Full path with the appended folder name.
    """
    full_path = os.path.join(path + "/" + folder_name)
    return full_path

def contains_full_waveform_data(las_file_path):
    """
    Determines whether a given .las file contains Full Waveform (FWF) data.

    This function:
    - Reads the .las file using the `laspy` library.
    - Checks if any Variable Length Record (VLR) in the file header has a record ID 
      between 100 and 354, indicating the presence of FWF data.
    - Logs an error if the file cannot be read.

    Args:
        las_file_path (str): Path to the .las file.

    Returns:
        bool: `True` if FWF data is present, `False` otherwise.
    """
    try:
        las = lp.read(las_file_path)
        for vlr in las.header.vlrs:
            if 99 < vlr.record_id < 355:
                return True
        return False
    except Exception as e:
        logging.error(f"Error reading LAS file: {e}")
        return False
    




# ------------------------------- MIT License ----------------------------------
#
# Copyright (c) 2025 Jan Richard Vahrenhold
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.