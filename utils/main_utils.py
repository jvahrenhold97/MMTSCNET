import logging
import argparse
import sys
import os
import laspy as lp

def setup_logging(log_level):
    """
    Sets up logging at the user-specified level.

    Args:
    log_level: Level of verbosity (Default/Debug).
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')

def parse_arguments():
    """
    Enables user-input argumnets to specify MMTSCNets functionalities.

    Returns:
    args: List of possible user-inputs.
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
                        help='Maximum scaling to apply when augmenting pointclouds. Range: [0.001 - 0.05]',
                        type=float, default=0.05)
    parser.add_argument('--ssstest',
                        help='Ratio for validation data. Range: [0.05 - 0.5]',
                        type=float, default=0.20)
    parser.add_argument('--capsel',
                        help='[ALS | TLS | ULS | ALL] - Which capture method should be used for training.',
                        type=str, default="ULS")
    parser.add_argument('--growsel',
                        help='[LEAF-ON | LEAF-OFF | ALL] - Which growth period should be used for training.',
                        type=str, default="LEAF-ON")
    parser.add_argument('--batchsize',
                        help='4, 8, 16, 32, ... always use Power of Two!',
                        type=int, default=16)
    parser.add_argument('--numpoints',
                        help='1024, 2048, ... always double!',
                        type=int, default=2048)
    parser.add_argument('--verbose',
                        help="Display debug logging if used.",
                        action='store_true')
    args = parser.parse_args()
    return args

def validate_inputs(datadir, workdir, modeldir, elimper, maxpcscale, ssstest, capsel, growsel, batchsize, numpoints):
    """
    Enables user-input argumnets to specify MMTSCNets functionalities.

    Args:
    datadir: See tooltips for args.
    workdir: See tooltips for args.
    modeldir: See tooltips for args.
    elimper: See tooltips for args.
    maxpcscale: See tooltips for args.
    ssstest: See tooltips for args.
    capsel: See tooltips for args.
    growsel: See tooltips for args.
    batchsize: See tooltips for args.
    numpoints: See tooltips for args.

    Returns:
    args: List of possible user-inputs.
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
    if maxpcscale > 0.001 and maxpcscale < 0.051:
        max_pcscale = maxpcscale
    elif maxpcscale < 0.001 or maxpcscale > 0.051:
        logging.error("Scaling factor is too small/large! Exiting now!")
        sys.exit(1)
    if ssstest > 0.05 and ssstest < 0.5:
        sss_test = ssstest
    else:
        logging.error("Train-Test ratio is too large/small! Exiting now!")
        sys.exit(1)
    if capsel == "ALS" or capsel == "ULS" or capsel == "TLS" or capsel == "ALL":
        cap_sel = capsel
    else:
        logging.error("Capture selection can only be [ALS | TLS | ULS | ALL]! Exiting now!")
        sys.exit(1)
    if growsel == "LEAF-ON" or growsel == "LEAF-OFF" or growsel == "ALL":
        grow_sel = growsel
    else:
        logging.error("Growth selection can only be [LEAF-ON | LEAF-OFF | ALL]! Exiting now!")
        sys.exit(1)
    if batchsize == 4 or batchsize == 8 or batchsize == 16 or batchsize == 32:
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
    Checks for the presence of FWF data.

    Args:
    data_dir: Directory containing source data.

    Returns:
    True/False
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
    Joins a filepath with a folder name to create a new filepath.

    Args:
    path: Filepath to directory where folder is supposed to be created.
    folder_name: Name of the folder to be created.

    Returns:
    full_path: The path with the appended folder name.
    """
    full_path = os.path.join(path + "/" + folder_name)
    return full_path

def contains_full_waveform_data(las_file_path):
    """
    Checks for the presence of FWFdata in a .las file.

    Args:
    las_file_path: Path to the .las file.

    Returns:
    True/False
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