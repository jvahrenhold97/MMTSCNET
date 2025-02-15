import os
import logging
import laspy as lp
import numpy as np
import zipfile
import shutil
from utils import main_utils
from scipy.spatial import KDTree
import sys

def create_config_directory(local_pathlist, capsel, growsel, fwf_av):
    """
    Creates a temporary working directory for MMTSCNet, including subdirectories for different data types.

    This function:
    - Generates a structured directory tree based on the presence of Full-Waveform (FWF) data.
    - Organizes point clouds, images, and numerical metadata in separate folders.
    - Ensures all necessary directories exist for further processing.

    Args:
        local_pathlist (list): List of base paths for MMTSCNet.
        capsel (str): User-specified acquisition method.
        growsel (str): User-specified leaf condition.
        fwf_av (bool): Indicates the presence of Full-Waveform (FWF) data.

    Returns:
        list: Updated `local_pathlist` containing paths to the created directories.
    """
    if fwf_av == True:
        config_dir = local_pathlist[4]
        local_dir = os.path.join(config_dir + "/DATA_" + capsel + "_" + growsel + "_FWF")
        local_las_dir = main_utils.join_paths(local_dir, "LAS")
        local_fwf_dir = main_utils.join_paths(local_dir, "FWF")
        local_img_dir = main_utils.join_paths(local_dir, "IMG")
        local_met_dir = main_utils.join_paths(local_dir, "MET")
        local_las_dir_p = main_utils.join_paths(local_dir, "LAS_P")
        local_fwf_dir_p = main_utils.join_paths(local_dir, "FWF_P")
        local_img_dir_p = main_utils.join_paths(local_dir, "IMG_P")
        local_met_dir_p = main_utils.join_paths(local_dir, "MET_P")
        create_working_folder(local_dir)
        create_working_folder(local_las_dir)
        create_working_folder(local_fwf_dir)
        create_working_folder(local_img_dir)
        create_working_folder(local_met_dir)
        create_working_folder(local_las_dir_p)
        create_working_folder(local_fwf_dir_p)
        create_working_folder(local_img_dir_p)
        create_working_folder(local_met_dir_p)
        local_pathlist.append(local_dir)
        local_pathlist.append(local_las_dir)
        local_pathlist.append(local_fwf_dir)
        local_pathlist.append(local_img_dir)
        local_pathlist.append(local_met_dir)
        local_pathlist.append(local_las_dir_p)
        local_pathlist.append(local_fwf_dir_p)
        local_pathlist.append(local_img_dir_p)
        local_pathlist.append(local_met_dir_p)
        return local_pathlist
    else:
        config_dir = local_pathlist[2]
        local_dir = os.path.join(config_dir + "/DATA_" + capsel + "_" + growsel)
        local_las_dir = main_utils.join_paths(local_dir, "LAS")
        local_img_dir = main_utils.join_paths(local_dir, "IMG")
        local_met_dir = main_utils.join_paths(local_dir, "MET")
        local_las_dir_p = main_utils.join_paths(local_dir, "LAS_P")
        local_img_dir_p = main_utils.join_paths(local_dir, "IMG_P")
        local_met_dir_p = main_utils.join_paths(local_dir, "MET_P")
        create_working_folder(local_dir)
        create_working_folder(local_las_dir)
        create_working_folder(local_img_dir)
        create_working_folder(local_met_dir)
        create_working_folder(local_las_dir_p)
        create_working_folder(local_img_dir_p)
        create_working_folder(local_met_dir_p)
        local_pathlist.append(local_dir)
        local_pathlist.append(local_las_dir)
        local_pathlist.append(local_img_dir)
        local_pathlist.append(local_met_dir)
        local_pathlist.append(local_las_dir_p)
        local_pathlist.append(local_img_dir_p)
        local_pathlist.append(local_met_dir_p)
        return local_pathlist

def create_working_directory(workdir_path, fwf_av):
    """
    Creates a structured working directory for MMTSCNet preprocessing.

    This function:
    - Organizes unzipped source data and configuration files.
    - Creates subdirectories for different data types (LAS, FWF, FPC).
    - Adapts directory structure based on the presence of Full-Waveform (FWF) data.

    Args:
        workdir_path (str): User-specified working directory.
        fwf_av (bool): Indicates the presence of Full-Waveform (FWF) data.

    Returns:
        list: List of created directory paths.
    """
    unzipped_data_folder_name = "data_unzipped"
    configuration_data_folder_name = "data_config"
    fpc_name = "FPC"
    las_name = "LAS"
    fwf_name = "FWF"
    paths_to_create = []
    if fwf_av == True:
        source_data_unzipped_path = main_utils.join_paths(workdir_path, unzipped_data_folder_name)
        source_data_unzipped_path_las = main_utils.join_paths(source_data_unzipped_path, las_name)
        source_data_unzipped_path_fwf = main_utils.join_paths(source_data_unzipped_path, fwf_name)
        source_data_unzipped_path_fpc = main_utils.join_paths(source_data_unzipped_path, fpc_name)
        source_data_config_path = main_utils.join_paths(workdir_path, configuration_data_folder_name)
        paths_to_create.append(source_data_unzipped_path)
        paths_to_create.append(source_data_unzipped_path_las)
        paths_to_create.append(source_data_unzipped_path_fwf)
        paths_to_create.append(source_data_unzipped_path_fpc)
        paths_to_create.append(source_data_config_path)
    else:
        source_data_unzipped_path = main_utils.join_paths(workdir_path, unzipped_data_folder_name)
        source_data_unzipped_path_las = main_utils.join_paths(source_data_unzipped_path, las_name)
        source_data_config_path = main_utils.join_paths(workdir_path, configuration_data_folder_name)
        paths_to_create.append(source_data_unzipped_path)
        paths_to_create.append(source_data_unzipped_path_las)
        paths_to_create.append(source_data_config_path)
    for new_path in paths_to_create:
        create_working_folder(new_path)
    return paths_to_create

def create_working_folder(path):
    """
    Creates a folder at the specified path if it does not already exist.

    This function:
    - Checks whether the directory exists.
    - Creates the directory if it does not exist.
    - Logs a debug message if the directory already exists.

    Args:
        path (str): File path including the folder name to be created.
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        logging.debug("Folder @ %s already exists!", path)

def unzip_all_datasets(SOURCE_DATASET_PATH, pathlist, fwf_av):
    """
    Unzips LAS and FWF data from the original dataset.

    This function:
    - Checks if datasets are already extracted to avoid redundant operations.
    - Extracts LAS and FWF files from ZIP archives.
    - Ensures the dataset follows the expected directory structure.
    - Handles cases where only LAS data is available (no FWF).

    Args:
        SOURCE_DATASET_PATH (str): Path to the source dataset containing ZIP files.
        pathlist (list): List of predefined working directories for MMTSCNet.
        fwf_av (bool): Indicates whether Full-Waveform (FWF) data is available.

    Raises:
        SystemExit: If the dataset structure does not match the expected format.
    """
    if fwf_av == True:
        UNZIPPED_LAS_PATH = pathlist[1]
        UNZIPPED_FWF_PATH = pathlist[2]
        is_dataset_extracted = get_is_dataset_extracted(UNZIPPED_LAS_PATH)
        if is_dataset_extracted==False:
            base_data_folders = get_las_and_fwf_base_dir_paths(SOURCE_DATASET_PATH)
            for data_folder_path in base_data_folders:
                if data_folder_path.split("/")[1] == "LAS":
                    for file in os.listdir(data_folder_path):
                        logging.info("Extracting files for plot %s", file)
                        if "zip" in file:
                            FILE_PATH = main_utils.join_paths(data_folder_path, file)
                            dataset_name = file.split(".")[0]
                            DATASET_PATH = main_utils.join_paths(UNZIPPED_LAS_PATH, dataset_name)
                            os.mkdir(DATASET_PATH)
                            with zipfile.ZipFile(FILE_PATH, "r") as zip:
                                zip.extractall(DATASET_PATH)
                elif data_folder_path.split("/")[1] == "FWF":
                    for file in os.listdir(data_folder_path):
                        logging.info("Extracting files for plot %s", file)
                        if "zip" in file:
                            FILE_PATH = main_utils.join_paths(data_folder_path, file)
                            dataset_name = file.split(".")[0]
                            DATASET_PATH = main_utils.join_paths(UNZIPPED_FWF_PATH, dataset_name)
                            os.mkdir(DATASET_PATH)
                            with zipfile.ZipFile(FILE_PATH, "r") as zip:
                                zip.extractall(DATASET_PATH)
                else:
                    logging.error("Folders don't have the required structure, exiting!")
                    sys.exit(1)
        else:
            logging.warning("Already extracted dataset found, skipping!")
    else:
        UNZIPPED_LAS_PATH = pathlist[1]
        is_dataset_extracted = get_is_dataset_extracted(UNZIPPED_LAS_PATH)
        if is_dataset_extracted==False:
            base_data_folders = get_las_and_fwf_base_dir_paths(SOURCE_DATASET_PATH)
            for data_folder_path in base_data_folders:
                if data_folder_path.split("/")[1] == "las":
                    for file in os.listdir(data_folder_path):
                        logging.info("Extracting files for plot %s", file)
                        if "zip" in file:
                            FILE_PATH = main_utils.join_paths(data_folder_path, file)
                            dataset_name = file.split(".")[0]
                            DATASET_PATH = main_utils.join_paths(UNZIPPED_LAS_PATH, dataset_name)
                            os.mkdir(DATASET_PATH)
                            with zipfile.ZipFile(FILE_PATH, "r") as zip:
                                zip.extractall(DATASET_PATH)
                else:
                    logging.error("Folders don't have the required structure, exiting!")
                    sys.exit(1)
        else:
            logging.warning("Already extracted dataset found, skipping!")

def get_is_dataset_extracted(las_unzipped_path):
    """
    Checks if the dataset has already been extracted.

    This function:
    - Counts the number of subdirectories in the specified LAS unzipped path.
    - Determines if extracted datasets exist based on directory presence.

    Args:
        las_unzipped_path (str): Path to the directory containing extracted LAS files.

    Returns:
        bool: True if the dataset has been extracted (i.e., contains subdirectories), False otherwise.
    """
    extracted_datasets_count = 0
    for subdir in os.listdir(las_unzipped_path):
        extracted_datasets_count+=1
    if extracted_datasets_count > 0:
        return True
    else:
        return False
    
def get_las_and_fwf_base_dir_paths(data_source_path):
    """
    Retrieves the file paths to the zipped LAS and FWF data folders.

    This function:
    - Iterates through the source data directory.
    - Collects paths to all subdirectories containing LAS and FWF data.
    - Returns a list of paths to be used for extraction.

    Args:
        data_source_path (str): Path to the source dataset directory.

    Returns:
        list: A list of file paths pointing to LAS and FWF data folders.
    """
    las_fwf_base_dir_paths = []
    for subdir in os.listdir(data_source_path):
        subdir_path = main_utils.join_paths(data_source_path, subdir)
        las_fwf_base_dir_paths.append(subdir_path)
    return las_fwf_base_dir_paths

def create_fpcs(fwf_unzipped_path, fpc_unzipped_path):
    """
    Merges FWF flight strips into a single FWF point cloud for each plot.

    This function:
    - Checks if FPCs (Full-Waveform Point Clouds) have already been created.
    - Iterates through each plot directory, combining individual flight strips.
    - Appends FWF `.las` and `.laz` files into a single consolidated file per plot.
    - Ensures that the final output contains FWF data.

    Args:
        fwf_unzipped_path (str): Path to the unzipped individual FWF flight strip point clouds.
        fpc_unzipped_path (str): Path where the merged FWF point clouds will be saved.

    Returns:
        bool: True if FPCs were successfully created, False otherwise.
    """
    if get_are_fwf_pcs_extracted(fpc_unzipped_path) == False:
        for plot_folder in os.listdir(fwf_unzipped_path):
            logging.info("Creating FWF FPC for plot %s", plot_folder)
            plot_path = main_utils.join_paths(fwf_unzipped_path, plot_folder)
            index = 0
            for fwf_file in os.listdir(plot_path):
                if fwf_file.lower().endswith(".las"):
                    if index == 0:
                        fwf_file_path = main_utils.join_paths(plot_path, fwf_file)
                        savename = str(plot_folder) + ".las"
                        out_las = main_utils.join_paths(fpc_unzipped_path, savename)
                        shutil.copy2(fwf_file_path, out_las)
                        index+=1
                    else:
                        in_las = main_utils.join_paths(plot_path, fwf_file)
                        savename = str(plot_folder) + ".las"
                        out_las = main_utils.join_paths(fpc_unzipped_path, savename)
                        append_to_las(in_las, out_las)
                        index+=1
                elif fwf_file.lower().endswith(".laz"):
                    if index == 0:
                        fwf_file_path = main_utils.join_paths(plot_path, fwf_file)
                        savename = str(plot_folder) + ".las"
                        out_las = main_utils.join_paths(fpc_unzipped_path, savename)
                        shutil.copy2(fwf_file_path, out_las)
                        index+=1
                    else:
                        in_las = main_utils.join_paths(plot_path, fwf_file)
                        savename = str(plot_folder) + ".las"
                        out_las = main_utils.join_paths(fpc_unzipped_path, savename)
                        append_to_las(in_las, out_las)
                        index+=1
                else:
                    pass
            if contains_full_waveform_data(out_las):
                logging.info("File contains FWF data after saving!")
    else:
        logging.warning("FPCs appear to have already been created!")

def get_are_fwf_pcs_extracted(fwf_working_path):
    """
    Checks if individual FWF flight strips have already been extracted.

    This function:
    - Counts the number of files in the specified FWF working directory.
    - Determines whether extracted FWF files exist based on file presence.

    Args:
        fwf_working_path (str): Path to the directory containing extracted FWF flight strips.

    Returns:
        bool: True if extracted FWF files are found, False otherwise.
    """
    index=0
    for file in os.listdir(fwf_working_path):
        index+=1
    if index > 0:
        return True
    else: 
        return False
    
def append_to_las(in_laz, out_las):
    """
    Merges one LAS/LAZ file into another while preserving VLRs (Variable Length Records).

    This function:
    - Appends points from `in_laz` to `out_las` in chunks to optimize memory usage.
    - Transfers VLRs and EVLRs if the input file contains Full-Waveform (FWF) data,
      and the target file does not already include them.
    - Ensures efficient handling of large point cloud datasets.

    Args:
        in_laz (str): Path to the LAS/LAZ file to be appended.
        out_las (str): Path to the target LAS file where points will be added.
    """
    with lp.open(out_las, mode='a') as outlas:
        with lp.open(in_laz) as inlas:
            if contains_full_waveform_data(in_laz) and not contains_full_waveform_data(out_las):
                for vlr in inlas.header.vlrs:
                    outlas.header.vlrs.append(vlr)
                for evlr in inlas.header.evlrs:
                    outlas.header.evlrs.append(evlr)
            for points in inlas.chunk_iterator(2_000_000):
                outlas.append_points(points)

def contains_full_waveform_data(las_file_path):
    """
    Checks if a LAS/LAZ file contains Full-Waveform (FWF) data.

    This function:
    - Reads the LAS/LAZ file header.
    - Scans the Variable Length Records (VLRs) to determine if FWF data is present.
    - FWF data is indicated by record IDs between 100 and 354.

    Args:
        las_file_path (str): Path to the LAS/LAZ file to check.

    Returns:
        bool: True if the file contains FWF data, False otherwise.

    Logs:
        - Logs an error if the file cannot be read.
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
    
def extract_single_trees_from_fpc(fpc_unzipped_path, las_unzipped_path, las_working_path, fwf_working_path, capsel, growsel):
    """
    Extracts individual trees from Full-Waveform (FWF) plot point clouds.

    This function:
    - Filters extracted single-tree point clouds based on user-defined acquisition method and growth selection.
    - Uses a k-d tree to efficiently map corresponding points between the regular and FWF point clouds.
    - Writes the extracted FWF single-tree point clouds to separate files.

    Args:
        fpc_unzipped_path (str): Path to the extracted FWF plot-level point clouds.
        las_unzipped_path (str): Path to the extracted single-tree LAS point clouds.
        las_working_path (str): Target directory for saving extracted single-tree LAS files.
        fwf_working_path (str): Target directory for saving extracted single-tree FWF files.
        capsel (str): User-specified acquisition selection.
        growsel (str): User-specified leaf-condition selection.

    Logs:
        - Warnings if FWF trees are already extracted.
        - Debug logs for processing steps and point cloud formats.
        - Info logs when a single tree is successfully extracted.
    """
    cap1, cap2, cap3, grow1, grow2 = get_capgrow(capsel, growsel)
    if get_are_fwf_pcs_extracted(fwf_working_path) == False:
        id_counter = 0
        tree_index = -1
        for fpc in os.listdir(fpc_unzipped_path):
            if fpc.lower().endswith(".las") or fpc.lower().endswith(".laz"):
                fpc_file_path = main_utils.join_paths(fpc_unzipped_path, fpc)
                fpc_name = fpc.split(".")[0]
                inFile = lp.read(fpc_file_path)
                logging.debug("Original FWF point format: %s", inFile.header.point_format)
                fpc_source_cloud, fpc_header_text = readLas(inFile)
                kd_tree = KDTree(fpc_source_cloud[:, :3], leafsize=64)
                for plot in os.listdir(las_unzipped_path):
                    if plot == fpc_name:
                        plot_path = main_utils.join_paths(las_unzipped_path, plot)
                        for plot_pc_folder in os.listdir(plot_path):
                            if plot_pc_folder == "single_trees":
                                single_trees_plot_pc_folder = main_utils.join_paths(plot_path, plot_pc_folder)
                                for single_tree_pc_folder in os.listdir(single_trees_plot_pc_folder):
                                    single_tree_pc_folder_path = main_utils.join_paths(single_trees_plot_pc_folder, single_tree_pc_folder)
                                    tree_index+=1
                                    for single_tree_pc in os.listdir(single_tree_pc_folder_path):
                                        if single_tree_pc.lower().endswith(".laz") or single_tree_pc.lower().endswith(".las"):
                                            if cap1 in single_tree_pc or cap2 in single_tree_pc or cap3 in single_tree_pc:
                                                if grow1 in single_tree_pc or grow2 in single_tree_pc:
                                                    single_tree_pc_path = main_utils.join_paths(single_tree_pc_folder_path, single_tree_pc)
                                                    inFile_target = lp.read(single_tree_pc_path)
                                                    target_cloud, header_txt_target = readLas(inFile_target)
                                                    n_source = fpc_source_cloud.shape[0]
                                                    dist, idx = kd_tree.query(target_cloud[:, :3], k=10, eps=0.0)
                                                    idx = np.unique(idx)
                                                    idx = idx[idx != n_source]
                                                    exported_points = inFile.points[idx].copy()
                                                    outFile = lp.LasData(inFile.header)
                                                    outFile.vlrs = inFile.vlrs
                                                    outFile.points = exported_points
                                                    species = single_tree_pc.split("_")[0]
                                                    retrieval = single_tree_pc.split("_")[3]
                                                    method = single_tree_pc.split("_")[5].split(".")[0].split("-")[0]
                                                    if "on" in single_tree_pc:
                                                        output_path_fwf_pc = os.path.join(fwf_working_path + "/" + str(tree_index) + "_FWF_" + species + "_" + method + "_" + retrieval + "_" + str(id_counter) + "_" + growsel + "_aug00.laz")
                                                        output_path_las_pc = os.path.join(las_working_path + "/" + str(tree_index) + "_REG_" + species + "_" + method + "_" + retrieval + "_" + str(id_counter) + "_" + growsel + "_aug00.laz")
                                                        outFile.write(output_path_fwf_pc)
                                                        shutil.copy2(single_tree_pc_path, output_path_las_pc)
                                                        logging.info("Extracted tree #%s for plot %s with point format %s", id_counter, plot, outFile.header.point_format)
                                                        id_counter+=1
                                                        logging.debug("Does file contain fwf data? - %s", contains_full_waveform_data(output_path_fwf_pc))
                                                    else:
                                                        output_path_fwf_pc = os.path.join(fwf_working_path + "/" + str(tree_index) + "_FWF_" + species + "_" + method + "_" + retrieval + "_" + str(id_counter) + "_" + growsel + "_aug00.laz")
                                                        output_path_las_pc = os.path.join(las_working_path + "/" + str(tree_index) + "_REG_" + species + "_" + method + "_" + retrieval + "_" + str(id_counter) + "_" + growsel + "_aug00.laz")
                                                        outFile.write(output_path_fwf_pc)
                                                        shutil.copy2(single_tree_pc_path, output_path_las_pc)
                                                        logging.info("Extracted tree #%s for plot %s with point format %s", id_counter, plot, outFile.header.point_format)
                                                        id_counter+=1
                                                        logging.debug("Does file contain fwf data? - %s", contains_full_waveform_data(output_path_fwf_pc))
                                                else:
                                                    pass
                            else:
                                pass
                    else:
                        pass
                else:
                    pass
            else:
                pass
    else:
        logging.warning("FWF single trees have already been extracted, skipping!")

def readLas(file):
    """
    Reads a LAS file and extracts its point cloud data and dimensions.

    This function:
    - Retrieves the dimensions of the LAS file.
    - Extracts the XYZ coordinates of all points as a NumPy array.

    Args:
        file (LasData): A LAS file object loaded using `laspy`.

    Returns:
        tuple:
            - np.ndarray: `source_cloud` - A (N, 3) array containing [X, Y, Z] coordinates of points.
            - list: `dimensions` - The dimensions of the LAS file (retrieved via `getDimensions()`).
    """
    dimensions = getDimensions(file)
    source_cloud = np.array([file.x, file.y, file.z]).T
    return source_cloud, dimensions

def getDimensions(file):
    """
    Retrieves the dimensions of a LAS file.

    This function:
    - Iterates through the point format of the LAS file.
    - Concatenates the names of available dimensions into a string.

    Args:
        file (LasData): A LAS file object loaded using `laspy`.

    Returns:
        str: A space-separated string containing the names of all dimensions in the LAS file.
    """
    dimensions = ""
    for dim in file.point_format:
        dimensions += " " + dim.name
    return dimensions

def get_capgrow(capsel, growsel):
    """
    Determines the acquisition method and leaf-condition based on user input.

    This function:
    - Assigns up to three acquisition methods (`cap1`, `cap2`, `cap3`) based on `capsel`.
    - Determines the leaf-condition selection (`grow1`, `grow2`) based on `growsel`.

    Args:
        capsel (str): User-specified acquisition selection. Options: ["ALL", "ALS", "ULS", "TLS"].
        growsel (str): User-specified leaf-condition selection. Options: ["LEAF-ON", "LEAF-OFF", "ALL"].

    Returns:
        tuple:
            - cap1 (str): First acquisition method.
            - cap2 (str): Second acquisition method.
            - cap3 (str): Third acquisition method.
            - grow1 (str): First leaf-condition.
            - grow2 (str): Second leaf-condition.
    """
    if capsel == "ALL":
        cap1 = "ALS"
        cap2 = "TLS"
        cap3 = "ULS"
        if growsel == "LEAF-ON":
            grow1 = "on"
            grow2 = "on"
            return cap1, cap2, cap3, grow1, grow2
        elif growsel == "LEAF-OFF":
            grow1 = "off"
            grow2 = "off"
            return cap1, cap2, cap3, grow1, grow2
        else:
            grow1 = "on"
            grow2 = "off"
            return cap1, cap2, cap3, grow1, grow2
    elif capsel == "ALS":
        cap1 = "ALS"
        cap2 = "ALS"
        cap3 = "ALS"
        if growsel == "LEAF-ON":
            grow1 = "on"
            grow2 = "on"
            return cap1, cap2, cap3, grow1, grow2
        elif growsel == "LEAF-OFF":
            grow1 = "off"
            grow2 = "off"
            return cap1, cap2, cap3, grow1, grow2
        else:
            grow1 = "on"
            grow2 = "off"
            return cap1, cap2, cap3, grow1, grow2
    elif capsel == "ULS":
        cap1 = "ULS"
        cap2 = "ULS"
        cap3 = "ULS"
        if growsel == "LEAF-ON":
            grow1 = "on"
            grow2 = "on"
            return cap1, cap2, cap3, grow1, grow2
        elif growsel == "LEAF-OFF":
            grow1 = "off"
            grow2 = "off"
            return cap1, cap2, cap3, grow1, grow2
        else:
            grow1 = "on"
            grow2 = "off"
            return cap1, cap2, cap3, grow1, grow2
    else:
        cap1 = "TLS"
        cap2 = "TLS"
        cap3 = "TLS"
        if growsel == "LEAF-ON":
            grow1 = "on"
            grow2 = "on"
            return cap1, cap2, cap3, grow1, grow2
        elif growsel == "LEAF-OFF":
            grow1 = "off"
            grow2 = "off"
            return cap1, cap2, cap3, grow1, grow2
        else:
            grow1 = "on"
            grow2 = "off"
            return cap1, cap2, cap3, grow1, grow2
        
def copy_files_for_prediction(las_unzipped_path, las_working_path, capsel, growsel):
    """
    Copies single-tree LAS/LAZ point clouds to the working directory for predictions.

    This function:
    - Filters point clouds based on acquisition method (`capsel`) and leaf-condition (`growsel`).
    - Renames files to follow a standardized format for later processing.
    - Skips processing if files have already been copied to the working directory.

    Args:
        las_unzipped_path (str): Path to the extracted single-tree LAS point clouds.
        las_working_path (str): Path to the target working directory for LAS point clouds.
        capsel (str): User-specified acquisition method.
        growsel (str): User-specified leaf condition.

    Logs:
        - Info message if files have already been copied.
    """
    if get_are_fwf_pcs_extracted(las_working_path) == False:
        tree_index = 0
        id_counter = 0
        for plot_folder in os.listdir(las_unzipped_path):
            plot_path = os.path.join(las_unzipped_path, plot_folder)
            plot_name = plot_folder
            for subfolder in os.listdir(plot_path):
                if subfolder == "single_trees":
                    id_counter += 1
                    subfolder_path = os.path.join(plot_path, subfolder)
                    for folder in os.listdir(subfolder_path):
                        folder_path = os.path.join(subfolder_path, folder)
                        for file in os.listdir(folder_path):
                            if file.lower().endswith(".las") or file.lower().endswith(".laz"):
                                cap1, cap2, cap3, grow1, grow2 = get_capgrow(capsel, growsel)
                                if cap1 in file or cap2 in file or cap3 in file:
                                    if grow1 in file or grow2 in file:
                                        species = file.split("_")[0]
                                        retrieval = file.split("_")[3]
                                        method = file.split("_")[5].split("-")[0]
                                        filepath = os.path.join(folder_path, file)
                                        if "on" in file:
                                            las_working_path_pc = os.path.join(las_working_path + "/" + str(tree_index) + "_REG_" + species + "_" + method + "_" + retrieval + "_" + str(id_counter) + "_LEAF-ON_aug00.laz")
                                        elif "off" in file:
                                            las_working_path_pc = os.path.join(las_working_path + "/" + str(tree_index) + "_REG_" + species + "_" + method + "_" + retrieval + "_" + str(id_counter) + "_LEAF-OFF_aug00.laz")
                                        else:
                                            pass
                                        shutil.copy2(filepath, las_working_path_pc)
                                        tree_index += 1
    else:
        logging.info("Files have been extracted already, skipping!")

def files_extracted(directory):
    """
    Counts the number of files in a given directory.

    This function:
    - Iterates through all files in the specified directory.
    - Returns the total count of files found.

    Args:
        directory (str): Path to the directory to check.

    Returns:
        int: Number of files in the directory.
    """
    counter = 0
    for file in os.listdir(directory):
        counter += 1
    return counter

    




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