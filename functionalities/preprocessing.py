import os
import laspy as lp
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
from collections import Counter
from scipy.spatial import KDTree, ConvexHull
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from utils import main_utils
import logging
from functionalities import workspace_setup
from collections import defaultdict
import shutil
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from scipy.stats import entropy
import matplotlib.colors as mcolors
import multiprocessing as mp
import sys
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def load_and_center(file):
    """
    Loads a point cloud from a file and centers it.

    This function:
    - Loads the point cloud data from the specified file.
    - Centers the point cloud by adjusting its coordinates.
    - Logs the processing step for tracking.

    Args:
        file (str): Path to the point cloud file.

    Returns:
        np.ndarray: Centered point cloud.
    """
    logging.info(f"Loading and centering: {file}")
    sys.stdout.flush()
    point_cloud = load_point_cloud(file)
    return center_point_cloud([point_cloud])[0]

def resample_single(args):
    """
    Resamples a single point cloud to a specified number of points.

    This function:
    - Calls `resample_pointcloud()` to ensure a fixed number of points.
    - Validates the output type and shape.
    - Raises errors if the resampling fails or the shape is incorrect.

    Args:
        args (tuple): A tuple containing:
            - point_cloud (np.ndarray): The input point cloud.
            - netpcsize (int): The target number of points.
            - idx (int): The index of the point cloud (for logging/debugging).

    Returns:
        np.ndarray: The resampled point cloud with shape `(netpcsize, 3)`.

    Raises:
        ValueError: If the resampling fails, returns an unexpected type, or has an incorrect shape.
    """
    point_cloud, netpcsize, idx = args
    try:
        resampled_pc = resample_pointcloud(point_cloud, netpcsize, idx)  
        if not isinstance(resampled_pc, np.ndarray):
            raise ValueError(f"Point cloud {idx} returned non-array type {type(resampled_pc)}")
        if resampled_pc.shape != (netpcsize, 3):
            raise ValueError(f"Point cloud {idx} has shape {resampled_pc.shape}, expected ({netpcsize}, 3)")
        return resampled_pc
    except Exception as e:
        raise ValueError(f"Error resampling point cloud {idx}: {e}")

def resample_pointclouds_fps(selected_pointclouds, netpcsize):
    """
    Loads, centers, and resamples a set of point clouds in parallel using Farthest Point Sampling (FPS).

    This function:
    - Uses multiprocessing to efficiently load and center point clouds.
    - Resamples each point cloud to a fixed number of points (`netpcsize`).
    - Ensures all point clouds maintain a consistent shape.

    Args:
        selected_pointclouds (list): List of file paths to the point clouds.
        netpcsize (int): The target number of points for resampling.

    Returns:
        np.ndarray: An array of resampled point clouds, each with shape `(netpcsize, 3)`.
    """
    num_workers = mp.cpu_count()//2
    with mp.Pool(processes=num_workers) as pool:
        centered_pointclouds = pool.map(load_and_center, selected_pointclouds)
    logging.info("Centered a set of %s pointclouds", len(centered_pointclouds))
    with mp.Pool(processes=num_workers) as pool:
        resampled_pointclouds = pool.map(resample_single, [(centered_pointclouds[i], netpcsize, i) for i in range(len(centered_pointclouds))])
    logging.info("Resampled a set of %s pointclouds", len(resampled_pointclouds))
    return np.array(resampled_pointclouds)

def remove_insufficient_pointclouds_fwf(reg_pc_folder, fwf_pc_folder, netpcsize):
    """
    Removes insufficiently dense point clouds from both regular and FWF datasets.

    This function:
    - Iterates through regular and FWF point cloud files.
    - Matches files based on tree ID and species.
    - Loads and checks the number of points in each matched point cloud.
    - Removes point clouds that contain fewer points than `netpcsize`.

    Args:
        reg_pc_folder (str): Path to the folder containing regular point clouds.
        fwf_pc_folder (str): Path to the folder containing full waveform (FWF) point clouds.
        netpcsize (int): Minimum required number of points for a point cloud to be retained.

    Logs:
        - Info message when an insufficient point cloud is removed.
    """
    for pointcloud in os.listdir(reg_pc_folder):
        reg_id = pointcloud.split("_")[0]
        reg_species = pointcloud.split("_")[2]
        for fwf_pointcloud in os.listdir(fwf_pc_folder):
            fwf_id = fwf_pointcloud.split("_")[0]
            fwf_species = fwf_pointcloud.split("_")[2]
            if reg_id == fwf_id and reg_species == fwf_species:
                file_path = os.path.join(reg_pc_folder, pointcloud)
                fwf_path = os.path.join(fwf_pc_folder, fwf_pointcloud)
                if os.path.isfile(file_path):
                    if os.path.isfile(fwf_path):
                        las_file = lp.read(file_path)
                        points = np.vstack((las_file.x, las_file.y, las_file.z)).transpose()
                        if len(points) < netpcsize:
                            logging.info("Found pointcloud with %s/%s points. Removing it!", len(points), netpcsize)
                            os.remove(os.path.join(reg_pc_folder, pointcloud))
                            os.remove(os.path.join(fwf_pc_folder, fwf_pointcloud))
                    else:
                        pass
                else:
                    pass

def remove_insufficient_pointclouds(reg_pc_folder, netpcsize):
    """
    Removes point clouds with fewer points than the required threshold.

    This function:
    - Iterates through point cloud files in the specified directory.
    - Loads each LAS file and extracts its point data.
    - Checks if the number of points is below the required `netpcsize`.
    - Deletes insufficient point clouds.

    Args:
        reg_pc_folder (str): Path to the folder containing point clouds.
        netpcsize (int): Minimum required number of points for a point cloud to be retained.

    Logs:
        - Info message when an insufficient point cloud is removed.
    """
    for pointcloud in os.listdir(reg_pc_folder):
        file_path = os.path.join(reg_pc_folder, pointcloud)
        las_file = lp.read(file_path)
        points = np.vstack((las_file.x, las_file.y, las_file.z)).transpose()
        if len(points) < netpcsize:
            logging.info("Found pointcloud with %s/%s points. Removing it!", len(points), netpcsize)
            os.remove(os.path.join(reg_pc_folder, pointcloud))

def remove_height_outliers(reg_pc_folder):
    """
    Removes point clouds with heights outside the range: mean_height ± 0.85 * std_dev for their species.

    This function:
    - Computes the mean and standard deviation of tree heights for each species.
    - Identifies trees whose height is an outlier based on ±0.85 * standard deviation.
    - Removes outlier point clouds from the dataset.

    Args:
        reg_pc_folder (str): Path to the folder containing regular .laz files.

    Returns:
        None

    Logs:
        - Warnings for any errors while processing files.
        - Info messages when an outlier point cloud is removed.
    """
    species_heights = {}
    for pointcloud in os.listdir(reg_pc_folder):
        if not pointcloud.endswith(".laz"):
            continue
        parts = pointcloud.split("_")
        if len(parts) < 3:
            continue
        reg_species = parts[2]
        file_path = os.path.join(reg_pc_folder, pointcloud)
        try:
            las_file = lp.read(file_path)
            points = np.vstack((las_file.x, las_file.y, las_file.z)).T
            tree_height = np.max(points[:, 2]) - np.min(points[:, 2])
            if reg_species not in species_heights:
                species_heights[reg_species] = []
            species_heights[reg_species].append(tree_height)
        except Exception as e:
            logging.warning(f"Error processing {pointcloud}: {e}")
    species_stats = {
        species: (np.mean(heights), np.std(heights))
        for species, heights in species_heights.items() if len(heights) > 0
    }
    for pointcloud in os.listdir(reg_pc_folder):
        if not pointcloud.endswith(".laz"):
            continue
        parts = pointcloud.split("_")
        if len(parts) < 3:
            continue
        reg_species = parts[2]
        file_path = os.path.join(reg_pc_folder, pointcloud)
        if reg_species not in species_stats:
            continue
        try:
            las_file = lp.read(file_path)
            points = np.vstack((las_file.x, las_file.y, las_file.z)).T
            tree_height = np.max(points[:, 2]) - np.min(points[:, 2])
            mean_height, std_dev = species_stats[reg_species]
            lower_bound = mean_height - 0.85 * std_dev
            upper_bound = mean_height + 0.85 * std_dev
            if tree_height < lower_bound or tree_height > upper_bound:
                logging.info(f"Removing outlier: {pointcloud} (Height: {tree_height:.2f}, Allowed: {lower_bound:.2f} - {upper_bound:.2f})")
                os.remove(file_path)
        except Exception as e:
            logging.warning(f"Error processing {pointcloud}: {e}")

def remove_height_outliers_fwf(reg_pc_folder, fwf_pc_folder):
    """
    Removes point clouds whose height is outside mean ± 0.85 * std_dev for their species.
    Also removes corresponding FWF point clouds.

    This function:
    - Computes the mean and standard deviation of tree heights for each species.
    - Identifies trees whose height is an outlier.
    - Removes the outlier's corresponding FWF file from the dataset.

    Args:
        reg_pc_folder (str): Path to the folder containing regular .laz files.
        fwf_pc_folder (str): Path to the folder containing FWF .laz files.

    Returns:
        None

    Logs:
        - Warnings for any errors while processing files.
        - Info messages when an outlier point cloud is removed.
    """
    species_heights = {}
    for pointcloud in os.listdir(reg_pc_folder):
        if not pointcloud.endswith(".laz"):
            continue
        parts = pointcloud.split("_")
        if len(parts) < 3:
            continue
        reg_species = parts[2]
        file_path = os.path.join(reg_pc_folder, pointcloud)
        try:
            las_file = lp.read(file_path)
            points = np.vstack((las_file.x, las_file.y, las_file.z)).T
            tree_height = np.max(points[:, 2]) - np.min(points[:, 2])
            if reg_species not in species_heights:
                species_heights[reg_species] = []
            species_heights[reg_species].append(tree_height)
        except Exception as e:
            logging.warning(f"Error processing {pointcloud}: {e}")
    species_stats = {
        species: (np.mean(heights), np.std(heights))
        for species, heights in species_heights.items() if len(heights) > 0
    }
    for pointcloud in os.listdir(reg_pc_folder):
        if not pointcloud.endswith(".laz"):
            continue
        parts = pointcloud.split("_")
        if len(parts) < 3:
            continue
        reg_id = parts[0]
        reg_species = parts[2]
        file_path = os.path.join(reg_pc_folder, pointcloud)
        if reg_species not in species_stats:
            continue
        try:
            las_file = lp.read(file_path)
            points = np.vstack((las_file.x, las_file.y, las_file.z)).T
            tree_height = np.max(points[:, 2]) - np.min(points[:, 2])
            mean_height, std_dev = species_stats[reg_species]
            lower_bound = mean_height - 0.85 * std_dev
            upper_bound = mean_height + 0.85 * std_dev
            if tree_height < lower_bound or tree_height > upper_bound:
                logging.info(f"Removing outlier: {pointcloud} (Height: {tree_height:.2f}, Allowed: {lower_bound:.2f} - {upper_bound:.2f})")
                os.remove(file_path)
                for fwf_pointcloud in os.listdir(fwf_pc_folder):
                    fwf_parts = fwf_pointcloud.split("_")
                    if len(fwf_parts) < 3:
                        continue
                    fwf_id = fwf_parts[0]
                    fwf_species = fwf_parts[2]
                    fwf_path = os.path.join(fwf_pc_folder, fwf_pointcloud)
                    if reg_id == fwf_id and reg_species == fwf_species:
                        logging.info(f"Removing corresponding FWF: {fwf_pointcloud}")
                        os.remove(fwf_path)
        except Exception as e:
            logging.warning(f"Error processing {pointcloud}: {e}")

def get_base_filenames(folder, keyword):
    """
    Generates a set of base filenames by removing a specific keyword.

    This function scans a given folder, extracts filenames that end with `.laz`,
    and removes the specified keyword (e.g., '_REG_' or '_FWF_') from them.

    Args:
        folder (str): Path to the folder containing `.laz` files.
        keyword (str): Substring to be removed from filenames.

    Returns:
        set: A set of modified filenames with the specified keyword removed.
    """
    return {f.replace(f"_{keyword}_", "_") for f in os.listdir(folder) if f.endswith('.laz')}

def remove_unmatched_files(reg_folder, fwf_folder):
    """
    Identifies and removes files that do not have a matching counterpart in the other folder.

    This function:
    - Retrieves filenames from the regular (REG) and FWF folders.
    - Determines the base filenames (without '_REG_' or '_FWF_') to find missing pairs.
    - Removes files that do not have a corresponding match in the other folder.

    Args:
        reg_folder (str): Path to the folder containing regular point clouds.
        fwf_folder (str): Path to the folder containing FWF point clouds.

    Returns:
        None

    Logs:
        - Info messages when files are removed.
    """
    reg_files = os.listdir(reg_folder)
    fwf_files = os.listdir(fwf_folder)
    reg_base = get_base_filenames(reg_folder, "REG")
    fwf_base = get_base_filenames(fwf_folder, "FWF")
    unmatched_reg = [f for f in reg_files if f.replace("_REG_", "_") not in fwf_base]
    unmatched_fwf = [f for f in fwf_files if f.replace("_FWF_", "_") not in reg_base]
    for file in unmatched_reg:
        file_path = os.path.join(reg_folder, file)
        os.remove(file_path)
    for file in unmatched_fwf:
        file_path = os.path.join(fwf_folder, file)
        os.remove(file_path)

def eliminate_unused_species_fwf(reg_pc_folder, fwf_pc_folder, elimination_percentage, netpcsize):
    """
    Removes point clouds of underrepresented species and ensures paired REG and FWF files remain.

    This function:
    - Selects point clouds from both regular (REG) and FWF folders.
    - Determines species distribution and removes species below the elimination threshold.
    - Ensures point clouds have at least half of the target netpcsize before keeping them.
    - Maintains only those point clouds with matching REG and FWF files.

    Args:
        reg_pc_folder (str): Path to the folder containing regular point clouds.
        fwf_pc_folder (str): Path to the folder containing FWF point clouds.
        elimination_percentage (float): Minimum percentage threshold for a species to be retained.
        netpcsize (int): Minimum required points per point cloud (half of netpcsize threshold).

    Returns:
        dict: Species distribution after underrepresented species have been removed.

    Logs:
        - Info message listing species that are retained.
        - Debug messages for removed point clouds.
    """
    pointclouds = select_pointclouds(reg_pc_folder)
    fwf_pointclouds = select_pointclouds(fwf_pc_folder)
    species_list = get_species_distribution_fwf(pointclouds, fwf_pointclouds)
    species_to_use, species_distribution = eliminate_underrepresented_species(species_list, elimination_percentage)
    logging.info("Species to use: %s", species_to_use)
    pointclouds_dict = defaultdict(lambda: {"REG": None, "FWF": None})
    def extract_species(filename):
        return filename.split("_")[2]
    for pc in os.listdir(reg_pc_folder):
        if pc.endswith(".laz"):
            species = extract_species(pc)
            pc_path = os.path.join(reg_pc_folder, pc)
            pc_points = load_point_cloud(pc_path)
            if species in species_to_use and len(pc_points) > netpcsize/2:
                base_name = pc.replace("_REG_", "_")
                pointclouds_dict[base_name]["REG"] = pc
            else:
                os.remove(os.path.join(reg_pc_folder, pc))
    for fpc in os.listdir(fwf_pc_folder):
        if fpc.endswith(".laz"):
            species = extract_species(fpc)
            fpc_path = os.path.join(fwf_pc_folder, fpc)
            fpc_points = load_point_cloud(fpc_path)
            if species in species_to_use and len(fpc_points) > netpcsize/2:
                base_name = fpc.replace("_FWF_", "_")
                pointclouds_dict[base_name]["FWF"] = fpc
            else:
                os.remove(os.path.join(fwf_pc_folder, fpc))
    return species_distribution

def move_pointclouds_to_preds_fwf(reg_pc_folder, fwf_pc_folder, reg_pc_pred_folder, fwf_pc_pred_folder):
    """
    Moves every 9th REG & FWF point cloud pair per species from source folders to prediction folders.

    This function:
    - Iterates through REG point clouds and finds matching FWF counterparts.
    - Ensures each species has an evenly distributed subset for prediction.
    - Moves every 9th occurrence of a species to the prediction folder.
    - Creates destination directories if they do not exist.

    Args:
        reg_pc_folder (str): Source folder for REG point clouds.
        fwf_pc_folder (str): Source folder for FWF point clouds.
        reg_pc_pred_folder (str): Destination folder for REG point clouds.
        fwf_pc_pred_folder (str): Destination folder for FWF point clouds.

    Logs:
        - Prints an error message if a file cannot be moved.
    """
    reg_pointclouds = sorted([pc for pc in os.listdir(reg_pc_folder)])
    fwf_pointclouds = sorted([pc for pc in os.listdir(fwf_pc_folder)])
    species_counters = {}
    for reg_pc in reg_pointclouds:
        reg_parts = reg_pc.split("_")
        if len(reg_parts) < 3:
            continue
        reg_id = reg_parts[0]
        species = reg_parts[2]
        fwf_pc = next((fpc for fpc in fwf_pointclouds if fpc.startswith(reg_id + "_") and species in fpc), None)
        if fwf_pc:
            if species not in species_counters:
                species_counters[species] = 0
            species_counters[species] += 1
            if species_counters[species] % 9 == 0:
                reg_src = os.path.join(reg_pc_folder, reg_pc)
                fwf_src = os.path.join(fwf_pc_folder, fwf_pc)
                reg_dst = os.path.join(reg_pc_pred_folder, reg_pc)
                fwf_dst = os.path.join(fwf_pc_pred_folder, fwf_pc)
                os.makedirs(reg_pc_pred_folder, exist_ok=True)
                os.makedirs(fwf_pc_pred_folder, exist_ok=True)
                try:
                    shutil.move(reg_src, reg_dst)
                except Exception as e:
                    print(f"❌ Error moving {reg_pc}: {e}")
                try:
                    shutil.move(fwf_src, fwf_dst)
                except Exception as e:
                    print(f"❌ Error moving {fwf_pc}: {e}")

def select_pointclouds(pointcloud_folder):
    """
    Retrieves all point cloud file paths from the specified directory.

    Args:
        pointcloud_folder (str): Directory containing point cloud files.

    Returns:
        list: A list of full file paths for all point clouds in the directory.
    """
    pointclouds = []
    for pointcloud in os.listdir(pointcloud_folder):
        pointcloud_f = os.path.join(pointcloud_folder, pointcloud)
        pointclouds.append(pointcloud_f)
    return pointclouds

def get_species_distribution_fwf(selected_pointclouds, selected_fwf_pointclouds):
    """
    Creates an ordered list of species present in both regular and FWF point clouds.

    Args:
        selected_pointclouds (list): File paths of regular point clouds.
        selected_fwf_pointclouds (list): File paths of FWF point clouds.

    Returns:
        list: Ordered list of species names present in both datasets.
    """
    species_list = []
    for pointcloud in selected_pointclouds:
        filename = os.path.split(pointcloud)[1]
        tree_id = filename.split("_")[0]
        species = filename.split("_")[2]
        pc_num = filename.split("_")[5]
        for fwf_pointcloud in selected_fwf_pointclouds:
            fwf_filename = os.path.split(fwf_pointcloud)[1]
            fwf_tree_id = fwf_filename.split("_")[0]
            fwf_pc_num = fwf_filename.split("_")[5]
            if tree_id == fwf_tree_id and pc_num in fwf_pc_num:
                species_list.append(species)
    return species_list

def eliminate_underrepresented_species(species_list, user_spec_percentage):
    """
    Filters out species with representation below a specified percentage threshold.

    Args:
        species_list (list): List of tree species names.
        user_spec_percentage (float): Minimum percentage required for a species to be retained.

    Returns:
        list: Species that meet the representation threshold.
        list: Distribution of all species and their counts.
    """
    decently_represented_species = []
    represented_species_distribution = []
    label_counts = Counter(species_list)
    for label, count in label_counts.items():
        percentage = calculate_percentage(count, len(species_list))
        logging.info("Percentage for class %s: %s", label, percentage)
        if percentage >= user_spec_percentage:
            decently_represented_species.append(label)
            represented_species_distribution.append([label, count])
    return decently_represented_species, represented_species_distribution

def calculate_percentage(abs_num, tot_num):
    """
    Computes the percentage of a given number relative to a total.

    Args:
        abs_num (int or float): The absolute count of elements.
        tot_num (int or float): The total count of elements.

    Returns:
        float: The percentage representation of abs_num in tot_num.
    """
    if abs_num == 0:
        return 0.0
    else:
        return (abs_num / tot_num) * 100
    
def get_maximum_distribution(spec_distr):
    """
    Determines the maximum sample count among species in the dataset.

    Args:
        spec_distr (list of lists): A list where each element is [species_name, count].

    Returns:
        int: The highest sample count among all species, rounded up.
    """
    max = 0
    for row in spec_distr:
        label = row[0]
        distr = row[1]
        if distr >= max:
            max = distr
    return np.ceil(max)

def get_species_dependent_pointcloud_pairs_fwf(selected_pointclouds, selected_fwf_pointclouds):
    """
    Matches regular point clouds with their corresponding FWF point clouds based on tree ID and point cloud number.

    Args:
        selected_pointclouds (list of str): List of file paths for regular point clouds.
        selected_fwf_pointclouds (list of str): List of file paths for FWF point clouds.

    Returns:
        list of list: A list where each element is a pair [regular_pointcloud, fwf_pointcloud] 
                      representing a matched species-dependent point cloud pair.
    """
    species_pairs = []
    for pointcloud in selected_pointclouds:
        filename = os.path.split(pointcloud)[1]
        tree_id = filename.split("_")[0]
        pc_num = filename.split("_")[5]
        for fwf_pointcloud in selected_fwf_pointclouds:
            fwf_filename = os.path.split(fwf_pointcloud)[1]
            fwf_tree_id = fwf_filename.split("_")[0]
            fwf_pc_num = fwf_filename.split("_")[5]
            if tree_id == fwf_tree_id and pc_num in fwf_pc_num:
                species_pairs.append([pointcloud, fwf_pointcloud])
    return species_pairs

def augment_selection_fwf(pointclouds, fwf_pointclouds, max_pc_scale, pc_path_selection, fwf_path_selection, species_distribution):
    """
    Augments point clouds with FWF data to balance species representation.

    This function ensures that underrepresented species in the dataset are 
    augmented up to the level of the most represented species.

    Args:
        pointclouds (list of str): List of file paths for regular point clouds.
        fwf_pointclouds (list of str): List of file paths for FWF point clouds.
        max_pc_scale (float): Maximum scaling factor applied during augmentation.
        pc_path_selection (str): Target directory for augmented regular point clouds.
        fwf_path_selection (str): Target directory for augmented FWF point clouds.
        species_distribution (list of lists): A list containing species distribution information.

    Returns:
        None: Augmented point clouds are saved directly in the specified directories.
    """
    if check_if_data_is_augmented_already(pointclouds) == False:
        max_representation = get_maximum_distribution(species_distribution)
        species_pc_pairs = get_species_dependent_pointcloud_pairs_fwf(pointclouds, fwf_pointclouds)
        augment_species_pointclouds_fwf(species_pc_pairs, max_representation, species_distribution, max_pc_scale, pc_path_selection, fwf_path_selection)
    else:
        logging.info("Augmented data found, loading!")

def check_if_data_is_augmented_already(pointclouds):
    """
    Checks if any of the point clouds in the provided list are augmented 
    (based on the presence of an augmentation identifier in the filename).

    Args:
        pointclouds (list of str): List of file paths for point clouds.

    Returns:
        bool: Returns True if any point cloud file is an augmentation (identified by 
              augmentation number in the filename), otherwise False.
    """
    for cloud in pointclouds:
        cloud_name = cloud.split("/")[-1].split(".")[0]
        augnum = cloud_name.split("_")[-1]
        if str(1) in augnum:
            return True
        elif str(2) in augnum:
            return True
        elif str(3) in augnum:
            return True
        elif str(4) in augnum:
            return True
        elif str(5) in augnum:
            return True
        elif str(6) in augnum:
            return True
        elif str(7) in augnum:
            return True
        elif str(8) in augnum:
            return True
        elif str(9) in augnum:
            return True
        else:
            pass
    return False

def augment_species_pointclouds_fwf(species_pc_pairs, max_representation, species_distribution, max_scale, pc_path_selection, fwf_path_selection):
    """
    Augments regular and FWF point clouds through rotation, scaling, mirroring, jittering, and noise addition.

    Args:
        species_pc_pairs (list of list): List of pairs of regular and FWF point cloud file paths.
        max_representation (int): Maximum number of samples for any species in the dataset.
        species_distribution (dict or list): Species distribution across the dataset.
        max_scale (float): Scaling factor range for augmentation.
        pc_path_selection (str): Destination path for augmented regular point clouds.
        fwf_path_selection (str): Destination path for augmented FWF point clouds.
    
    Process:
        - Extracts species information and computes necessary augmentation factor.
        - Loads and processes point cloud data for augmentation.
        - Applies transformations: rotation, scaling, mirroring, jittering, and noise addition.
        - Saves the augmented point clouds in the specified paths.
    """
    pair_index = 0
    for species_pairs in species_pc_pairs:
        current_species = get_species_for_pairs_list(species_pairs)
        current_species_amount = get_abs_num(current_species, species_distribution)
        upscale_fac = get_upscale_factor(current_species_amount, max_representation)
        current_reg_pc = species_pairs[0]
        filename_reg_full = os.path.split(current_reg_pc)[1]
        filename_reg_ext = filename_reg_full.split(".")[-1]
        filename_reg_f = filename_reg_full.split(".")[0]
        filenameparts_reg = filename_reg_f.split("_")[:-1]
        filename_reg = filenameparts_reg[0] + "_" + filenameparts_reg[1] + "_" + filenameparts_reg[2] + "_" + filenameparts_reg[3] + "_" + filenameparts_reg[4] + "_" + filenameparts_reg[5] + "_" + filenameparts_reg[6] 
        current_fwf_pc = species_pairs[1]
        filename_fwf_full = os.path.split(current_fwf_pc)[1]
        filename_fwf_ext = filename_fwf_full.split(".")[-1]
        filename_fwf_f = filename_fwf_full.split(".")[0]
        filenameparts_fwf = filename_fwf_f.split("_")[:-1]
        filename_fwf = filenameparts_fwf[0] + "_" + filenameparts_fwf[1] + "_" + filenameparts_fwf[2] + "_" + filenameparts_fwf[3] + "_" + filenameparts_fwf[4] + "_" + filenameparts_fwf[5] + "_" + filenameparts_fwf[6]
        reg_points, reg_pc = load_point_cloud_and_file(species_pairs[0])
        fwf_points, fwf_pc = load_point_cloud_and_file(species_pairs[1])
        logging.debug("Number of points reg: %s", len(reg_points))
        logging.debug("Number of points reg: %s", len(fwf_points))
        for i in range(0, int(upscale_fac)*4):
            pair_index+=1
            outFile_r = lp.LasData(reg_pc.header)
            outFile_f = lp.LasData(fwf_pc.header)
            outFile_r.vlrs = reg_pc.vlrs
            outFile_f.vlrs = fwf_pc.vlrs
            angle = pick_random_angle(np.random.randint(1, 360))
            new_reg_points = reg_points
            new_fwf_points = fwf_points
            exported_points_reg = center_pointcloud_o3d(new_reg_points)
            exported_points_fwf = center_pointcloud_o3d(new_fwf_points)
            rotated_reg_pc = rotate_point_cloud(exported_points_reg, angle)
            rotated_fwf_pc = rotate_point_cloud(exported_points_fwf, angle)
            scale_factors = np.random.uniform(1 - max_scale, 1 + max_scale, size=3)
            scaled_rotated_reg_pc = scale_point_cloud(rotated_reg_pc, scale_factors)
            scaled_rotated_fwf_pc = scale_point_cloud(rotated_fwf_pc, scale_factors)
            scaled_rotated_reg_pc += np.random.uniform(-0.045, 0.045, scaled_rotated_reg_pc.shape)
            scaled_rotated_fwf_pc += np.random.uniform(-0.045, 0.045, scaled_rotated_fwf_pc.shape)
            if np.random.rand() > 0.5:
                scaled_rotated_reg_pc[:, 0] *= -1  
                scaled_rotated_fwf_pc[:, 0] *= -1
            if np.random.rand() > 0.5:
                scaled_rotated_reg_pc[:, 1] *= -1  
                scaled_rotated_fwf_pc[:, 1] *= -1
            noise_std = 0.005  
            scaled_rotated_reg_pc += np.random.normal(0, noise_std, scaled_rotated_reg_pc.shape)
            scaled_rotated_fwf_pc += np.random.normal(0, noise_std, scaled_rotated_fwf_pc.shape)
            jittered_shuffled_reg_pc = np.random.permutation(scaled_rotated_reg_pc)
            jittered_shuffled_fwf_pc = np.random.permutation(scaled_rotated_fwf_pc)
            adjust_las_header(outFile_r, jittered_shuffled_reg_pc)
            adjust_las_header(outFile_f, jittered_shuffled_fwf_pc)
            outFile_r.x = jittered_shuffled_reg_pc[:, 0]
            outFile_r.y = jittered_shuffled_reg_pc[:, 1]
            outFile_r.z = jittered_shuffled_reg_pc[:, 2]
            outFile_f.x = jittered_shuffled_fwf_pc[:, 0]
            outFile_f.y = jittered_shuffled_fwf_pc[:, 1]
            outFile_f.z = jittered_shuffled_fwf_pc[:, 2]
            new_filename_reg = f"{filename_reg}_aug0{pair_index}{i}.{filename_reg_ext}"
            new_filename_fwf = f"{filename_fwf}_aug0{pair_index}{i}.{filename_fwf_ext}"
            savepath_reg = os.path.join(pc_path_selection, new_filename_reg)
            savepath_fwf = os.path.join(fwf_path_selection, new_filename_fwf)
            save_point_cloud(savepath_reg, reg_pc, outFile_r)
            save_point_cloud(savepath_fwf, fwf_pc, outFile_f)
            if main_utils.contains_full_waveform_data(savepath_fwf):
                logging.debug("The pointcloud still has FWF data after augmentation!")
            else:
                logging.debug("FWF data has been lost!")

def center_pointcloud_o3d(pointcloud):
    """
    Centers the point cloud by subtracting the mean of the points from each point.
    
    Args:
    pointcloud: The point cloud to be centered, expected as a numpy array of shape (N, 3),
                where N is the number of points and each point is represented by its (x, y, z) coordinates.
    
    Returns:
    numpy.ndarray: A centered point cloud as a numpy array with the same shape as the input.
    """
    center = np.mean(pointcloud, axis=0)
    centered_points = pointcloud - center
    return np.asarray(centered_points)

def get_species_for_pairs_list(species_pairs):
    """
    Retrieves species for the first entry in a point cloud pair.

    Args:
    species_pairs: List of pairs of regular and FWF point cloud file paths.

    Returns:
    species: The species of the regular point cloud.
    """
    first_spec_pair = species_pairs[0]
    filename = os.path.split(first_spec_pair)[1]
    species = filename.split("_")[2]
    return species

def get_abs_num(species, species_distribution):
    """
    Retrieves the absolute number of samples for a species.

    Args:
    species: Species name.
    species_distribution: Distribution of species across the dataset.

    Returns:
    abs_num: Absolute number of samples of the specified species in the dataset.
    """
    abs_num = None
    for spec_num in species_distribution:
        current_spec = spec_num[0]
        if current_spec == species:
            abs_num = spec_num[1]
            break
    if abs_num is None:
        raise ValueError(f"Species '{species}' not found in species_distribution!")
    return abs_num

def get_upscale_factor(abs_num, max):
    """
    Retrieves the upscale factor for a species to balance the dataset.

    Args:
    abs_num: Absolute number of samples of a species in the dataset.
    max: Highest number of samples for any species in the dataset.

    Returns:
    np.round(fac): Integer upscaling factor to balance the species with the highest representation.
    """
    fac = max / abs_num
    return np.round(fac)

def load_point_cloud_and_file(file_path):
    """
    Retrieves las points and the las file from a filepath.

    Args:
    file_path: Filepath of a las point cloud.

    Returns:
    points: Array of individual points of the point cloud.
    las_file: Las file instance with Header and VLRs.
    """
    try:
        las_file = lp.read(file_path)
        points = np.vstack((las_file.x, las_file.y, las_file.z)).transpose()
    except OSError as e:
        logging.error("Error loading file %s: %s", file_path, e)
        raise
    return points, las_file

def pick_random_angle(index):
    """
    Picks a random angle between 0 and 360 degrees, incements in 15 degree steps.

    Returns:
    random_angle: Generated angle in degrees.
    """
    return index

def rotate_point_cloud(point_cloud, angle):
    """
    Rotates a point cloud around the z-axis for a specified angle.

    Args:
    point_cloud: Las point cloud points array.
    angle: Rotation angle around z-axis.

    Returns:
    rotated_point_cloud: Las point cloud points array.
    """
    angle_rad = np.radians(angle)
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                  [np.sin(angle_rad), np.cos(angle_rad), 0],
                  [0, 0, 1]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.copy())
    pcd.rotate(R, center=pcd.get_center())
    return np.asarray(pcd.points)

def scale_point_cloud(point_cloud, scale_factors):
    """
    Scale a point cloud by a specified factor.

    Args:
    point_cloud: Las point cloud points array.
    scale_factors: Scaling factor.

    Returns:
    scaled_point_cloud: Scaled array of las point cloud points.
    """
    scaled_point_cloud = point_cloud * scale_factors
    return scaled_point_cloud

def adjust_las_header(las, points):
    """
    Adjusts a las file header to allow scaling operations.

    Args:
    las: Las file instance.
    points: Array of point cloud points.
    """
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    min_z, max_z = np.min(points[:, 2]), np.max(points[:, 2])
    new_offset = [min_x, min_y, min_z]
    original_scale = np.array(las.header.scale)
    las.header.offset = new_offset
    las.header.scale = original_scale
    logging.debug("Updated Scale: %s", original_scale)
    logging.debug("Updated Offset: %s", new_offset)

def save_point_cloud(file_path, orig_las_file, outFile):
    """
    Saves a point cloud to the specified path.

    Args:
    file_path: Savepath for the point cloud.
    orig_las_file: Laspy point cloud file instance.
    outFile: Laspy point cloud file instance.
    """
    if orig_las_file.evlrs:
        outFile.evlrs = orig_las_file.evlrs.copy()
        outFile.vlrs = orig_las_file.vlrs.copy()
        outFile.intensity = orig_las_file.intensity.copy()
        outFile.write(file_path)
    else:
        outFile.vlrs = orig_las_file.vlrs.copy()
        outFile.intensity = orig_las_file.intensity.copy()
        outFile.write(file_path)

def get_colored_images_generated(las_working_folder, img_working_folder):
    """
    Checks if all images have been generated for each individual tree point cloud.

    Args:
    las_working_folder: Filepath to las point clouds.
    img_working_folder: Savepath for generated images.

    Returns:
    True/False
    """
    las_list = []
    img_list = []
    for pointcloud in os.listdir(las_working_folder):
        las_list.append(pointcloud)
    for image in os.listdir(img_working_folder):
        img_list.append(image)
    imlistlength = len(img_list)/2
    if imlistlength == len(las_list) or imlistlength > 0:
        logging.info("Images have already been generated, skipping!")
        return True
    else:
        return False

def get_maximum_unscaled_image_size(las_working_folder, img_working_folder):
    """
    Computes the maximum unscaled image size for colored depth images based on point cloud data.

    Args:
    las_working_folder: Filepath to las point clouds.
    img_working_folder: Savepath for generated images.

    Returns:
    max_image_size: Maximum image size before padding.
    """
    if get_colored_images_generated(las_working_folder, img_working_folder) == False:
        image_sizes = []
        for pointcloud in tqdm(os.listdir(las_working_folder), desc="Computing max size and height:"):
            pointcloud_path = main_utils.join_paths(las_working_folder, pointcloud)
            pc = lp.read(pointcloud_path)
            voxels, abs_height = create_voxel_grid_from_las(pc)
            vox_pos_list, max_img_size = get_voxel_positions(voxels)
            zero_frontal_image, zero_sideways_image = create_empty_images(max_img_size)
            frontal_image, sideways_image = fill_and_scale_empty_images(vox_pos_list, zero_frontal_image, zero_sideways_image)
            max_image_size_before_padding = frontal_image.shape[0]
            image_sizes.append(max_image_size_before_padding)
        max_image_size = max(image_sizes)
        return max_image_size
    else:
        return 0

def generate_colored_images(IMG_SIZE, las_working_folder, img_working_folder, abs_max_img_size):
    """
    Generates colored depth images from LAS point clouds and saves them.

    This function processes each LAS point cloud in the specified working folder, converts the point clouds 
    into voxel grids, and then creates colored depth images (both frontal and sideways views). The images 
    are scaled and saved in the specified image working folder.

    Args:
    IMG_SIZE (int): The user-specified network input image size (224).
    las_working_folder (str): Filepath to the folder containing the LAS point cloud files.
    img_working_folder (str): Folder path where the generated colored images will be saved.
    abs_max_img_size (int): The absolute maximum size of images before any padding is applied.

    Returns:
    None: This function does not return anything. It saves the generated images to the `img_working_folder`.
    """
    if get_colored_images_generated(las_working_folder, img_working_folder) == False:
        pcid = 0
        for pointcloud in os.listdir(las_working_folder):
            pointcloud_path = main_utils.join_paths(las_working_folder, pointcloud)
            pc = lp.read(pointcloud_path)
            tree_id = pointcloud.split("_")[0]
            species = pointcloud.split("_")[2]
            method = pointcloud.split("_")[3]
            date = pointcloud.split("_")[4]
            ind_id = pointcloud.split("_")[5]
            leaf_cond = pointcloud.split("_")[6]
            augnum = pointcloud.split("_")[7].split(".")[0]
            voxels, abs_height = create_voxel_grid_from_las(pc)
            vox_pos_list, max_img_size = get_voxel_positions(voxels)
            zero_frontal_image, zero_sideways_image = create_empty_images(max_img_size)
            frontal_image, sideways_image = fill_and_scale_empty_images(vox_pos_list, zero_frontal_image, zero_sideways_image)
            save_voxelized_pointcloud_images(IMG_SIZE, frontal_image, sideways_image, tree_id, species, method, date, ind_id, leaf_cond, img_working_folder, zero_frontal_image, str(pcid), augnum, abs_max_img_size)
            pcid+=1
            logging.info("Generated frontal and sideways views of point cloud %s!", pcid)
    else:
        pass

def get_colored_images_generated(las_working_folder, img_working_folder):
    """
    Checks if colored images have been generated for each individual tree point cloud in the given directories.

    This function compares the number of LAS point clouds and their corresponding generated images. If the 
    number of images matches half the number of point clouds (since each point cloud generates two images), 
    the function returns `True`, indicating that the images have already been generated. Otherwise, it returns `False`.

    Args:
    las_working_folder (str): Filepath to the folder containing the LAS point cloud files.
    img_working_folder (str): Filepath to the folder containing the generated image files.

    Returns:
    bool: `True` if all images have already been generated, `False` otherwise.
    """
    las_list = []
    img_list = []
    for pointcloud in os.listdir(las_working_folder):
        las_list.append(pointcloud)
    for image in os.listdir(img_working_folder):
        img_list.append(image)
    imlistlength = len(img_list)/2
    if imlistlength == len(las_list) or imlistlength > 0:
        logging.info("Images have already been generated, skipping!")
        return True
    else:
        return False
    
def create_voxel_grid_from_las(pointcloud):
    """
    Creates a voxel grid from a point cloud, with a specified voxel size of 0.04, and calculates the absolute height.

    This function converts a LAS point cloud into a voxel grid representation by first extracting the XYZ coordinates 
    from the point cloud, then rotating the point cloud for alignment. It uses Open3D to create a voxel grid with a voxel size of 0.04. 
    Additionally, it computes the absolute height of the point cloud by finding the difference between the maximum and minimum 
    Z-values of the points.

    Args:
    pointcloud (object): A point cloud object containing the X, Y, and Z coordinates.

    Returns:
    tuple: A tuple containing:
        - vox_grid: A voxel grid created from the point cloud with a voxel size of 0.04.
        - absolute_height: The absolute height of the point cloud, calculated as the difference between the maximum and minimum Z-values.
    """
    points = np.vstack([pointcloud.x, pointcloud.y, pointcloud.z]).transpose()
    pcd_las_o3d = o3d.geometry.PointCloud()
    pcd_las_o3d.points = o3d.utility.Vector3dVector(points)
    R = pcd_las_o3d.get_rotation_matrix_from_xyz((-1.5, 0, 0))
    pcd_las_o3d.rotate(R, center=(0, 0, 0))
    vox_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_las_o3d, voxel_size=0.04)
    absolute_height = np.max(points[:, 2]) - np.min(points[:, 2])
    return vox_grid, absolute_height

def get_voxel_positions(voxel_grid):
    """
    Gets indices for individual voxels.

    Args:
    voxel_grid: Voxel grid.

    Returns:
    voxel_position_list: List of individual voxel positions.
    maximum_image_size: Maximum height/width for an image generated from the voxel grid.
    """
    vox_list = voxel_grid.get_voxels()
    voxel_position_list = []
    for voxel in vox_list:
        voxel_position_list.append(voxel.grid_index.tolist())
    maximum_grid_index = np.max(voxel_position_list)
    maximum_image_size = maximum_grid_index + 3
    logging.debug("%s", maximum_image_size)
    return voxel_position_list, maximum_image_size

def create_empty_images(maximum_image_size):
    """
    Creates empty image arrays of a given size.

    Args:
    maximum_image_size: Maximum height/width for an image generated from the voxel grid.

    Returns:
    empty_image_frontal: Zero-array of dimensions maximum_image_size X maximum_image_size
    empty_image_sideways: Zero-array of dimensions maximum_image_size X maximum_image_size
    """
    empty_image_frontal = np.zeros((maximum_image_size, maximum_image_size), int)
    empty_image_sideways = np.zeros((maximum_image_size, maximum_image_size), int)
    return empty_image_frontal, empty_image_sideways

def fill_and_scale_empty_images(voxel_positions_list, empty_image_frontal, empty_image_topdown):
    """
    Fills empty image arrays with values according to the voxel positions.

    Args:
    voxel_positions_list: List of individual voxel positions.
    empty_image_frontal: Zero-array of dimensions maximum_image_size X maximum_image_size
    empty_image_topdown: Zero-array of dimensions maximum_image_size X maximum_image_size

    Returns:
    image_frontal: Corrected frontal view (Y-Z plane).
    image_topdown: Corrected top-down view (X-Y plane).
    """
    for voxel_position in voxel_positions_list:
        voxel_position_x = voxel_position[0]
        voxel_position_y = voxel_position[1]
        voxel_position_z = voxel_position[2]
        empty_image_frontal[voxel_position_y + 1, voxel_position_z + 1] += 1
        empty_image_topdown[voxel_position_x + 1, voxel_position_z + 1] += 1
    image_frontal = np.interp(empty_image_frontal, (empty_image_frontal.min(), empty_image_frontal.max()), (0, 255))
    image_topdown = np.interp(empty_image_topdown, (empty_image_topdown.min(), empty_image_topdown.max()), (0, 255))
    image_frontal = np.rot90(image_frontal, k=2, axes=(1, 0))  
    image_topdown = np.rot90(image_topdown, k=1, axes=(1, 0))  
    return image_frontal, image_topdown

def pad_image(img, pad_t, pad_r, pad_l):
    """
    Pads an image with zeros around all sides by the specified amount.

    This function adds padding to an image by extending its boundaries with zero values. 
    The padding is applied to the top, right, and left sides according to the user-specified values. 
    The bottom padding is implicitly handled by adjusting the height with respect to other sides.

    Args:
    img (ndarray): The input image array (2D).
    pad_t (int): Number of rows to add as padding on the top of the image.
    pad_r (int): Number of columns to add as padding on the right side of the image.
    pad_l (int): Number of columns to add as padding on the left side of the image.

    Returns:
    ndarray: The padded image array with zeros added on all sides.
    """
    height, width = img.shape
    pad_left = np.zeros((height, int(pad_l)))
    img = np.concatenate((pad_left, img), axis = 1)
    pad_up = np.zeros((int(pad_t), int(pad_l) + width))
    img = np.concatenate((pad_up, img), axis = 0)
    pad_right = np.zeros((height + int(pad_t), int(pad_r)))
    img = np.concatenate((img, pad_right), axis = 1)
    return img

def center_image(img, abs_max_img_size, is_topdown=False):
    """
    Crops an image to its contents' bounding box and pads it to be centered.

    The function first finds the bounding box of the non-zero pixels and crops the image 
    accordingly. Then it pads the cropped image to a specified size, ensuring it is centered.
    - If `is_topdown` is False (frontal view), the object stays at the bottom of the image.
    - If `is_topdown` is True (top-down view), the object is fully centered in the image.

    Args:
    img (ndarray): The input image array (2D).
    abs_max_img_size (int): The target size for the padded image (final image size).
    is_topdown (bool): If True, centers the object in the image, if False, places it at the bottom.

    Returns:
    ndarray: The cropped and padded image array, centered as per the provided conditions.
    """
    col_sum = np.where(np.sum(img, axis=0) > 0)
    row_sum = np.where(np.sum(img, axis=1) > 0)
    if len(row_sum[0]) == 0 or len(col_sum[0]) == 0:
        return img
    y1, y2 = row_sum[0][0], row_sum[0][-1] + 1
    x1, x2 = col_sum[0][0], col_sum[0][-1] + 1
    cropped_image = img[y1:y2, x1:x2]
    pad_left = (abs_max_img_size - cropped_image.shape[1]) // 2
    pad_right = abs_max_img_size - cropped_image.shape[1] - pad_left
    if is_topdown:
        pad_top = (abs_max_img_size - cropped_image.shape[0]) // 2
        pad_bottom = abs_max_img_size - cropped_image.shape[0] - pad_top
    else:
        pad_top = abs_max_img_size - cropped_image.shape[0]
        pad_bottom = 0
    centered_image = np.pad(
        cropped_image, ((pad_top, pad_bottom), (pad_left, pad_right)), 
        mode='constant', constant_values=0
    )
    return centered_image

def save_colored_image(image, id, species, method, date, ind_id, leaf_cond, angle, pcid, augnum, SAVE_DIR):
    """
    Saves an image array as a .tiff file.

    This function takes the input image and saves it as a .tiff file in the specified directory.
    It uses a custom colormap for visualization and includes various metadata in the file name.

    Args:
    image (ndarray): Image array to be saved.
    id (str): Tree ID from the source point cloud.
    species (str): Tree species of the source point cloud.
    method (str): Capture method used for the point cloud.
    date (str): Capture date of the point cloud.
    ind_id (str): Individual point cloud ID.
    leaf_cond (str): Leaf condition of the source point cloud (e.g., LEAF-ON or LEAF-OFF).
    angle (str): Frontal or sideways angle of the point cloud.
    pcid (str): Point cloud ID.
    augnum (str): Augmentation number if the point cloud has been augmented.
    SAVE_DIR (str): Directory path to save the image.

    Returns:
    None: The image is saved directly to the specified location.
    """
    colors = [(0, 0, 0)] + [plt.cm.twilight(i / 255) for i in range(1, 256)]
    custom_cmap = mcolors.ListedColormap(colors, name="custom_twilight")
    norm = plt.Normalize(vmin=image.min(), vmax=image.max())
    image = custom_cmap(norm(image))
    save_path = os.path.join(SAVE_DIR + "/" + str(id) + "_" + species + "_" + method + "_" + date + "_" + str(ind_id) + "_" + leaf_cond + "_" + angle + "_" + pcid + "_" + augnum + ".tiff")
    plt.imsave(save_path, image)
    
def save_voxelized_pointcloud_images(IMG_SIZE, image_frontal, image_sideways, id, species, method, date, ind_id, leaf_cond, SAVE_DIR, empty_image_frontal, pointcloud_id, augmentation_number, abs_max_img_size):
    """
    Save frontal and top-down images with proper centering.
    - Frontal image (Y-Z): No bottom padding.
    - Top-down image (X-Y): Fully centered.
    """
    image_frontal_to_save = center_image(image_frontal, abs_max_img_size, is_topdown=False)
    image_frontal_resized = cv2.resize(image_frontal_to_save, (IMG_SIZE, IMG_SIZE))
    save_colored_image(image_frontal_resized, id, species, method, date, ind_id, leaf_cond, "frontal", pointcloud_id, augmentation_number, SAVE_DIR)
    image_sideways_to_save = center_image(image_sideways, abs_max_img_size, is_topdown=True)
    image_sideways_resized = cv2.resize(image_sideways_to_save, (IMG_SIZE, IMG_SIZE))
    save_colored_image(image_sideways_resized, id, species, method, date, ind_id, leaf_cond, "sideways", pointcloud_id, augmentation_number, SAVE_DIR)
    
def read_image(filepath):
    """
    Reads an image file into an image array.

    Args:
    filepath: Filepath of the file to open.

    Returns:
    image_array: Image array with RGB channels.
    """
    image = Image.open(filepath).convert('RGB')
    image_array = np.array(image)
    return image_array

def get_user_specified_data_fwf(pc_path, fwf_path, img_path, cap_sel, grow_sel):
    """
    Selects data based on user-specifications.

    Args:
    pc_path: Working directory point cloud path.
    fwf_path: Working directory FWF point cloud path.
    img_path: Working directory image path.
    cap_sel: User-specified acquisition method.
    grow_sel User-specified leaf-condition.

    Returns:
    selected_pointclouds: Point clouds selected based on the specifications.
    selected_fwf_pointclouds: FWF point clouds selected based on the specifications.
    selected_images: Images selected based on the specifications.
    """
    selected_pointclouds = select_data_according_to_specifications(cap_sel, grow_sel, pc_path)
    selected_fwf_pointclouds = select_data_according_to_specifications(cap_sel, grow_sel, fwf_path)
    selected_images = select_data_according_to_specifications(cap_sel, grow_sel, img_path)
    return selected_pointclouds, selected_fwf_pointclouds, selected_images

def select_data_according_to_specifications(capsel, grosel, path):
    """
    Selects files which filenames include user-specified arguments.

    Args:
    capsel: User-specified selection of capture methods.
    grosel: User-specified selection of leaf conditions.
    path: Directory where files will be checked for naming matches.

    Returns:
    path_list: List of paths with files which names include the user-specified selection criteria.
    """
    path_list = get_list_of_selected_files(capsel, grosel, path)
    return path_list

def get_list_of_selected_files(capsel, grosel, search_directory):
    """
    Selects files which filenames include user-specified arguments.

    Args:
    capsel: User-specified selection of capture methods.
    grosel: User-specified selection of leaf conditions.
    search_directory: Directory where files will be checked for naming matches.

    Returns:
    pathlib: List of paths with files which names include the user-specified selection criteria.
    """
    pathlib = []
    for file in os.listdir(search_directory):
        if capsel == "ALL":
            if grosel == "ALL":
                if "ALS" in file or "TLS" in file or "ULS" in file:
                    if "LEAF-ON" in file or "LEAF-OFF" in file:
                        filepath = main_utils.join_paths(search_directory, file)
                        pathlib.append(filepath)
                    else:
                        pass
                else:
                    pass
            elif grosel == "LEAF-ON":
                if "ALS" in file or "TLS" in file or "ULS" in file:
                    if "LEAF-ON" in file:
                        filepath = main_utils.join_paths(search_directory, file)
                        pathlib.append(filepath)
                    else:
                        pass
                else:
                    pass
            elif grosel == "LEAF-OFF":
                if "ALS" in file or "TLS" in file or "ULS" in file:
                    if "LEAF-OFF" in file:
                        filepath = main_utils.join_paths(search_directory, file)
                        pathlib.append(filepath)
                    else:
                        pass
                else:
                    pass
            else:
                pass
        elif capsel == "ALS":
            if grosel == "ALL":
                if "ALS" in file:
                    if "LEAF-ON" in file or "LEAF-OFF" in file:
                        filepath = main_utils.join_paths(search_directory, file)
                        pathlib.append(filepath)
                    else:
                        pass
                else:
                    pass
            elif grosel == "LEAF-ON":
                if "ALS" in file:
                    if "LEAF-ON" in file:
                        filepath = main_utils.join_paths(search_directory, file)
                        pathlib.append(filepath)
                    else:
                        pass
                else:
                    pass
            elif grosel == "LEAF-OFF":
                if "ALS" in file:
                    if "LEAF-OFF" in file:
                        filepath = main_utils.join_paths(search_directory, file)
                        pathlib.append(filepath)
                    else:
                        pass
                else:
                    pass
            else:
                pass
        elif capsel == "TLS":
            if grosel == "ALL":
                if "TLS" in file:
                    if "LEAF-ON" in file or "LEAF-OFF" in file:
                        filepath = main_utils.join_paths(search_directory, file)
                        pathlib.append(filepath)
                    else:
                        pass
                else:
                    pass
            elif grosel == "LEAF-ON":
                if "TLS" in file:
                    if "LEAF-ON" in file:
                        filepath = main_utils.join_paths(search_directory, file)
                        pathlib.append(filepath)
                    else:
                        pass
                else:
                    pass
            elif grosel == "LEAF-OFF":
                if "TLS" in file:
                    if "LEAF-OFF" in file:
                        filepath = main_utils.join_paths(search_directory, file)
                        pathlib.append(filepath)
                    else:
                        pass
                else:
                    pass
            else:
                pass
        elif capsel == "ULS":
            if grosel == "ALL":
                if "ULS" in file:
                    if "LEAF-ON" in file or "LEAF-OFF" in file:
                        filepath = main_utils.join_paths(search_directory, file)
                        pathlib.append(filepath)
                    else:
                        pass
                else:
                    pass
            elif grosel == "LEAF-ON":
                if "ULS" in file:
                    if "LEAF-ON" in file:
                        filepath = main_utils.join_paths(search_directory, file)
                        pathlib.append(filepath)
                    else:
                        pass
                else:
                    pass
            elif grosel == "LEAF-OFF":
                if "ULS" in file:
                    if "LEAF-OFF" in file:
                        filepath = main_utils.join_paths(search_directory, file)
                        pathlib.append(filepath)
                    else:
                        pass
                else:
                    pass
            else:
                pass
    return pathlib

def load_point_cloud(file_path):
    """
    Retrieves las points from a filepath.

    Args:
    file_path: Filepath of a las point cloud.

    Returns:
    points: Array of individual points of the point cloud.
    """
    try:
        las_file = lp.read(file_path)
        points = np.vstack((las_file.x, las_file.y, las_file.z)).transpose()
    except OSError as e:
        logging.error("Error loading file %s: %s", file_path, e)
        raise
    return points

def center_point_cloud(points_list):
    """
    Centers a point cloud around the coordinate origin.

    Args:
    points_list: Array of las point cloud points.

    Returns:
    center_points: Centered array of las point cloud points.
    """
    center_points = []
    for points in points_list:
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        center_points.append(centered_points)
    return center_points

def non_uniform_grid_partition(point_cloud, num_clusters=8):
    """
    Partitions a point cloud into non-uniform grid cells using adaptive clustering (K-Means).
    
    Args:
        point_cloud (numpy array): N x 3 array of points.
        num_clusters (int): Number of adaptive clusters.

    Returns:
        cluster_labels (numpy array): Cluster labels for each point.
    """
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(point_cloud)
    return cluster_labels

def farthest_point_sampling(points, num_samples):
    """
    Applies standard Farthest Point Sampling (FPS) to a given point cloud.

    Args:
        points (numpy array): N x 3 array of points.
        num_samples (int): Number of points to sample.

    Returns:
        sampled_points (numpy array): Downsampled N' x 3 point cloud.
    """
    farthest_pts = np.zeros((num_samples, 3))
    farthest_pts[0] = points[np.random.randint(len(points))]  
    distances = cdist(points, farthest_pts[0].reshape(1, 3)).squeeze()

    for i in range(1, num_samples):
        idx = np.argmax(distances)  
        farthest_pts[i] = points[idx]
        new_dist = cdist(points, farthest_pts[i].reshape(1, 3)).squeeze()
        distances = np.minimum(distances, new_dist)  

    return farthest_pts

def resample_pointcloud(point_cloud, num_samples, iteration, num_clusters=8):
    """
    Resamples a point cloud using Non-Uniform Grid Sampling + Farthest Point Sampling (NGFPS).

    This function applies a combination of non-uniform grid partitioning and farthest point sampling 
    to resample a point cloud. The goal is to retain representative points from different regions of 
    the point cloud, with adjustments for uneven density.

    Args:
    point_cloud (ndarray): The input point cloud array to be resampled.
    num_samples (int): The desired number of points in the output resampled point cloud.
    iteration (int): The current iteration of the sampling process (used for logging or stopping criteria).
    num_clusters (int, optional): The number of clusters for grid partitioning. Defaults to 8.

    Returns:
    ndarray: The downsampled point cloud with the specified number of samples.
    """
    cluster_labels = non_uniform_grid_partition(point_cloud, num_clusters)
    sampled_points = []
    total_points_collected = 0
    for cluster_idx in range(num_clusters):
        cluster_points = point_cloud[cluster_labels == cluster_idx]
        if len(cluster_points) == 0:
            continue  
        num_cluster_samples = max(3, int((len(cluster_points) / len(point_cloud)) * num_samples))
        num_cluster_samples = min(num_samples - total_points_collected, num_cluster_samples)
        if len(cluster_points) > num_cluster_samples:
            sampled_cluster = farthest_point_sampling(cluster_points, num_cluster_samples)
        else:
            sampled_cluster = cluster_points  
        sampled_points.append(sampled_cluster)
        total_points_collected += sampled_cluster.shape[0]
        if total_points_collected >= num_samples:  
            break
    downsampled_pc = np.vstack(sampled_points)
    if downsampled_pc.shape[0] < num_samples:
        missing_points = num_samples - downsampled_pc.shape[0]
        pad_indices = np.random.choice(downsampled_pc.shape[0], missing_points, replace=True)
        downsampled_pc = np.vstack([downsampled_pc, downsampled_pc[pad_indices]])
    return downsampled_pc

def process_single_pointcloud_fwf(args):
    """
    Computes numerical features for a single point cloud and its corresponding FWF file.
    
    This function processes a single point cloud file and its corresponding FWF file, 
    calculates combined metrics for both, and returns them as a NumPy array.
    
    Args:
    args (tuple): A tuple containing the paths to the point cloud and FWF file, 
                  the current index, and the total count of files.
                  (pointcloud_path, fwf_path, index, total_count)
    
    Returns:
    arrmetrics (ndarray): A NumPy array containing the computed numerical features.
    """
    pointcloud_path, fwf_path, idx, total_count = args
    logging.info(f"Processing point cloud {idx+1}/{total_count}")
    las_points = load_point_cloud(pointcloud_path)
    fwf_file = load_point_cloud_file(fwf_path)
    metrics, _ = compute_combined_metrics_fwf(las_points, fwf_file)
    return np.asarray(metrics)

def process_single_pointcloud(args):
    """
    Computes numerical features for a single point cloud.
    
    This function processes a single point cloud file, calculates combined metrics for the 
    point cloud, and returns them as a NumPy array.
    
    Args:
    args (tuple): A tuple containing the path to the point cloud file, the current index, 
                  and the total count of files.
                  (pointcloud_path, index, total_count)
    
    Returns:
    arrmetrics (ndarray): A NumPy array containing the computed numerical features.
    """
    pointcloud_path, idx, total_count = args
    logging.info(f"Processing point cloud {idx+1}/{total_count}")
    las_points = load_point_cloud(pointcloud_path)
    metrics, _ = compute_combined_metrics(las_points)
    return np.asarray(metrics)

def generate_metrics_for_selected_pointclouds_fwf(selected_pointclouds, filtered_fwf_pointclouds, metrics_dir, capsel, growsel, prev_elim_features):
    """
    Generates numerical features for regular and FWF point clouds in parallel.

    Args:
    selected_pointclouds: List of point cloud paths.
    filtered_fwf_pointclouds: List of FWF point cloud paths.
    metrics_dir: Savepath for numerical features.
    capsel: User-specified acquisition method.
    growsel: User-specified leaf-condition.

    Returns:
    combined_metrics: Array of numerical features for each individual tree.
    """
    savename = f"training_generated_metrics_{capsel}_{growsel}.csv"
    metrics_path = main_utils.join_paths(metrics_dir, savename)
    if not workspace_setup.get_are_fwf_pcs_extracted(metrics_dir):
        num_workers = mp.cpu_count() // 2  
        args_list = [(selected_pointclouds[i], filtered_fwf_pointclouds[i], i, len(selected_pointclouds)) for i in range(len(selected_pointclouds))]
        with mp.Pool(processes=num_workers) as pool:
            all_metrics = pool.map(process_single_pointcloud_fwf, args_list)
        combined_metrics = np.vstack(all_metrics)
        save_metrics_to_csv_pandas(all_metrics, metrics_path)
    else:
        logging.info("Previously generated metrics found, importing!")
        combined_metrics = load_metrics_from_path(metrics_path)
    feature_names = ["height_quantile_25", "height_quantile_50", "height_quantile_75", "dens0", "dens1", "dens2", "dens3", "dens4", "dens5", "dens6", "dens7", "dens8", "dens9", "dec0",
                    "dec1", "dec2", "dec3", "dec4", "dec5", "dec6", "dec7", "dec8", "max_crown_diameter", "clustering_degree", "intensity_mean", "intensity_std", "intensity_skewness", "intensity_kurtosis", "mean_pulse_widths",
                    "crown_volume", "segdens0", "segdens1", "segdens2", "segdens3", "segdens4", "segdens5", "segdens6", "segdens7", "highest_branch", "lowest_branch", "longest_spread", "longest_cross_spread",
                    "equivalent_crown_diameter", "canopy_width_x", "canopy_width_y", "canopy_volume", "point_density", "lai", "canopy_closure", "crown_base_height", "std_dev_height",
                    "height_kurtosis", "height_skewness", "crown_area", "crown_perimeter", "crown_volume_to_height_ratio", "canopy_cover_fraction", "stem_volume", "canopy_base_height", "fwhm", "echo_width",
                    "surface_area", "surface_to_volume_ratio", "avg_nn_dist", "fract_dimension", "bb_dims", "crown_shape_indices",
                    "convex_hull_compactness", "gini_height", "canopy_porosity", "lambda_1_2", "lambda_2_3",
                    "linearity", "planarity", "sphericity", "branch_angle_variance", "curvature",
                    "height_variation_coeff", "entropy_height", "leaf_inclination",
                    "leaf_curvature", "anisotropy", "canopy_skewness", "canopy_kurtosis", "canopy_ellipticity",
                    "branch_density", "crown_asymmetry", "crown_circularity", "intensity_contrast",
                    "density_gradient", "crown_compactness", "crown_symmetry", "surface_roughness", "local_dens_variation"]
    df_metrics = pd.DataFrame(combined_metrics, columns=feature_names)
    if prev_elim_features:
        df_metrics_reduced = df_metrics.drop(columns=prev_elim_features)
        selected_features = df_metrics_reduced.columns.tolist()
        combined_metrics = df_metrics_reduced.to_numpy()
        highly_correlated_features = prev_elim_features
    else:
        correlation_matrix = df_metrics.corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        highly_correlated_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
        df_metrics_reduced = df_metrics.drop(columns=highly_correlated_features)
        selected_features = df_metrics_reduced.columns.tolist()
        combined_metrics = df_metrics_reduced.to_numpy()
        logging.info(f"Removed {len(highly_correlated_features)} redundant features due to high correlation.")
        logging.info(f"Remaining features: {len(selected_features)}")
    return combined_metrics, selected_features, highly_correlated_features

def load_point_cloud_file(file_path):
    """
    Retrieves the las file from a filepath.

    Args:
    file_path: Filepath of a las point cloud.

    Returns:
    las_file: Las file instance with Header and VLRs.
    """
    try:
        las_file = lp.read(file_path)
    except OSError as e:
        logging.error("Error loading file %s: %s", file_path, e)
        raise
    return las_file

def load_metrics_from_path(metrics_path):
    """
    Loads numerical features from a CSV file at the specified path.

    Args:
    metrics_path: Filepath of the CSV file.

    Returns:
    metrics_combined: Array of numerical features for each individual tree.
    """
    metrics_combined = np.genfromtxt(metrics_path, delimiter=',', dtype=float)
    return metrics_combined
 
def save_metrics_to_csv_pandas(metrics_list, file_name):
    """
    Saves generated metrics to a CSV file at the specified path.

    Args:
    metrics_list: List of numerical features.
    file_name: Filename for the CSV file.
    """
    df = pd.DataFrame(metrics_list)
    df.to_csv(file_name, index=False, header=False)

def compute_combined_metrics_fwf(points, las_file):
    """
    Generates the numerical features for a point cloud.

    Args:
    points: Array of las file point cloud points.
    las_file: Laspy LAS file instance.

    Returns:
    metrics: List of numerical features for the point cloud.
    """
    metrics = []
    if 'intensity' in las_file.point_format.dimension_names:
        intensities = las_file.intensity
    else:
        intensities = None
    all_waveform_data = []
    z_values = points[:, 2]
    for vlr in las_file.header.vlrs:
        if 99 < vlr.record_id < 355:
            waveform_bytes = vlr.record_data_bytes()
            waveform_data = np.frombuffer(waveform_bytes, dtype=np.int16)
            all_waveform_data.extend(waveform_data)
    logging.debug("Waveform data: %s", all_waveform_data)
    height_quantile_25 = compute_height_quantile(points, 25)
    height_quantile_50 = compute_height_quantile(points, 50)
    height_quantile_75 = compute_height_quantile(points, 75)
    dens0, dens1, dens2, dens3, dens4, dens5, dens6, dens7, dens8, dens9 = compute_point_density_normalized_height(points)
    dec0, dec1, dec2, dec3, dec4, dec5, dec6, dec7, dec8 = compute_height_density_deciles(points)
    max_crown_diameter = compute_maximum_crown_diameter(points)
    clustering_degree = compute_points_relative_clustering_degree(points)
    intensity_mean = compute_intensity_mean(intensities) if intensities is not None else 0.0
    intensity_std = compute_intensity_std(intensities) if intensities is not None else 0.0
    intensity_skewness = compute_intensity_skewness(intensities) if intensities is not None else 0.0
    intensity_kurtosis = compute_intensity_kurtosis(intensities) if intensities is not None else 0.0
    mean_pulse_widths = compute_mean_pulse_widths(all_waveform_data)
    crown_volume = compute_crown_volume(points)
    segdens0, segdens1, segdens2, segdens3, segdens4, segdens5, segdens6, segdens7 = compute_vertical_segments_distribution(points)
    tree_height = compute_tree_height(points)
    highest_branch, lowest_branch = compute_highest_lowest_branches(points)
    longest_spread = compute_longest_spread(points)
    longest_cross_spread = compute_longest_cross_spread(points)
    equivalent_crown_diameter = compute_equivalent_crown_diameter(points)
    canopy_width_x, canopy_width_y = compute_canopy_width(points)
    canopy_volume = compute_canopy_volume(points)
    point_density = compute_point_density(points, canopy_volume)
    lai = compute_lai(points, tree_height)
    canopy_closure = compute_canopy_closure(points, canopy_width_x, canopy_width_y)
    crown_base_height = compute_crown_base_height(points)
    std_dev_height = compute_std_dev_height(points)
    height_kurtosis = compute_kurtosis(points)
    height_skewness = compute_skewness(points)
    crown_area = compute_crown_area(points)
    crown_perimeter = compute_crown_perimeter(points)
    crown_volume_to_height_ratio = compute_crown_volume_to_height_ratio(crown_volume, tree_height)
    canopy_cover_fraction = compute_canopy_cover_fraction(points, canopy_width_x, canopy_width_y)
    stem_volume = estimate_stem_volume(points, tree_height)
    canopy_base_height = compute_canopy_base_height(points)
    fwhm = compute_fwhm(all_waveform_data)
    echo_width = compute_echo_width(all_waveform_data)
    surface_area = compute_surface_area(points)
    surface_to_volume_ratio = compute_surface_to_volume_ratio(surface_area, canopy_volume)
    avg_nn_dist = compute_average_nearest_neighbor_distance(points)
    fract_dimension = compute_fractal_dimension(points, k=2)
    bb_dims = compute_bounding_box_dimensions(points)
    crown_shape_indices = compute_crown_shape_indices(points)
    convex_hull_compactness = compute_convex_hull_compactness(points)
    gini_height = compute_gini_coefficient(z_values)
    canopy_porosity = compute_canopy_porosity(points, canopy_volume)
    lambda_1_2, lambda_2_3 = compute_eigenvalue_ratios(points)
    linearity, planarity, sphericity = compute_linearity_planarity_sphericity(points)
    branch_angle_variance = compute_branch_angle_distribution(points)
    curvature = compute_curvature(points)
    height_variation_coeff = compute_height_variation_coefficient(z_values)
    entropy_height = compute_entropy_height_distribution(z_values)
    branch_density = compute_branch_density_profile(points)
    canopy_skewness = compute_canopy_skewness(points)
    canopy_kurtosis = compute_canopy_kurtosis(points)
    canopy_ellipticity = compute_canopy_ellipticity(points)
    leaf_inclination = compute_leaf_inclination_angle_distribution(points)
    leaf_curvature = compute_leaf_surface_curvature(points)
    anisotropy = compute_point_cloud_anisotropy(points)
    crown_asymmetry = compute_crown_asymmetry(points)
    crown_circularity = compute_crown_circularity(points)
    intensity_contrast = compute_intensity_contrast(intensities)
    density_gradient = compute_vertical_density_gradient(points)
    tc_compactness = compute_tree_crown_compactness(points)
    crown_symmetry = compute_crown_symmetry(points)
    surface_roughness = compute_surface_roughness(points)
    local_dens_variation = compute_local_density_variation(points)
    metrics.extend([
        float(height_quantile_25), float(height_quantile_50), float(height_quantile_75),
        float(dens0), float(dens1), float(dens2), float(dens3), float(dens4), float(dens5), float(dens6), float(dens7), float(dens8), float(dens9),
        float(dec0), float(dec1), float(dec2), float(dec3), float(dec4), float(dec5), float(dec6), float(dec7), float(dec8),
        float(max_crown_diameter), float(clustering_degree), float(intensity_mean), float(intensity_std), 
        float(intensity_skewness), float(intensity_kurtosis), float(mean_pulse_widths),
        float(crown_volume), float(segdens0), float(segdens1), float(segdens2), float(segdens3), 
        float(segdens4), float(segdens5), float(segdens6), float(segdens7), float(highest_branch), 
        float(lowest_branch), float(longest_spread), float(longest_cross_spread), float(equivalent_crown_diameter),
        float(canopy_width_x), float(canopy_width_y), float(canopy_volume), float(point_density), float(lai),
        float(canopy_closure), float(crown_base_height), float(std_dev_height), float(height_kurtosis),
        float(height_skewness), float(crown_area), float(crown_perimeter), float(crown_volume_to_height_ratio),
        float(canopy_cover_fraction), float(stem_volume), float(canopy_base_height), float(fwhm), float(echo_width),
        float(surface_area), float(surface_to_volume_ratio), float(avg_nn_dist), float(fract_dimension),
        float(bb_dims), float(crown_shape_indices), float(convex_hull_compactness), float(gini_height), float(canopy_porosity), float(lambda_1_2), float(lambda_2_3),
        float(linearity), float(planarity), float(sphericity), float(branch_angle_variance), float(curvature),
        float(height_variation_coeff), float(entropy_height), float(leaf_inclination),
        float(leaf_curvature), float(anisotropy), float(canopy_skewness), float(canopy_kurtosis), float(canopy_ellipticity),
        float(branch_density), float(crown_asymmetry), float(crown_circularity), float(intensity_contrast),
        float(density_gradient), float(tc_compactness), float(crown_symmetry), float(surface_roughness), float(local_dens_variation)
    ])
    feature_names = ["height_quantile_25", "height_quantile_50", "height_quantile_75", "dens0", "dens1", "dens2", "dens3", "dens4", "dens5", "dens6", "dens7", "dens8", "dens9", "dec0",
                     "dec1", "dec2", "dec3", "dec4", "dec5", "dec6", "dec7", "dec8", "max_crown_diameter", "clustering_degree", "intensity_mean", "intensity_std", "intensity_skewness", "intensity_kurtosis", "mean_pulse_widths",
                     "crown_volume", "segdens0",
                     "segdens1", "segdens2", "segdens3", "segdens4", "segdens5", "segdens6", "segdens7", "highest_branch", "lowest_branch", "longest_spread", "longest_cross_spread",
                     "equivalent_crown_diameter", "canopy_width_x", "canopy_width_y", "canopy_volume", "point_density", "lai", "canopy_closure", "crown_base_height", "std_dev_height",
                     "height_kurtosis", "height_skewness", "crown_area", "crown_perimeter", "crown_volume_to_height_ratio", "canopy_cover_fraction", "stem_volume", "canopy_base_height", "fwhm", "echo_width",
                     "surface_area", "surface_to_volume_ratio", "avg_nn_dist", "fract_dimension", "bb_dims", "crown_shape_indices",
                     "convex_hull_compactness", "gini_height", "canopy_porosity", "lambda_1_2", "lambda_2_3",
                     "linearity", "planarity", "sphericity", "branch_angle_variance", "curvature",
                     "height_variation_coeff", "entropy_height", "leaf_inclination",
                     "leaf_curvature", "anisotropy", "canopy_skewness", "canopy_kurtosis", "canopy_ellipticity",
                     "branch_density", "crown_asymmetry", "crown_circularity", "intensity_contrast",
                     "density_gradient", "crown_compactness", "crown_symmetry", "surface_roughness", "local_dens_variation"]
    return metrics, feature_names

def match_images_with_pointclouds(selected_pointclouds, selected_images):
    """
    Matches images with their corresponding source point cloud.

    Args:
    selected_pointclouds: List of point cloud file paths.
    selected_images: List of image file paths.

    Returns:
    frontal_images: Ordered list of frontal view image arrays.
    sideways_images: Ordered list of sideways view image arrays.
    """
    frontal_images = []
    sideways_images = []
    for pointcloud_filepath in selected_pointclouds:
        filename_full = pointcloud_filepath.split("/")[-1]
        tree_id = filename_full.split("_")[0]
        species = filename_full.split("_")[2]
        capmeth = filename_full.split("_")[3]
        capdate = filename_full.split("_")[4]
        indid = filename_full.split("_")[5]
        leaf_cond = filename_full.split("_")[6]
        augnum = filename_full.split("_")[7].split(".")[0]
        for image_filepath in selected_images:
            image_filename_full = image_filepath.split("/")[-1]
            image_parts = image_filename_full.split("_")
            if (image_parts[0] == tree_id and
                image_parts[1] == species and
                image_parts[2] == capmeth and
                image_parts[3] == capdate and
                image_parts[4] == indid and
                image_parts[5] == leaf_cond and
                image_parts[-1].split(".")[0] == augnum):
                if "frontal" in image_filename_full:
                    imagearray = read_image(image_filepath)
                    frontal_images.append(imagearray)
                elif "sideways" in image_filename_full:
                    imagearray = read_image(image_filepath)
                    sideways_images.append(imagearray)
                else:
                    pass
    return frontal_images, sideways_images

def drop_nan_columns(arr1, arr2):
    """
    Removes columns with NaN values from two input arrays and returns the cleaned arrays.
    
    This function identifies columns that contain NaN values in either of the input arrays 
    and removes these columns from both arrays, ensuring that the resulting arrays have 
    the same shape. It also returns the indices of the dropped columns.
    
    Args:
    arr1 (ndarray): The first input array.
    arr2 (ndarray): The second input array.
    
    Returns:
    arr1_cleaned (ndarray): The first cleaned array with NaN columns removed.
    arr2_cleaned (ndarray): The second cleaned array with NaN columns removed.
    dropped_indices (list): List of column indices that were dropped.
    """
    nan_columns_arr1 = np.any(np.isnan(arr1), axis=0)
    nan_columns_arr2 = np.any(np.isnan(arr2), axis=0)
    nan_columns = nan_columns_arr1 | nan_columns_arr2
    arr1_cleaned = arr1[:, ~nan_columns]
    arr2_cleaned = arr2[:, ~nan_columns]
    dropped_indices = np.where(nan_columns)[0].tolist()
    return arr1_cleaned, arr2_cleaned, dropped_indices

def generate_training_data(capsel, growsel, filtered_pointclouds, resampled_pointclouds, filtered_pointclouds_pred, resampled_pointclouds_pred, combined_metrics, combined_metrics_pred, images_frontal, images_sideways, images_frontal_pred, images_sideways_pred, sss_testsize, metrics_dir, metrics_dir_pred, rfe_threshold, feature_names):
    """
    Generates the training and validation data for the model, including preprocessing and feature selection.
    
    Args:
    capsel: User-specified acquisition selection method.
    growsel: User-specified leaf-condition selection.
    filtered_pointclouds: List of filtered pointclouds for training.
    resampled_pointclouds: List of resampled pointclouds for training.
    filtered_pointclouds_pred: List of filtered pointclouds for prediction.
    resampled_pointclouds_pred: List of resampled pointclouds for prediction.
    combined_metrics: Combined metrics for the training data.
    combined_metrics_pred: Combined metrics for the prediction data.
    images_frontal: List of frontal images.
    images_sideways: List of sideways images.
    images_frontal_pred: List of frontal images for prediction.
    images_sideways_pred: List of sideways images for prediction.
    sss_testsize: Test size for Stratified Shuffle Split.
    metrics_dir: Directory containing metrics for the training data.
    metrics_dir_pred: Directory containing metrics for the prediction data.
    rfe_threshold: Threshold for Recursive Feature Elimination.
    feature_names: List of feature names for RFE.
    
    Returns:
    X_pc_train: Training point clouds.
    X_pc_val: Validation point clouds.
    X_pc_pred: Point clouds for prediction.
    X_metrics_train: Training metrics.
    X_metrics_val: Validation metrics.
    X_metrics_pred: Metrics for prediction.
    X_img_1_train: Training frontal images.
    X_img_1_val: Validation frontal images.
    X_img_1_pred: Frontal images for prediction.
    X_img_2_train: Training sideways images.
    X_img_2_val: Validation sideways images.
    X_img_2_pred: Sideways images for prediction.
    y_train: Training labels.
    y_val: Validation labels.
    y_pred: Labels for prediction.
    num_classes: Number of classes.
    onehot_to_label_dict: Dictionary for one-hot to label mapping.
    """
    tree_labels = np.array(get_labels_for_trees(filtered_pointclouds))
    tree_labels_pred = np.array(get_labels_for_trees(filtered_pointclouds_pred))
    label_encoder = LabelEncoder()
    elimination_labels = label_encoder.fit_transform(tree_labels)
    numeric_tree_labels = elimination_labels.astype(int)
    onehot_to_label_dict = {numeric_tree_labels[i]: tree_labels[i] for i in range(len(tree_labels))}
    rfe_metrics = []
    rfe_metrics_pred = []
    for file in os.listdir(metrics_dir):
        if "training_rfe" in file:
            rfe_metrics.append(file)
        else:
            pass
    for file_pred in os.listdir(metrics_dir_pred):
        if "training_rfe" in file_pred:
            rfe_metrics_pred.append(file_pred)
        else:
            pass
    if len(rfe_metrics) > 0:
        rfe_metrics_path = main_utils.join_paths(metrics_dir, rfe_metrics[0])
        combined_eliminated_metrics = load_metrics_from_path(rfe_metrics_path)
        rfe_metrics_pred_path = main_utils.join_paths(metrics_dir_pred, rfe_metrics_pred[0])
        combined_eliminated_metrics_pred = load_metrics_from_path(rfe_metrics_pred_path)
        logging.info("Loaded metrics of shape %s and %s", combined_eliminated_metrics.shape, combined_eliminated_metrics_pred.shape)
    else:
        combined_eliminated_metrics, eliminated_features = perform_recursive_feature_elimination_with_threshold(capsel, growsel, combined_metrics, elimination_labels, metrics_dir, rfe_threshold, feature_names)
        logging.info("Eliminated features shape: %s", len(eliminated_features))
        combined_eliminated_metrics_pred = remove_eliminated_features(combined_metrics_pred, feature_names, eliminated_features, metrics_dir_pred, capsel, growsel)
        logging.info("Metrics shape after Recursive Feature Elimination: %s, %s", combined_eliminated_metrics.shape, combined_eliminated_metrics_pred.shape)
    logging.debug("Tree species to train on: %s", np.unique(tree_labels))
    logging.info("One-Hot encoding labels!")
    encoder = OneHotEncoder(sparse_output=False)
    y_orig = encoder.fit_transform(tree_labels.reshape(-1, 1))
    num_classes = len(encoder.categories_[0])
    X_pc_orig = resampled_pointclouds
    X_metrics_orig = combined_eliminated_metrics
    X_img_1_orig = images_frontal
    X_img_2_orig = images_sideways
    X_pc, X_metrics, X_img_1, X_img_2, y = balance_classes(X_pc_orig, X_metrics_orig, X_img_1_orig, X_img_2_orig, y_orig, onehot_to_label_dict)
    y_pred = encoder.fit_transform(tree_labels_pred.reshape(-1, 1))
    X_pc_pred = resampled_pointclouds_pred
    X_metrics_pred = combined_eliminated_metrics_pred
    X_img_1_pred = images_frontal_pred
    X_img_2_pred = images_sideways_pred
    logging.info("Performing Stratified-Shuffle-Split!")
    sss = StratifiedShuffleSplit(n_splits=5, test_size=sss_testsize, random_state=42)
    for train_index_temp, pred_index in sss.split(X_pc, np.argmax(y, axis=1)):
        X_pc_train, X_pc_val = X_pc[train_index_temp], X_pc[pred_index]
        X_metrics_train, X_metrics_val = X_metrics[train_index_temp], X_metrics[pred_index]
        X_img_1_train, X_img_1_val = X_img_1[train_index_temp], X_img_1[pred_index]
        X_img_2_train, X_img_2_val = X_img_2[train_index_temp], X_img_2[pred_index]
        y_train, y_val = y[train_index_temp], y[pred_index]
    print_class_distribution(y_train, y_val, y_pred, onehot_to_label_dict)
    return X_pc_train, X_pc_val, X_pc_pred, X_metrics_train, X_metrics_val, X_metrics_pred, X_img_1_train, X_img_1_val, X_img_1_pred, X_img_2_train, X_img_2_val, X_img_2_pred, y_train, y_val, y_pred, num_classes, onehot_to_label_dict

def get_labels_for_trees(selected_pointclouds):
    """
    Retrieves tree labels from a list of point cloud file paths.

    Args:
    selected_pointclouds: List of point cloud file paths.

    Returns:
    tree_labels: List of tree species names.
    """
    tree_labels = []
    for pointcloud in selected_pointclouds:
        filename_full = pointcloud.split("/")[-1].split(".")[0]
        tree_species = filename_full.split("_")[2]
        tree_labels.append(tree_species)
    return tree_labels

def perform_recursive_feature_elimination_with_threshold(capsel, growsel, X, y, metrics_dir, importance_threshold, feature_names):
    """
    Perform Recursive Feature Elimination (RFE) using RandomForestClassifier and omit features with low importance.

    Args:
    capsel: User-specified acquisition method.
    growsel: User-specified leaf-condition.
    X: Features matrix.
    y: Target vector.
    metrics_dir: Savepath for numerical features.
    importance_threshold: Threshold for feature importance to retain features.
    feature_names: List of feature names.

    Returns:
    X_reduced: Reduced feature set.
    """
    logging.info("Performing Recursive Feature Elimination!")
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    feature_importances = model.feature_importances_
    selected_features_mask = feature_importances > importance_threshold
    selected_feature_names = [name for name, selected in zip(feature_names, selected_features_mask) if selected]
    eliminated_feature_names = [name for name, selected in zip(feature_names, selected_features_mask) if not selected]
    X_reduced = X[:, selected_features_mask]
    savename = f"training_rfe_generated_metrics_{capsel}_{growsel}.csv"
    metrics_path = os.path.join(metrics_dir, savename)
    save_metrics_to_csv_pandas(X_reduced, metrics_path)
    logging.info(f"Number of features kept: {len(selected_feature_names)}")
    logging.info(f"Number of features eliminated: {len(eliminated_feature_names)}")
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)
    plt.figure(figsize=(12, len(importance_df) / 2))
    plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
    plt.axvline(x=importance_threshold, color='red', linestyle='--', label=f'Threshold ({importance_threshold:.2f})')
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.legend()
    plot_path = os.path.join(metrics_dir, f"feature_importances_{capsel}_{growsel}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logging.info(f"Feature importances plot saved to {plot_path}")
    return X_reduced, eliminated_feature_names

def remove_eliminated_features(combined_metrics_pred, feature_names, eliminated_features, metrics_dir, capsel, growsel):
    """
    Removes columns from a DataFrame where the feature name is in the eliminated_features list.

    Args:
    combined_metrics_pred: DataFrame containing numerical features (no column names).
    feature_names: List of feature names corresponding to DataFrame columns.
    eliminated_features: List of feature names that should be removed.

    Returns:
    reduced_df: DataFrame with eliminated features removed.
    remaining_features: List of feature names that remain after elimination.
    """
    feature_names = np.array(feature_names)
    mask = np.isin(feature_names, eliminated_features, invert=True)
    try:
        reduced_array = combined_metrics_pred[:, mask]
    except IndexError as e:
        print(f"Error: {e}")
        print(f"Mask shape: {mask.shape}, Array shape: {combined_metrics_pred.shape}")
        raise
    remaining_features = feature_names[mask]
    savename = f"training_rfe_generated_metrics_{capsel}_{growsel}.csv"
    metrics_path = os.path.join(metrics_dir, savename)
    np.savetxt(metrics_path, reduced_array, delimiter=",")
    metrics_array = load_metrics_from_path(metrics_path)
    return metrics_array

def balance_classes(X_pc_unb, X_metrics_unb, X_img_1_unb, X_img_2_unb, y_pc_unb, onehot_to_label_dict):
    """
    Balances classes by randomly downsampling overrepresented classes.

    Args:
        X: Input data (can be point clouds, metrics, or images).
        y: One-hot encoded labels corresponding to X.
        onehot_to_label_dict: Dictionary to map numeric labels to class names.

    Returns:
        X_balanced: Balanced input data.
        y_balanced: Balanced one-hot encoded labels.
    """
    y_decoded = [onehot_to_label_dict[np.argmax(label)] for label in y_pc_unb]
    class_counts = Counter(y_decoded)
    min_count = min(class_counts.values())
    X_pc_balanced, X_metrics_balanced, X_img_1_balanced, X_img_2_balanced, y_balanced = [], [], [], [], []
    for class_name in class_counts.keys():
        indices = [i for i, label in enumerate(y_decoded) if label == class_name]
        selected_indices = np.random.choice(indices, min_count, replace=False)
        X_pc_balanced.extend(X_pc_unb[selected_indices])
        X_metrics_balanced.extend(X_metrics_unb[selected_indices])
        X_img_1_balanced.extend(X_img_1_unb[selected_indices])
        X_img_2_balanced.extend(X_img_2_unb[selected_indices])
        y_balanced.extend(y_pc_unb[selected_indices])
    X_pc = np.array(X_pc_balanced)
    X_metrics = np.array(X_metrics_balanced)
    X_img_1 = np.array(X_img_1_balanced)
    X_img_2 = np.array(X_img_2_balanced)
    y_pc = np.array(y_balanced)
    balanced_counts = Counter([onehot_to_label_dict[np.argmax(label)] for label in y_balanced])
    return X_pc, X_metrics, X_img_1, X_img_2, y_pc

def print_class_distribution(y_train, y_val, y_pred, onehot_to_label_dict):
    """
    Prints class distribution statistics for training and validation datasets.

    Args:
        y_train: One-hot encoded training labels.
        y_val: One-hot encoded validation labels.
        onehot_to_label_dict: Dictionary to map numeric labels to class names.
    """
    y_train_decoded = [onehot_to_label_dict[np.argmax(label)] for label in y_train]
    y_val_decoded = [onehot_to_label_dict[np.argmax(label)] for label in y_val]
    y_pred_decoded = [onehot_to_label_dict[np.argmax(label)] for label in y_pred]
    train_counts = Counter(y_train_decoded)
    val_counts = Counter(y_val_decoded)
    pred_counts = Counter(y_pred_decoded)
    all_classes = sorted(set(train_counts.keys()).union(val_counts.keys()).union(pred_counts.keys()))
    data = {
        "Class": all_classes,
        "Train Count": [train_counts.get(cls, 0) for cls in all_classes],
        "Validation Count": [val_counts.get(cls, 0) for cls in all_classes],
        "Prediction Count": [pred_counts.get(cls, 0) for cls in all_classes],
    }
    df = pd.DataFrame(data)
    df["Total"] = df["Train Count"] + df["Validation Count"] + df["Prediction Count"]
    df.sort_values(by="Total", ascending=False, inplace=True)

def compute_crown_asymmetry(points):
    hull = ConvexHull(points[:, :2])
    max_diameter = np.max([np.linalg.norm(points[hull.vertices[i]] - points[hull.vertices[j]])
                           for i in range(len(hull.vertices)) for j in range(i + 1, len(hull.vertices))])
    hull_center = np.mean(points[hull.vertices], axis=0)
    distances = np.linalg.norm(points[hull.vertices] - hull_center, axis=1)
    std_dev_dist = np.std(distances)
    return std_dev_dist / max_diameter

def compute_crown_circularity(points):
    hull = ConvexHull(points[:, :2])
    area = hull.volume
    bbox = np.ptp(points[:, :2], axis=0)
    fitted_ellipse_area = np.pi * bbox[0] * bbox[1] / 4
    return area / fitted_ellipse_area

def compute_intensity_contrast(intensities):
    return (np.max(intensities) - np.min(intensities)) / (np.max(intensities) + 1e-10)

def compute_first_last_return_intensity_ratio(waveform_data):
    if len(waveform_data) < 2:
        return 0
    return waveform_data[0] / waveform_data[-1] if waveform_data[-1] > 0 else 0

def compute_leaf_inclination_angle_distribution(points):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(points)
    vertical_vector = np.array([0, 0, 1])
    inclination_angles = np.arccos(np.dot(pca.components_, vertical_vector))
    return np.mean(inclination_angles)  

def compute_leaf_surface_curvature(points):
    curvature = np.linalg.norm(np.gradient(points, axis=0), axis=1).std()
    return curvature

def compute_point_cloud_anisotropy(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    eigenvalues = np.sort(pca.explained_variance_)[::-1]
    return eigenvalues[0] / np.sum(eigenvalues)

def compute_branch_density_profile(points, height_bins=10):
    z_values = points[:, 2]
    hist, _ = np.histogram(z_values, bins=height_bins)
    return np.mean(hist / np.sum(hist))  

def compute_canopy_skewness(points):
    z_values = points[:, 2]
    return np.mean((z_values - np.mean(z_values))**3) / (np.std(z_values)**3 + 1e-8)

def compute_canopy_kurtosis(points):
    z_values = points[:, 2]
    return np.mean((z_values - np.mean(z_values))**4) / (np.std(z_values)**4 + 1e-8)

def compute_canopy_ellipticity(points):
    canopy_points = points[:, :2]  
    hull = ConvexHull(canopy_points)
    bbox_dims = np.ptp(canopy_points, axis=0)
    return bbox_dims[0] / bbox_dims[1] if bbox_dims[1] > 0 else 1

def compute_convex_hull_compactness(points):
    hull = ConvexHull(points)
    bbox_volume = np.prod(np.ptp(points, axis=0))
    return hull.volume / bbox_volume if bbox_volume > 0 else 0

def compute_gini_coefficient(z_values):
    sorted_z = np.sort(z_values)
    n = len(z_values)
    cum_z = np.cumsum(sorted_z, dtype=float)
    return (2.0 * np.sum((np.arange(1, n+1) * sorted_z))) / (n * np.sum(sorted_z)) - (n + 1) / n

def compute_canopy_porosity(points, canopy_volume):
    point_density = len(points) / canopy_volume if canopy_volume > 0 else 0
    return 1 - point_density

def compute_eigenvalue_ratios(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    eigenvalues = pca.explained_variance_
    lambda_1_2 = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > 0 else 0
    lambda_2_3 = eigenvalues[1] / eigenvalues[2] if eigenvalues[2] > 0 else 0
    return lambda_1_2, lambda_2_3

def compute_linearity_planarity_sphericity(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    eigenvalues = np.sort(pca.explained_variance_)[::-1]
    linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0] if eigenvalues[0] > 0 else 0
    planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0] if eigenvalues[0] > 0 else 0
    sphericity = eigenvalues[2] / eigenvalues[0] if eigenvalues[0] > 0 else 0
    return linearity, planarity, sphericity

def compute_branch_angle_distribution(points):
    angles = np.arctan2(points[:, 1], points[:, 0])  
    return np.var(angles)  

def compute_curvature(points):
    curvature = np.linalg.norm(np.gradient(points, axis=0), axis=1).mean()
    return curvature

def compute_height_variation_coefficient(z_values):
    return np.std(z_values) / np.mean(z_values) if np.mean(z_values) > 0 else 0

def compute_entropy_height_distribution(z_values, bins=10):
    hist, _ = np.histogram(z_values, bins=bins, density=True)
    return entropy(hist + 1e-10)  

def compute_waveform_echo_ratio(waveform_data):
    if len(waveform_data) < 2:
        return 0
    return waveform_data[0] / waveform_data[-1] if waveform_data[-1] > 0 else 0

def compute_bounding_box_dimensions(points):
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    bounding_box_dimensions = max_coords - min_coords
    euclidean_distance = np.linalg.norm(bounding_box_dimensions)
    return euclidean_distance

def compute_crown_shape_indices(points):
    canopy_width_x, canopy_width_y = compute_canopy_width(points)
    tree_height = compute_tree_height(points)
    height_to_width_ratio = tree_height / max(canopy_width_x, canopy_width_y)
    return height_to_width_ratio

def compute_surface_to_volume_ratio(surface_area, volume):
    return surface_area / volume

def compute_average_nearest_neighbor_distance(points):
    kdtree = KDTree(points)
    distances, _ = kdtree.query(points, k=2)
    nearest_neighbor_distances = distances[:, 1]
    return np.mean(nearest_neighbor_distances)

def compute_fractal_dimension(points, k=2):
    kdtree = KDTree(points)
    distances, _ = kdtree.query(points, k=k+1)
    nearest_neighbor_distances = distances[:, 1:]
    r = np.mean(nearest_neighbor_distances, axis=0)
    N = np.arange(1, k+1)
    log_r = np.log(r)
    log_N = np.log(N)
    slope, _ = np.polyfit(log_r, log_N, 1)
    return slope

def compute_surface_area(points):
    hull = ConvexHull(points)
    return hull.area

def compute_tree_height(points):
    min_z = np.min(points[:, 2])
    max_z = np.max(points[:, 2])
    return max_z - min_z

def compute_height_quantile(points, quantile):
    z_values = points[:, 2]
    return np.percentile(z_values, quantile)

def compute_point_density_normalized_height(points, height_bins=10):
    z_values = points[:, 2]
    min_z, max_z = np.min(z_values), np.max(z_values)
    normalized_heights = (z_values - min_z) / (max_z - min_z)
    density, _ = np.histogram(normalized_heights, bins=height_bins)
    dens0 = density[0] / len(points)
    dens1 = density[1] / len(points)
    dens2 = density[2] / len(points)
    dens3 = density[3] / len(points)
    dens4 = density[4] / len(points)
    dens5 = density[5] / len(points)
    dens6 = density[6] / len(points)
    dens7 = density[7] / len(points)
    dens8 = density[8] / len(points)
    dens9 = density[9] / len(points)
    return dens0, dens1, dens2, dens3, dens4, dens5, dens6, dens7, dens8, dens9

def compute_height_density_deciles(points):
    z_values = points[:, 2]
    deciles = [np.percentile(z_values, i * 10) for i in range(1, 10)]
    dec0 = deciles[0]
    dec1 = deciles[1]
    dec2 = deciles[2]
    dec3 = deciles[3]
    dec4 = deciles[4]
    dec5 = deciles[5]
    dec6 = deciles[6]
    dec7 = deciles[7]
    dec8 = deciles[8]
    return dec0, dec1, dec2, dec3, dec4, dec5, dec6, dec7, dec8

def compute_maximum_crown_diameter(points):
    hull = ConvexHull(points[:, :2])
    max_diameter = np.max([np.linalg.norm(points[hull.vertices[i]] - points[hull.vertices[j]])
                           for i in range(len(hull.vertices)) for j in range(i + 1, len(hull.vertices))])
    return max_diameter

def determine_radius(scale='medium'):
    if scale == 'fine':
        return 0.1
    elif scale == 'medium':
        return 0.5
    elif scale == 'large':
        return 1
    else:
        raise ValueError("Invalid scale value. Choose 'fine', 'medium', or 'large'.")

def compute_points_relative_clustering_degree(points):
    radius = determine_radius(scale='medium')
    kdtree = KDTree(points[:, :3])
    clustering_degrees = []
    for point in points:
        indices = kdtree.query_ball_point(point[:3], radius)
        neighbor_points = points[indices]
        if len(neighbor_points) > 1:
            local_density = len(neighbor_points) / ((4/3) * np.pi * radius**3)
            clustering_degrees.append(local_density)
    if clustering_degrees:
        return np.mean(clustering_degrees)
    else:
        return 0.0

def compute_intensity_mean(intensities):
    return np.mean(intensities)

def compute_intensity_std(intensities):
    return np.std(intensities)

def compute_intensity_skewness(intensities):
    return skew(intensities)

def compute_intensity_kurtosis(intensities):
    return kurtosis(intensities)

def compute_mean_pulse_widths(waveform_data):
    return np.mean(waveform_data, axis=0)

def compute_crown_volume(points):
    hull = ConvexHull(points)
    return hull.volume

def compute_vertical_segments_distribution(points, num_segments=8):
    z_values = points[:, 2]
    min_z, max_z = np.min(z_values), np.max(z_values)
    segment_heights = np.linspace(min_z, max_z, num_segments + 1)
    segment_densities = []
    for i in range(num_segments):
        mask = (z_values >= segment_heights[i]) & (z_values < segment_heights[i + 1])
        segment_densities.append(np.sum(mask))
    segdens0 = segment_densities[0]
    segdens1 = segment_densities[1]
    segdens2 = segment_densities[2]
    segdens3 = segment_densities[3]
    segdens4 = segment_densities[4]
    segdens5 = segment_densities[5]
    segdens6 = segment_densities[6]
    segdens7 = segment_densities[7]
    return segdens0, segdens1, segdens2, segdens3, segdens4, segdens5, segdens6, segdens7

def compute_highest_lowest_branches(points):
    z_values = points[:, 2]
    hist, bin_edges = np.histogram(z_values, bins=50)
    hls_index = np.argmax(hist > (0.05 * len(points)))
    lls_index = np.argmax(hist[::-1] > (0.05 * len(points)))
    hls = bin_edges[hls_index]
    lls = bin_edges[::-1][lls_index]
    return hls, lls

def compute_longest_spread(points):
    hull = ConvexHull(points[:, :2])
    max_spread = np.max([np.linalg.norm(points[hull.vertices[i]] - points[hull.vertices[j]])
                         for i in range(len(hull.vertices)) for j in range(i + 1, len(hull.vertices))])
    return max_spread

def compute_longest_cross_spread(points):
    hull = ConvexHull(points[:, :2])
    cross_spreads = [np.linalg.norm(points[hull.vertices[i]] - points[hull.vertices[j]])
                     for i in range(len(hull.vertices)) for j in range(i + 1, len(hull.vertices))]
    cross_spreads.sort()
    return cross_spreads[-2] if len(cross_spreads) > 1 else cross_spreads[0]

def compute_equivalent_crown_diameter(points):
    hull = ConvexHull(points[:, :2])
    crown_area = hull.volume
    return np.sqrt(4 * crown_area / np.pi)

def compute_canopy_width(points):
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    width_x = max_x - min_x
    width_y = max_y - min_y
    return width_x, width_y

def compute_canopy_volume(points):
    hull = ConvexHull(points)
    return hull.volume

def compute_point_density(points, volume):
    return len(points) / volume

def compute_lai(points, height):
    z_values = points[:, 2]
    gaps = np.histogram(z_values, bins=100)[0]
    gap_fraction = gaps / np.sum(gaps)
    epsilon = 1e-10
    gap_fraction = np.where(gap_fraction == 0, epsilon, gap_fraction)
    lai = -np.log(gap_fraction).sum() / height
    return lai

def compute_canopy_closure(points, width_x, width_y):
    canopy_area = width_x * width_y
    canopy_closure = len(points) / canopy_area
    return canopy_closure

def compute_crown_base_height(points):
    z_values = points[:, 2]
    hist, bin_edges = np.histogram(z_values, bins=50)
    base_height_index = np.argmax(hist > (0.05 * len(points)))
    return bin_edges[base_height_index]

def compute_std_dev_height(points):
    return np.std(points[:, 2])

def compute_kurtosis(points):
    return kurtosis(points[:, 2])

def compute_skewness(points):
    return skew(points[:, 2])

def compute_crown_area(points):
    hull = ConvexHull(points[:, :2])
    return hull.volume

def compute_crown_perimeter(points):
    hull = ConvexHull(points[:, :2])
    return hull.area

def compute_crown_volume_to_height_ratio(volume, height):
    return volume / height

def compute_canopy_cover_fraction(points, width_x, width_y):
    canopy_area = ConvexHull(points[:, :2]).volume
    ground_area = width_x * width_y
    canopy_cover_fraction = canopy_area / ground_area
    return canopy_cover_fraction

def compute_stem_volume(diameter, height):
    radius = diameter / 2
    return np.pi * (radius ** 2) * height

def estimate_stem_volume(points, height):
    canopy_area = ConvexHull(points[:, :2]).volume
    diameter = np.sqrt(canopy_area / np.pi)
    return compute_stem_volume(diameter, height)

def compute_canopy_base_height(points):
    z_values = points[:, 2]
    hist, bin_edges = np.histogram(z_values, bins=100)
    canopy_base_index = np.argmax(hist > (0.05 * np.max(hist)))
    return bin_edges[canopy_base_index]

def compute_fwhm(waveform_data):
    max_amplitude = max(waveform_data)
    half_max_amplitude = max_amplitude / 2
    start_index = next(i for i, x in enumerate(waveform_data) if x >= half_max_amplitude)
    end_index = len(waveform_data) - next(i for i, x in enumerate(reversed(waveform_data)) if x >= half_max_amplitude)
    fwhm = end_index - start_index
    return fwhm

def compute_echo_width(waveform_data, threshold=0.1):
    max_amplitude = max(waveform_data)
    half_max_amplitude = max_amplitude * threshold
    crossing_indices = [i for i in range(1, len(waveform_data)-1) if (waveform_data[i-1] <= half_max_amplitude and waveform_data[i] > half_max_amplitude) or 
                                                               (waveform_data[i-1] > half_max_amplitude and waveform_data[i] <= half_max_amplitude)]
    if len(crossing_indices) < 2:
        return 0
    fwhms = []
    for i in range(0, len(crossing_indices)-1, 2):
        start_index = crossing_indices[i]
        end_index = crossing_indices[i+1]
        fwhm = end_index - start_index
        fwhms.append(fwhm)
    if not fwhms:
        return 0
    echo_width = max(fwhms) - min(fwhms)
    return echo_width


def compute_vertical_density_gradient(points, bins=5):
    z_values = points[:, 2]
    min_z, max_z = np.min(z_values), np.max(z_values)
    heights = np.linspace(min_z, max_z, bins+1)
    densities = []
    for i in range(bins):
        count = np.sum((z_values >= heights[i]) & (z_values < heights[i+1]))
        densities.append(count)
    gradient = np.polyfit(range(bins), densities, 1)[0]  
    return gradient

def compute_tree_crown_compactness(points):
    hull = ConvexHull(points[:, :2])  
    crown_area = hull.volume
    num_points = len(points)
    compactness = num_points / crown_area  
    return compactness

def compute_crown_symmetry(points):
    hull = ConvexHull(points[:, :2])
    center = np.mean(points[hull.vertices], axis=0)
    distances = np.linalg.norm(points[hull.vertices] - center, axis=1)
    left_half = distances[points[hull.vertices][:, 0] < center[0]]
    right_half = distances[points[hull.vertices][:, 0] > center[0]]
    return np.std(left_half) / np.std(right_half)  

def compute_surface_roughness(points):
    kdtree = KDTree(points)
    distances, _ = kdtree.query(points, k=10)
    return np.std(distances[:, 1])  

def compute_local_density_variation(points, radius=0.5):
    kdtree = KDTree(points[:, :3])
    densities = []
    for point in points:
        indices = kdtree.query_ball_point(point[:3], radius)
        densities.append(len(indices))
    return np.std(densities)  

def eliminate_unused_species(reg_pc_folder, elimination_percentage, netpcsize):
    """
    Removes point clouds of underrepresented species based on the given elimination percentage.
    
    Args:
    reg_pc_folder: Directory containing regular point clouds.
    elimination_percentage: Minimum percentage of representation for species to be kept.
    netpcsize: Threshold for minimum number of points required in a point cloud.
    
    Returns:
    species_distribution: Distribution of species after filtering.
    """
    pointclouds = select_pointclouds(reg_pc_folder)
    species_list = get_species_distribution(pointclouds)
    species_to_use, species_distribution = eliminate_underrepresented_species(species_list, elimination_percentage)
    def extract_species(filename):
        return filename.split("_")[2]
    for pc in os.listdir(reg_pc_folder):
        if pc.endswith(".laz"):
            pc_path = os.path.join(reg_pc_folder, pc)
            pc_points = load_point_cloud(pc_path)
            species = extract_species(pc)
            if species in species_to_use and len(pc_points) > netpcsize/2:
                pass
            else:
                os.remove(os.path.join(reg_pc_folder, pc))
    return species_distribution

def move_pointclouds_to_preds(reg_pc_folder, reg_pc_pred_folder):
    """
    Moves every 9th point cloud per species from reg_pc_folder to reg_pc_pred_folder.
    Assumes species name is the third element in the filename when split by '_'.

    Args:
    reg_pc_folder (str): Source folder containing point clouds.
    reg_pc_pred_folder (str): Destination folder for selected point clouds.
    """
    reg_pointclouds = sorted([pc for pc in os.listdir(reg_pc_folder) if pc.endswith(".laz")])
    species_counters = {}
    for reg_pc in reg_pointclouds:
        parts = reg_pc.split("_")
        if len(parts) < 3:
            continue
        species = parts[2]
        if species not in species_counters:
            species_counters[species] = 0
        species_counters[species] += 1
        if species_counters[species] % 9 == 0:
            reg_src = os.path.join(reg_pc_folder, reg_pc)
            reg_dst = os.path.join(reg_pc_pred_folder, reg_pc)
            os.makedirs(reg_pc_pred_folder, exist_ok=True)
            try:
                shutil.move(reg_src, reg_dst)
            except Exception as e:
                print(f"❌ Error moving {reg_pc}: {e}")

def get_species_distribution(selected_pointclouds):
    """
    Create an ordered list of species present in a list of point clouds.

    Args:
    selected_pointclouds: List of point cloud file paths.

    Returns:
    species_list: Ordered list of species names.
    """
    species_list = []
    for pointcloud in selected_pointclouds:
        filename = pointcloud.split("/")[-1]
        species = filename.split("_")[2]
        species_list.append(species)
    return species_list

def augment_selection(pointclouds, max_pc_scale, pc_path_selection, species_distribution):
    """
    Main utility for augmentation of point clouds.

    Args:
    pointclouds: List of pooint cloud file paths.
    elimination_percentage: User-specified elimination threshold.
    max_pc_scale: User-specified scaling factor.
    pc_path_selection: Savepath for regular point clouds.
    pc_size: User-specified resampling target.
    capsel: User-specified acquisition selection.
    """
    if check_if_data_is_augmented_already(pointclouds) == False:
        max_representation = get_maximum_distribution(species_distribution)
        augment_species_pointclouds(pointclouds, max_representation, species_distribution, max_pc_scale, pc_path_selection)
    else:
        logging.info("Augmented data found, loading!")

def augment_species_pointclouds(species_pcs, max_representation, species_distribution, max_scale, pc_path_selection):
    """
    Augments point clouds of species by applying random transformations (rotation, scaling, jittering, dropout).
    This function generates augmented versions of the point clouds, adding variability to the dataset.
    
    Args:
    species_pcs: List of file paths to species' point clouds.
    max_representation: Maximum representation of species to ensure balanced dataset.
    species_distribution: Distribution of species across the dataset.
    max_scale: Maximum scaling factor for augmenting the point clouds.
    pc_path_selection: Directory to save augmented point clouds.
    """
    pc_index = 0
    for pointcloud in species_pcs:
        current_species = get_species_for_pointcloud(pointcloud)
        current_species_amount = get_abs_num(current_species, species_distribution)
        upscale_fac = get_upscale_factor(current_species_amount, max_representation)
        pc_name_full = os.path.split(pointcloud)[1]
        pc_name_extension = pc_name_full.split(".")[-1]
        pc_name_f = pc_name_full.split(".")[0]
        pc_name_parts = pc_name_f.split("_")[:-1]
        filename_pc = pc_name_parts[0] + "_" + pc_name_parts[1] + "_" + pc_name_parts[2] + "_" + pc_name_parts[3] + "_" + pc_name_parts[4] + "_" + pc_name_parts[5] + "_" + pc_name_parts[6] 
        pc_points, pc = load_point_cloud_and_file(pointcloud)
        logging.debug("Number of points: %s", len(pc_points))
        for i in range(0, int(upscale_fac)*4):
            pc_index+=1
            outFile_p = lp.LasData(pc.header)
            outFile_p.vlrs = pc.vlrs
            new_reg_points = pc_points
            exported_points_reg = center_pointcloud_o3d(new_reg_points)
            angle = pick_random_angle(np.random.randint(1, 360))
            rotated_pc = rotate_point_cloud(exported_points_reg, angle)
            scale_factors = np.random.uniform(1 - max_scale, 1 + max_scale, size=3)
            scaled_rotated_pc = scale_point_cloud(rotated_pc, scale_factors)
            if np.random.rand() > 0.5:
                scaled_rotated_pc[:, 0] *= -1  
            if np.random.rand() > 0.5:
                scaled_rotated_pc[:, 1] *= -1  
            noise_std = 0.005  
            scaled_rotated_pc += np.random.normal(0, noise_std, scaled_rotated_pc.shape)
            dropout_prob = np.random.uniform(0.05, 0.15)  
            mask = np.random.rand(scaled_rotated_pc.shape[0]) > dropout_prob
            scaled_rotated_pc = scaled_rotated_pc[mask]
            scaled_rotated_pc += np.random.uniform(-0.045, 0.045, scaled_rotated_pc.shape)
            jittered_shuffled_pc = np.random.permutation(scaled_rotated_pc)
            outFile_p.points = outFile_p.points[:jittered_shuffled_pc.shape[0]]
            adjust_las_header(outFile_p, jittered_shuffled_pc)
            outFile_p.x = jittered_shuffled_pc[:, 0]
            outFile_p.y = jittered_shuffled_pc[:, 1]
            outFile_p.z = jittered_shuffled_pc[:, 2]
            new_filename_pc = filename_pc + "_" + "aug0" + str(pc_index) + str(i) + "." + pc_name_extension
            logging.info("Created point cloud %s!", new_filename_pc)
            savepath_pc = os.path.join(pc_path_selection + "/" + new_filename_pc)
            save_point_cloud(savepath_pc, pc, outFile_p)

def get_species_for_pointcloud(species_pc):
    """
    Retrieves species of a point cloud.

    Args:
    species_pc: Individual tree point cloud.

    Returns:
    species: The species of the point cloud.
    """
    filename = os.path.split(species_pc)[1]
    species = filename.split("_")[2]
    return species

def get_user_specified_data(pc_path, img_path, cap_sel, grow_sel):
    """
    Selects data based on user-specifications.

    Args:
    pc_path: Working directory point cloud path.
    img_path: Working directory image path.
    cap_sel: User-specified acquisition method.
    grow_sel User-specified leaf-condition.

    Returns:
    selected_pointclouds: Point clouds selected based on the specifications.
    selected_images: Images selected based on the specifications.
    """
    selected_pointclouds = select_data_according_to_specifications(cap_sel, grow_sel, pc_path)
    selected_images = select_data_according_to_specifications(cap_sel, grow_sel, img_path)
    return selected_pointclouds, selected_images

def generate_metrics_for_selected_pointclouds(selected_pointclouds, metrics_dir, capsel, growsel, prev_elim_features):
    """
    Generates numerical features for regular and FWF point clouds in parallel.

    Args:
    selected_pointclouds: List of point cloud paths.
    filtered_fwf_pointclouds: List of FWF point cloud paths.
    metrics_dir: Savepath for numerical features.
    capsel: User-specified acquisition method.
    growsel: User-specified leaf-condition.

    Returns:
    combined_metrics: Array of numerical features for each individual tree.
    """
    savename = f"training_generated_metrics_{capsel}_{growsel}.csv"
    metrics_path = main_utils.join_paths(metrics_dir, savename)
    if not workspace_setup.get_are_fwf_pcs_extracted(metrics_dir):
        num_workers = mp.cpu_count() // 2  
        args_list = [(selected_pointclouds[i], i, len(selected_pointclouds)) for i in range(len(selected_pointclouds))]
        with mp.Pool(processes=num_workers) as pool:
            all_metrics = pool.map(process_single_pointcloud, args_list)
        combined_metrics = np.vstack(all_metrics)
        save_metrics_to_csv_pandas(all_metrics, metrics_path)
    else:
        logging.info("Previously generated metrics found, importing!")
        combined_metrics = load_metrics_from_path(metrics_path)
    feature_names = ["height_quantile_25", "height_quantile_50", "height_quantile_75", "dens0", "dens1", "dens2", "dens3", "dens4", "dens5", "dens6", "dens7", "dens8", "dens9", "dec0",
                     "dec1", "dec2", "dec3", "dec4", "dec5", "dec6", "dec7", "dec8", "max_crown_diameter", "clustering_degree", "crown_volume", "segdens0",
                     "segdens1", "segdens2", "segdens3", "segdens4", "segdens5", "segdens6", "segdens7", "highest_branch", "lowest_branch", "longest_spread", "longest_cross_spread",
                     "equivalent_crown_diameter", "canopy_width_x", "canopy_width_y", "canopy_volume", "point_density", "lai", "canopy_closure", "crown_base_height", "std_dev_height",
                     "height_kurtosis", "height_skewness", "crown_area", "crown_perimeter", "crown_volume_to_height_ratio", "canopy_cover_fraction", "stem_volume", "canopy_base_height",
                     "surface_area", "surface_to_volume_ratio", "avg_nn_dist", "fract_dimension", "bb_dims", "crown_shape_indices",
                     "convex_hull_compactness", "gini_height", "canopy_porosity", "lambda_1_2", "lambda_2_3",
                     "linearity", "planarity", "sphericity", "branch_angle_variance", "curvature",
                     "height_variation_coeff", "entropy_height", "leaf_inclination",
                     "leaf_curvature", "anisotropy", "canopy_skewness", "canopy_kurtosis", "canopy_ellipticity", "branch_density", "crown_asymmetry", "crown_circularity",
                     "density_gradient", "crown_compactness", "crown_symmetry", "surface_roughness", "local_dens_variation"]
    df_metrics = pd.DataFrame(combined_metrics, columns=feature_names)
    if prev_elim_features:
        df_metrics_reduced = df_metrics.drop(columns=prev_elim_features)
        selected_features = df_metrics_reduced.columns.tolist()
        combined_metrics = df_metrics_reduced.to_numpy()
        highly_correlated_features = prev_elim_features
    else:
        correlation_matrix = df_metrics.corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        highly_correlated_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
        df_metrics_reduced = df_metrics.drop(columns=highly_correlated_features)
        selected_features = df_metrics_reduced.columns.tolist()
        combined_metrics = df_metrics_reduced.to_numpy()
        logging.info(f"Removed {len(highly_correlated_features)} redundant features due to high correlation.")
        logging.info(f"Remaining features: {len(selected_features)}")
    return combined_metrics, selected_features, highly_correlated_features

def compute_combined_metrics(points):
    """
    Generates the numerical features for a point cloud.

    Args:
    points: Array of las file point cloud points.

    Returns:
    metrics: List of numerical features for the point cloud.
    """
    metrics = []
    z_values = points[:, 2]
    height_quantile_25 = compute_height_quantile(points, 25)
    height_quantile_50 = compute_height_quantile(points, 50)
    height_quantile_75 = compute_height_quantile(points, 75)
    dens0, dens1, dens2, dens3, dens4, dens5, dens6, dens7, dens8, dens9 = compute_point_density_normalized_height(points)
    dec0, dec1, dec2, dec3, dec4, dec5, dec6, dec7, dec8 = compute_height_density_deciles(points)
    max_crown_diameter = compute_maximum_crown_diameter(points)
    clustering_degree = compute_points_relative_clustering_degree(points)
    crown_volume = compute_crown_volume(points)
    segdens0, segdens1, segdens2, segdens3, segdens4, segdens5, segdens6, segdens7 = compute_vertical_segments_distribution(points)
    tree_height = compute_tree_height(points)
    highest_branch, lowest_branch = compute_highest_lowest_branches(points)
    longest_spread = compute_longest_spread(points)
    longest_cross_spread = compute_longest_cross_spread(points)
    equivalent_crown_diameter = compute_equivalent_crown_diameter(points)
    canopy_width_x, canopy_width_y = compute_canopy_width(points)
    canopy_volume = compute_canopy_volume(points)
    point_density = compute_point_density(points, canopy_volume)
    lai = compute_lai(points, tree_height)
    canopy_closure = compute_canopy_closure(points, canopy_width_x, canopy_width_y)
    crown_base_height = compute_crown_base_height(points)
    std_dev_height = compute_std_dev_height(points)
    height_kurtosis = compute_kurtosis(points)
    height_skewness = compute_skewness(points)
    crown_area = compute_crown_area(points)
    crown_perimeter = compute_crown_perimeter(points)
    crown_volume_to_height_ratio = compute_crown_volume_to_height_ratio(crown_volume, tree_height)
    canopy_cover_fraction = compute_canopy_cover_fraction(points, canopy_width_x, canopy_width_y)
    stem_volume = estimate_stem_volume(points, tree_height)
    canopy_base_height = compute_canopy_base_height(points)
    surface_area = compute_surface_area(points)
    surface_to_volume_ratio = compute_surface_to_volume_ratio(surface_area, canopy_volume)
    avg_nn_dist = compute_average_nearest_neighbor_distance(points)
    fract_dimension = compute_fractal_dimension(points, k=2)
    bb_dims = compute_bounding_box_dimensions(points)
    crown_shape_indices = compute_crown_shape_indices(points)
    convex_hull_compactness = compute_convex_hull_compactness(points)
    gini_height = compute_gini_coefficient(z_values)
    canopy_porosity = compute_canopy_porosity(points, canopy_volume)
    lambda_1_2, lambda_2_3 = compute_eigenvalue_ratios(points)
    linearity, planarity, sphericity = compute_linearity_planarity_sphericity(points)
    branch_angle_variance = compute_branch_angle_distribution(points)
    curvature = compute_curvature(points)
    height_variation_coeff = compute_height_variation_coefficient(z_values)
    entropy_height = compute_entropy_height_distribution(z_values)
    branch_density = compute_branch_density_profile(points)
    canopy_skewness = compute_canopy_skewness(points)
    canopy_kurtosis = compute_canopy_kurtosis(points)
    canopy_ellipticity = compute_canopy_ellipticity(points)
    leaf_inclination = compute_leaf_inclination_angle_distribution(points)
    leaf_curvature = compute_leaf_surface_curvature(points)
    anisotropy = compute_point_cloud_anisotropy(points)
    crown_asymmetry = compute_crown_asymmetry(points)
    crown_circularity = compute_crown_circularity(points)
    density_gradient = compute_vertical_density_gradient(points)
    tc_compactness = compute_tree_crown_compactness(points)
    crown_symmetry = compute_crown_symmetry(points)
    surface_roughness = compute_surface_roughness(points)
    local_dens_variation = compute_local_density_variation(points)
    metrics.extend([
        float(height_quantile_25), float(height_quantile_50), float(height_quantile_75),
        float(dens0), float(dens1), float(dens2), float(dens3), float(dens4), float(dens5), float(dens6), float(dens7), float(dens8), float(dens9),
        float(dec0), float(dec1), float(dec2), float(dec3), float(dec4), float(dec5), float(dec6), float(dec7), float(dec8),
        float(max_crown_diameter), float(clustering_degree),
        float(crown_volume), float(segdens0), float(segdens1), float(segdens2), float(segdens3), 
        float(segdens4), float(segdens5), float(segdens6), float(segdens7), float(highest_branch), 
        float(lowest_branch), float(longest_spread), float(longest_cross_spread), float(equivalent_crown_diameter),
        float(canopy_width_x), float(canopy_width_y), float(canopy_volume), float(point_density), float(lai),
        float(canopy_closure), float(crown_base_height), float(std_dev_height), float(height_kurtosis),
        float(height_skewness), float(crown_area), float(crown_perimeter), float(crown_volume_to_height_ratio),
        float(canopy_cover_fraction), float(stem_volume), float(canopy_base_height),
        float(surface_area), float(surface_to_volume_ratio), float(avg_nn_dist), float(fract_dimension),
        float(bb_dims), float(crown_shape_indices), float(convex_hull_compactness), float(gini_height), float(canopy_porosity), float(lambda_1_2), float(lambda_2_3),
        float(linearity), float(planarity), float(sphericity), float(branch_angle_variance), float(curvature),
        float(height_variation_coeff), float(entropy_height), float(leaf_inclination),
        float(leaf_curvature), float(anisotropy), float(canopy_skewness), float(canopy_kurtosis), float(canopy_ellipticity),
        float(branch_density), float(crown_asymmetry), float(crown_circularity),
        float(density_gradient), float(tc_compactness), float(crown_symmetry), float(surface_roughness), float(local_dens_variation)
    ])
    feature_names = ["height_quantile_25", "height_quantile_50", "height_quantile_75", "dens0", "dens1", "dens2", "dens3", "dens4", "dens5", "dens6", "dens7", "dens8", "dens9", "dec0",
                     "dec1", "dec2", "dec3", "dec4", "dec5", "dec6", "dec7", "dec8", "max_crown_diameter", "clustering_degree", "crown_volume", "segdens0",
                     "segdens1", "segdens2", "segdens3", "segdens4", "segdens5", "segdens6", "segdens7", "highest_branch", "lowest_branch", "longest_spread", "longest_cross_spread",
                     "equivalent_crown_diameter", "canopy_width_x", "canopy_width_y", "canopy_volume", "point_density", "lai", "canopy_closure", "crown_base_height", "std_dev_height",
                     "height_kurtosis", "height_skewness", "crown_area", "crown_perimeter", "crown_volume_to_height_ratio", "canopy_cover_fraction", "stem_volume", "canopy_base_height",
                     "surface_area", "surface_to_volume_ratio", "avg_nn_dist", "fract_dimension", "bb_dims", "crown_shape_indices",
                     "convex_hull_compactness", "gini_height", "canopy_porosity", "lambda_1_2", "lambda_2_3",
                     "linearity", "planarity", "sphericity", "branch_angle_variance", "curvature",
                     "height_variation_coeff", "entropy_height", "leaf_inclination",
                     "leaf_curvature", "anisotropy", "canopy_skewness", "canopy_kurtosis", "canopy_ellipticity", "branch_density", "crown_asymmetry", "crown_circularity",
                     "density_gradient", "crown_compactness", "crown_symmetry", "surface_roughness", "local_dens_variation"]
    return metrics, feature_names

    




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