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
from sklearn.feature_selection import f_classif
import random
from utils import main_utils
import logging
from random import uniform
from sklearn.neighbors import NearestNeighbors
from functionalities import workspace_setup
import time
from scipy.stats import spearmanr

#----------------------------------------------------------------
#---------             Numerical Features              ----------
#---------          As described in my thesis          ----------
#----------------------------------------------------------------

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

def compute_maximum_height(points):
    return np.max(points[:, 2])

def compute_crown_height(points):
    return compute_maximum_height(points) - np.min(points[:, 2])

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

#----------------------------------------------------------------
#---------                  Functions                  ----------
#----------------------------------------------------------------

def select_pointclouds(pointcloud_folder):
    """
    Select all point clouds from the specified folder.

    Args:
    pointcloud_folder: Filepath to search in.

    Returns:
    pointclouds: List of point cloud filepaths.
    """
    pointclouds = []
    for pointcloud in os.listdir(pointcloud_folder):
        pointcloud_f = os.path.join(pointcloud_folder, pointcloud)
        pointclouds.append(pointcloud_f)
    return pointclouds

def check_if_data_is_augmented_already(pointclouds):
    """
    Check if any of the point clouds in the list are augmentations.

    Args:
    pointclouds: List of point cloud file paths.

    Returns:
    True/False
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

def get_species_distribution_fwf(selected_pointclouds, selected_fwf_pointclouds):
    """
    Create an ordered list of species present in lists of regular and fwf point clouds.

    Args:
    selected_pointclouds: List of point cloud file paths.
    selected_fwf_pointclouds: List of FWF point cloud file paths.

    Returns:
    species_list: Ordered list of species names.
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

def get_species_dependent_pointcloud_pairs_fwf(species_to_use, selected_pointclouds, selected_fwf_pointclouds, pc_size, capsel):
    """
    Checks a list of point clouds for their species and number of points and sorts out irrelevant point clouds.

    Args:
    species_to_use: Filtered list of species after applying the elimination threshold.
    selected_pointclouds: List of point cloud file paths.
    selected_fwf_pointclouds: List of FWF point cloud file paths.
    pc_size: Point cloud resampling target number of points.
    capsel: User-specified acquisition method.

    Returns:
    species_pairs: Ordered list of regular and FWF point cloud pairs of the filtered species.
    """
    species_pairs = []
    for pointcloud in selected_pointclouds:
        filename = os.path.split(pointcloud)[1]
        tree_id = filename.split("_")[0]
        species = filename.split("_")[2]
        pc_num = filename.split("_")[5]
        if species in species_to_use:
            for fwf_pointcloud in selected_fwf_pointclouds:
                fwf_filename = os.path.split(fwf_pointcloud)[1]
                fwf_tree_id = fwf_filename.split("_")[0]
                fwf_species = fwf_filename.split("_")[2]
                fwf_pc_num = fwf_filename.split("_")[5]
                if fwf_species in species_to_use:
                    if tree_id == fwf_tree_id and pc_num in fwf_pc_num:
                        species_pairs.append([pointcloud, fwf_pointcloud])
                else:
                    if os.path.isfile(fwf_pointcloud):
                        os.remove(fwf_pointcloud)
        else:
            if os.path.isfile(pointcloud):
                os.remove(pointcloud)
    if capsel == "ALS" or capsel == "ALL":
        return species_pairs
    else:
        new_species_pairs = []
        for pair in species_pairs:
            las_file = lp.read(pair[0])
            las_points = np.vstack((las_file.x, las_file.y, las_file.z)).transpose()
            if len(las_points) < pc_size:
                if os.path.isfile(pair[0]):
                    os.remove(pair[0])
                if os.path.isfile(pair[1]):
                    os.remove(pair[1])
            else:
                new_species_pairs.append(pair)
        return new_species_pairs

def get_species_dependent_pointclouds(species_to_use, selected_pointclouds, pc_size, capsel):
    """
    Checks a list of point clouds for their species and number of points and sorts out irrelevant point clouds.

    Args:
    species_to_use: Filtered list of species after applying the elimination threshold.
    selected_pointclouds: List of point cloud file paths.
    pc_size: Point cloud resampling target number of points.
    capsel: User-specified acquisition method.

    Returns:
    species_pairs: Ordered list of point clouds of the filtered species.
    """
    species_clouds = []
    for pointcloud in selected_pointclouds:
        filename = os.path.split(pointcloud)[1]
        species = filename.split("_")[2]
        if species in species_to_use:
            las_file = lp.read(pointcloud)
            las_points = np.vstack((las_file.x, las_file.y, las_file.z)).transpose()
            if len(las_points) < pc_size:
                if os.path.isfile(pointcloud):
                    os.remove(pointcloud)
            else:
                species_clouds.append(pointcloud)
        else:
            if os.path.isfile(pointcloud):
                os.remove(pointcloud)
    return species_clouds

def eliminate_underrepresented_species(species_list, user_spec_percentage=5.0):
    """
    Eliminates all species with less than x% representation.

    Args:
    species_list: List of tree species names.
    user_spec_percentage: User-specified elimination percentage.

    Returns:
    decently_represented_species: Species names of species above the threshold.
    represented_species_distribution: Distribution of species across the entire dataset.
    """
    decently_represented_species = []
    represented_species_distribution = []
    label_counts = Counter(species_list)
    for label, count in label_counts.items():
        percentage = calculate_percentage(count, len(species_list))
        if percentage >= user_spec_percentage:
            decently_represented_species.append(label)
            represented_species_distribution.append([label, count])
    return decently_represented_species, represented_species_distribution

def calculate_percentage(abs_num, tot_num):
    """
    Calculates the percentage of an absolute number present in a total amount.

    Args:
    abs_num: Absolute number of elements.
    tot_num: Total number of elements.

    Returns:
    (abs_num / tot_num) * 100: Percentage of elements in the total amount.
    """
    if abs_num == 0:
        return 0.0
    else:
        return (abs_num / tot_num) * 100
    
def get_maximum_distribution(spec_distr):
    """
    Eliminates all species with less than x% representation.

    Args:
    spec_distr: Distribution of species across the entire dataset.

    Returns:
    np.ceil(max): Highest number of samples for any species in the dataset.
    """
    max = 0
    for row in spec_distr:
        label = row[0]
        distr = row[1]
        if distr >= max:
            max = distr
    return np.ceil(max)

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

def get_species_for_pointcloud(species_pc):
    """
    Retrieves species of a point cloud pair.

    Args:
    species_pc: Individual tree point cloud.

    Returns:
    species: The species of the point cloud.
    """
    filename = os.path.split(species_pc)[1]
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
    for spec_num in species_distribution:
        current_spec = spec_num[0]
        if current_spec == species:
            abs_num = spec_num[1]
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
        logging.info("Successfully loaded point cloud %s", file_path)
    except OSError as e:
        logging.error("Error loading file %s: %s", file_path, e)
        raise
    return points

def pick_random_angle():
    """
    Picks a random angle between 0 and 360 degrees, incements in 15 degree steps.

    Returns:
    random_angle: Generated angle in degrees.
    """
    random_index = random.randint(0, 24)
    random_angle = random_index * 15
    return random_angle

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
    new_scale = [(max_x - min_x) / (2**31 - 1), 
                 (max_y - min_y) / (2**31 - 1), 
                 (max_z - min_z) / (2**31 - 1)]
    las.header.offset = new_offset
    las.header.scale = new_scale
    logging.debug("Updated Scale: %s", new_scale)
    logging.debug("Updated Offset: %s", new_offset)

def augment_species_pointclouds_fwf(species_pc_pairs, max_representation, species_distribution, max_scale, pc_path_selection, fwf_path_selection):
    """
    Augments regular and FWF las point clouds.

    Args:
    species_pc_pairs: List of regular and FWf point cloud pairs.
    max_representation: Highest absolute number of samples for any species in the dataset.
    species_distribution: Distribution of species across the entire dataset.
    max_scale: User-specified scaling factor range.
    pc_path_selection: Savepath for augmented regular point clouds.
    fwf_path_selection: Savepath for augmented FWF point clouds.
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
            angle = pick_random_angle()
            exported_points_reg = reg_points
            exported_points_fwf = fwf_points
            rotated_reg_pc = rotate_point_cloud(exported_points_reg, angle)
            rotated_fwf_pc = rotate_point_cloud(exported_points_fwf, angle)
            scale_factors = np.random.uniform(1 - max_scale, 1 + max_scale, size=3)
            scaled_rotated_reg_pc = scale_point_cloud(rotated_reg_pc, scale_factors)
            scaled_rotated_fwf_pc = scale_point_cloud(rotated_fwf_pc, scale_factors)
            adjust_las_header(outFile_r, scaled_rotated_reg_pc)
            adjust_las_header(outFile_f, scaled_rotated_fwf_pc)
            outFile_r.x = scaled_rotated_reg_pc[:, 0]
            outFile_r.y = scaled_rotated_reg_pc[:, 1]
            outFile_r.z = scaled_rotated_reg_pc[:, 2]
            outFile_f.x = scaled_rotated_fwf_pc[:, 0]
            outFile_f.y = scaled_rotated_fwf_pc[:, 1]
            outFile_f.z = scaled_rotated_fwf_pc[:, 2]
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

def augment_species_pointclouds(species_pcs, max_representation, species_distribution, max_scale, pc_path_selection):
    """
    Augments las point clouds.

    Args:
    species_pc_pairs: List of regular and FWf point cloud pairs.
    max_representation: Highest absolute number of samples for any species in the dataset.
    species_distribution: Distribution of species across the entire dataset.
    max_scale: User-specified scaling factor range.
    pc_path_selection: Savepath for augmented point clouds.
    """
    pc_index = 0
    for pointcloud in species_pcs:
        current_species = get_species_for_pointcloud(pointcloud)
        current_species_amount = get_abs_num(current_species, species_distribution)
        upscale_fac = get_upscale_factor(current_species_amount, max_representation)
        pc_filepath = os.path.dirname(pointcloud)
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
            angle = pick_random_angle()
            exported_points_pc = pc_points
            rotated_pc = rotate_point_cloud(exported_points_pc, angle)
            scale_factors = np.random.uniform(1 - max_scale, 1 + max_scale, size=3)
            scaled_rotated_pc = scale_point_cloud(rotated_pc, scale_factors)
            adjust_las_header(outFile_p, scaled_rotated_pc)
            logging.debug("Number of points after transformation: %s", len(scaled_rotated_pc))
            outFile_p.x = scaled_rotated_pc[:, 0]
            outFile_p.y = scaled_rotated_pc[:, 1]
            outFile_p.z = scaled_rotated_pc[:, 2]
            new_filename_pc = filename_pc + "_" + "aug0" + str(pc_index) + str(i) + "." + pc_name_extension
            logging.info("Created point cloud %s!", new_filename_pc)
            savepath_pc = os.path.join(pc_path_selection + "/" + new_filename_pc)
            save_point_cloud(savepath_pc, pc, outFile_p)

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
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.rotate(R, center=(0, 0, 0))
    rotated_point_cloud = np.asarray(pcd.points)
    return rotated_point_cloud

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

def augment_selection_fwf(pointclouds, fwf_pointclouds, elimination_percentage, max_pc_scale, pc_path_selection, fwf_path_selection, pc_size, capsel):
    """
    Main utility for augmentation of point clouds with FWF data present.

    Args:
    pointclouds: List of pooint cloud file paths.
    fwf_pointclouds: List of FWF point cloud file paths.
    elimination_percentage: User-specified elimination threshold.
    max_pc_scale: User-specified scaling factor.
    pc_path_selection: Savepath for regular point clouds.
    fwf_path_selection: Savepath for FWF point clouds.
    pc_size: User-specified resampling target.
    capsel: User-specified acquisition selection.
    """
    if check_if_data_is_augmented_already(pointclouds) == False:
        species_list = get_species_distribution_fwf(pointclouds, fwf_pointclouds)
        species_to_use, species_distribution = eliminate_underrepresented_species(species_list, elimination_percentage)
        species_pc_pairs = get_species_dependent_pointcloud_pairs_fwf(species_to_use, pointclouds, fwf_pointclouds, pc_size, capsel)
        max_representation = get_maximum_distribution(species_distribution)
        augment_species_pointclouds_fwf(species_pc_pairs, max_representation, species_distribution, max_pc_scale, pc_path_selection, fwf_path_selection)
    else:
        logging.info("Augmented data found, loading!")

def augment_selection(pointclouds, elimination_percentage, max_pc_scale, pc_path_selection, pc_size, capsel):
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
        species_list = get_species_distribution(pointclouds)
        species_to_use, species_distribution = eliminate_underrepresented_species(species_list, elimination_percentage)
        species_pointclouds = get_species_dependent_pointclouds(species_to_use, pointclouds, pc_size, capsel)
        max_representation = get_maximum_distribution(species_distribution)
        augment_species_pointclouds(species_pointclouds, max_representation, species_distribution, max_pc_scale, pc_path_selection)
    else:
        logging.info("Augmented data found, loading!")

def generate_colored_images(IMG_SIZE, las_working_folder, img_working_folder):
    """
    Main utility function for the generation of colored depth images.

    Args:
    IMG_SIZE: User-specified network input image size (224)
    las_working_folder: Filepath to las point clouds.
    img_working_folder: Savepath for generated images.
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
            voxels = create_voxel_grid_from_las(pc)
            vox_pos_list, max_img_size = get_voxel_positions(voxels)
            zero_frontal_image, zero_sideways_image = create_empty_images(max_img_size)
            frontal_image, sideways_image = fill_and_scale_empty_images(vox_pos_list, zero_frontal_image, zero_sideways_image)
            save_voxelized_pointcloud_images(IMG_SIZE, frontal_image, sideways_image, tree_id, species, method, date, ind_id, leaf_cond, img_working_folder, zero_frontal_image, str(pcid), augnum)
            pcid+=1
            logging.info("Generated frontal and sideways views of point cloud %s!", pcid)
    else:
        pass

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
    
def create_voxel_grid_from_las(pointcloud):
    """
    Creates a voxel grid with voxel size 0.03 for a point cloud.

    Args:
    pointcloud: Array of las point cloud points.

    Returns:
    vox_grid: Voxel grid with voxel size 0.03
    """
    points = np.vstack([pointcloud.x, pointcloud.y, pointcloud.z]).transpose()
    pcd_las_o3d = o3d.geometry.PointCloud()
    pcd_las_o3d.points = o3d.utility.Vector3dVector(points)
    R = pcd_las_o3d.get_rotation_matrix_from_xyz((-1.5, 0, 0))
    pcd_las_o3d.rotate(R, center=(0, 0, 0))
    vox_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_las_o3d, voxel_size=0.03)
    return vox_grid

def get_voxel_positions(voxel_grid):
    """
    Gets indices for individual voxels.

    Args:
    voxel_grid: Voxel grid with voxel size 0.03.

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

def fill_and_scale_empty_images(voxel_positions_list, empty_image_frontal, empty_image_sideways):
    """
    Fills empty image arrays with values according to the voxel positions.

    Args:
    voxel_positions_list: List of individual voxel positions.
    empty_image_frontal: Zero-array of dimensions maximum_image_size X maximum_image_size
    empty_image_sideways: Zero-array of dimensions maximum_image_size X maximum_image_size

    Returns:
    image_frontal: Unscaled image array of the frontal view.
    image_sideways: Unscaled image array of the sideways view.
    """
    for voxel_position in voxel_positions_list:
        voxel_position_x = voxel_position[0]
        voxel_position_y = voxel_position[1]
        voxel_position_z = voxel_position[2]
        empty_image_frontal[voxel_position_x+1, voxel_position_y+1] = empty_image_frontal[voxel_position_x+1, voxel_position_y+1] + 1
        empty_image_sideways[voxel_position_y+1, voxel_position_z+1] = empty_image_sideways[voxel_position_y+1, voxel_position_z+1] + 1
    image_frontal = np.interp(empty_image_frontal, (empty_image_frontal.min(), empty_image_frontal.max()), (0, 255))
    image_frontal = np.rot90(image_frontal, k=3, axes=(1,0))
    image_sideways = np.interp(empty_image_sideways, (empty_image_sideways.min(), empty_image_sideways.max()), (0, 255))
    image_sideways = np.rot90(image_sideways, k=2, axes=(1,0))
    return image_frontal, image_sideways

def pad_image(img, pad_t, pad_r, pad_b, pad_l):
    """
    Pads an image around all sides by the given amount of rows/columns of zeros.

    Args:
    img: Image array.
    pad_t: Top padding.
    pad_r: Right padding.
    pad_b: Bottom padding.
    pad_l: Left padding.

    Returns:
    img: Padded image array.
    """
    height, width = img.shape
    pad_left = np.zeros((height, int(pad_l)))
    img = np.concatenate((pad_left, img), axis = 1)
    pad_up = np.zeros((int(pad_t), int(pad_l) + width))
    img = np.concatenate((pad_up, img), axis = 0)
    pad_right = np.zeros((height + int(pad_t), int(pad_r)))
    img = np.concatenate((img, pad_right), axis = 1)
    pad_bottom = np.zeros((int(pad_b), int(pad_l) + width + int(pad_r)))
    img = np.concatenate((img, pad_bottom), axis = 0)
    return img

def center_image(img, empty_image_frontal):
    """
    Crops and image to the contents bounding box and pads it to square dimensions.

    Args:
    img: Image array.
    empty_image_frontal: Zero-array of the same dimensions.

    Returns:
    padded_image: Padded image array, centered in the image frame.
    """
    col_sum = np.where(np.sum(img, axis=0) > 0)
    row_sum = np.where(np.sum(img, axis=1) > 0)
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]
    cropped_image = img[y1:y2, x1:x2]
    zero_axis_fill = (empty_image_frontal.shape[0] - cropped_image.shape[0])
    one_axis_fill = (empty_image_frontal.shape[1] - cropped_image.shape[1])
    top = zero_axis_fill / 2
    bottom = zero_axis_fill - top
    left = one_axis_fill / 2
    right = one_axis_fill - left
    padded_image = pad_image(cropped_image, top, left, bottom, right)
    return padded_image

def save_colored_image(image, id, species, method, date, ind_id, leaf_cond, angle, pcid, augnum, SAVE_DIR):
    """
    Saves an image array as a .tiff file.

    Args:
    image: Image array.
    id: Tree id.
    species: Tree species of the source point cloud.
    method: Capture method of the source point cloud.
    date: Capture date of the source point cloud.
    ind_id: Individual point cloud id.
    leaf_cond: Leaf-condition of the source point cloud.
    angle: Frontal/Sideways.
    pcid: Point cloud id.
    augnum: Augmentation number of the source point cloud.
    SAVE_DIR: Savepath for the image.

    """
    cmap = plt.cm.gist_stern
    norm = plt.Normalize(vmin=image.min(), vmax=image.max())
    image = cmap(norm(image))
    save_path = os.path.join(SAVE_DIR + "/" + str(id) + "_" + species + "_" + method + "_" + date + "_" + str(ind_id) + "_" + leaf_cond + "_" + angle + "_" + pcid + "_" + augnum + ".tiff")
    plt.imsave(save_path, image)
    
def save_voxelized_pointcloud_images(IMG_SIZE, image_frontal, image_sideways, id, species, method, date, ind_id, leaf_cond, SAVE_DIR, empty_image_frontal, pointcloud_id, augmentation_number):
    """
    Main utility function for the saving of the colored depth images.

    Args:
    IMG_SIZE: Network input image size (224).
    image_frontal: Frontal image array.
    image_sideways: Sideways image array.
    id: Tree id.
    species: Tree species of the source point cloud.
    method: Capture method of the source point cloud.
    date: Capture date of the source point cloud.
    ind_id: Individual point cloud id.
    leaf_cond: Leaf-condition of the source point cloud.
    pointcloud_id: Point cloud id.
    empty_image_frontal: Zero array of image array dimensions.
    augnum: Augmentation number of the source point cloud.
    SAVE_DIR: Savepath for the image.
    """
    image_frontal_to_save = center_image(image_frontal, empty_image_frontal)
    image_frontal_resized = cv2.resize(image_frontal_to_save, (IMG_SIZE, IMG_SIZE))
    save_colored_image(image_frontal_resized, id, species, method, date, ind_id, leaf_cond, "frontal", pointcloud_id, augmentation_number, SAVE_DIR)
    image_sideways_to_save = center_image(image_sideways, empty_image_frontal)
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

def filter_data_for_selection_fwf(selected_pointclouds, selected_fwf_pointclouds, selected_images, min_repres):
    """
    Filters data based on dataset representation with FWF data present.

    Args:
    selected_pointclouds: Point clouds selected based on the specifications.
    selected_fwf_pointclouds: FWF point clouds selected based on the specifications.
    selected_images: Images selected based on the specifications.
    min_repres: Minimum representation for any class of the dataset.

    Returns:
    filtered_pointclouds: Point clouds selected based on the specifications.
    filtered_fwf_pointclouds: FWF point clouds selected based on the specifications.
    filtered_images: Images selected based on the specifications.
    """
    filtered_pointclouds = filter_classes_by_representation(selected_pointclouds, min_repres)
    filtered_fwf_pointclouds = filter_classes_by_representation(selected_fwf_pointclouds, min_repres)
    filtered_images = filter_images_by_representation(selected_images, min_repres)
    return filtered_pointclouds, filtered_fwf_pointclouds, filtered_images

def filter_data_for_selection(selected_pointclouds, selected_images, min_repres):
    """
    Filters data based on dataset representation.

    Args:
    selected_pointclouds: Point clouds selected based on the specifications.
    selected_images: Images selected based on the specifications.
    min_repres: Minimum representation for any class of the dataset.

    Returns:
    filtered_pointclouds: Point clouds selected based on the specifications.
    filtered_images: Images selected based on the specifications.
    """
    filtered_pointclouds = filter_classes_by_representation(selected_pointclouds, min_repres)
    filtered_images = filter_images_by_representation(selected_images, min_repres)
    return filtered_pointclouds, filtered_images

def filter_classes_by_representation(selected_pointclouds, threshold):
    """
    Filters point clouds based on dataset representation.

    Args:
    selected_pointclouds: Point clouds selected based on the specifications.
    threshold: Minimum representation for any class of the dataset.

    Returns:
    filtered_pointclouds: Point clouds selected based on the specifications.
    """
    tree_labels = np.array(get_labels_for_trees(selected_pointclouds))
    total_samples = len(tree_labels)
    label_counts = Counter(tree_labels)
    class_percentages = {label: count / total_samples * 100 for label, count in label_counts.items()}
    logging.info(f"Class percentages: {class_percentages}")
    #valid_classes = ["FagSyl", "PicAbi", "PseMen", "QuePet"]
    valid_classes = [label for label, percentage in class_percentages.items() if percentage >= threshold]
    filtered_pointclouds = [pc for pc in selected_pointclouds if pc.split("/")[-1].split(".")[0].split("_")[2] in valid_classes]
    return filtered_pointclouds

def filter_images_by_representation(selected_images, threshold):
    """
    Filters images based on dataset representation.

    Args:
    selected_images: Images selected based on the specifications.
    threshold: Minimum representation for any class of the dataset.

    Returns:
    filtered_images: Images selected based on the specifications.
    """
    tree_labels = np.array(get_labels_for_trees_from_images(selected_images))
    total_samples = len(tree_labels)
    label_counts = Counter(tree_labels)
    class_percentages = {label: count / total_samples * 100 for label, count in label_counts.items()}
    #valid_classes = ["FagSyl", "PicAbi", "PseMen", "QuePet"]
    valid_classes = [label for label, percentage in class_percentages.items() if percentage >= threshold]
    filtered_images = [im for im in selected_images if im.split("/")[-1].split(".")[0].split("_")[1] in valid_classes]
    return filtered_images

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

def get_labels_for_trees_from_images(selected_images):
    """
    Retrieves tree labels from a list of image file paths.

    Args:
    selected_images: List of image file paths.

    Returns:
    tree_labels: List of tree species names.
    """
    tree_labels = []
    for image in selected_images:
        filename_full = image.split("/")[-1].split(".")[0]
        tree_species = filename_full.split("_")[1]
        tree_labels.append(tree_species)
    return tree_labels

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

def resample_point_cloud_density_based(points, num_points, num_segments=20):
    """
    Resamples point cloud segments density-based.

    Args:
    points: Array of als point cloud points.
    num_points: Resampling target number of points.
    num_segments=20: Number of height segments a point cloud is divided into

    Returns:
    resampled_points: Resampeld array of las point cloud points.
    """
    logging.info("Resampling a pointcloud from %s points to %s points!", len(points), num_points)
    points = np.array(points, dtype=np.float64)
    min_height = np.min(points[:, 2])
    max_height = np.max(points[:, 2])
    segment_height = (max_height - min_height) / num_segments
    segment_indices = np.floor((points[:, 2] - min_height) / segment_height).astype(int)
    segment_indices = np.clip(segment_indices, 0, num_segments - 1)
    segment_counts = np.bincount(segment_indices, minlength=num_segments)
    total_points = np.sum(segment_counts)
    segment_counts = segment_counts.astype(float)
    num_points = float(num_points)
    points_per_segment = (segment_counts / total_points) * num_points
    resampled_points = []
    cumulative_segment_counts = np.cumsum(segment_counts).astype(int)
    segments = np.split(points, cumulative_segment_counts[:-1])
    for i, segment in enumerate(segments):
        if len(segment) > 0:
            num_sampled_points = int(points_per_segment[i])
            if len(segment) > num_sampled_points:
                indices = np.random.choice(len(segment), num_sampled_points, replace=False)
            else:
                indices = np.random.choice(len(segment), num_sampled_points, replace=True)
            resampled_points.extend(segment[indices])
    while len(resampled_points) < num_points:
        additional_points_needed = int(num_points - len(resampled_points))
        indices = np.random.choice(points.shape[0], additional_points_needed, replace=True)
        resampled_points.extend(points[indices])
    resampled_points = np.array(resampled_points)[:int(num_points)]
    return resampled_points

def generate_metrics_for_selected_pointclouds_fwf(selected_pointclouds, filtered_fwf_pointclouds, metrics_dir, capsel, growsel):
    """
    Generates numerical features for regular and FWF point clouds.

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
    if workspace_setup.get_are_fwf_pcs_extracted(metrics_dir) == False:
        all_metrics = []
        for i in range(len(selected_pointclouds)):
            las_points = load_point_cloud(selected_pointclouds[i])
            fwf_file = load_point_cloud_file(filtered_fwf_pointclouds[i])
            metrics, feature_names = compute_combined_metrics_fwf(las_points, fwf_file)
            arrmetrics = np.asarray(metrics)
            all_metrics.append(arrmetrics)
            logging.info("Computed %s/%s metrics!", i+1, len(selected_pointclouds))
        save_metrics_to_csv_pandas(all_metrics, metrics_path)
        combined_metrics = np.vstack(all_metrics)
    else:
        logging.info("Previously generated metrics found, importing!")
        combined_metrics = load_metrics_from_path(metrics_path)
        feature_names = ["height_quantile_25", "height_quantile_50", "height_quantile_75", "dens0", "dens1", "dens2", "dens3", "dens4", "dens5", "dens6", "dens7", "dens8", "dens9", "dec0",
                     "dec1", "dec2", "dec3", "dec4", "dec5", "dec6", "dec7", "dec8", "max_crown_diameter", "clustering_degree", "intensity_mean", "intensity_std", "intensity_skewness", "intensity_kurtosis", "mean_pulse_widths",
                     "max_height", "crown_height", "crown_volume", "segdens0",
                     "segdens1", "segdens2", "segdens3", "segdens4", "segdens5", "segdens6", "segdens7", "tree_height", "highest_branch", "lowest_branch", "longest_spread", "longest_cross_spread",
                     "equivalent_crown_diameter", "canopy_width_x", "canopy_width_y", "canopy_volume", "point_density", "lai", "canopy_closure", "crown_base_height", "std_dev_height",
                     "height_kurtosis", "height_skewness", "crown_area", "crown_perimeter", "crown_volume_to_height_ratio", "canopy_cover_fraction", "stem_volume", "canopy_base_height", "fwhm", "echo_width",
                     "surface_area", "surface_to_volume_ratio", "avg_nn_dist", "fract_dimension", "bb_dims", "crown_shape_indices"]
    return combined_metrics, feature_names

def generate_metrics_for_selected_pointclouds(selected_pointclouds, metrics_dir, capsel, growsel):
    """
    Generates numerical features for point clouds.

    Args:
    selected_pointclouds: List of point cloud paths.
    metrics_dir: Savepath for numerical features.
    capsel: User-specified acquisition method.
    growsel: User-specified leaf-condition.

    Returns:
    combined_metrics: Array of numerical features for each individual tree.
    """
    savename = f"training_generated_metrics_{capsel}_{growsel}.csv"
    metrics_path = main_utils.join_paths(metrics_dir, savename)
    if workspace_setup.get_are_fwf_pcs_extracted(metrics_dir) == False:
        all_metrics = []
        for i in range(len(selected_pointclouds)):
            las_points = load_point_cloud(selected_pointclouds[i])
            metrics, feature_names = compute_combined_metrics(las_points)
            arrmetrics = np.asarray(metrics)
            all_metrics.append(arrmetrics)
            logging.info("Computed %s/%s metrics!", i+1, len(selected_pointclouds))
        save_metrics_to_csv_pandas(all_metrics, metrics_path)
        combined_metrics = np.vstack(all_metrics)
    else:
        logging.info("Previously generated metrics found, importing!")
        combined_metrics = load_metrics_from_path(metrics_path)
        feature_names = ["height_quantile_25", "height_quantile_50", "height_quantile_75", "dens0", "dens1", "dens2", "dens3", "dens4", "dens5", "dens6", "dens7", "dens8", "dens9", "dec0",
                     "dec1", "dec2", "dec3", "dec4", "dec5", "dec6", "dec7", "dec8", "max_crown_diameter", "clustering_degree", "max_height", "crown_height", "crown_volume", "segdens0",
                     "segdens1", "segdens2", "segdens3", "segdens4", "segdens5", "segdens6", "segdens7", "tree_height", "highest_branch", "lowest_branch", "longest_spread", "longest_cross_spread",
                     "equivalent_crown_diameter", "canopy_width_x", "canopy_width_y", "canopy_volume", "point_density", "lai", "canopy_closure", "crown_base_height", "std_dev_height",
                     "height_kurtosis", "height_skewness", "crown_area", "crown_perimeter", "crown_volume_to_height_ratio", "canopy_cover_fraction", "stem_volume", "canopy_base_height",
                     "surface_area", "surface_to_volume_ratio", "avg_nn_dist", "fract_dimension", "bb_dims", "crown_shape_indices"]
    return combined_metrics, feature_names

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
    max_height = compute_maximum_height(points)
    crown_height = compute_crown_height(points)
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
    metrics.extend([
        float(height_quantile_25), float(height_quantile_50), float(height_quantile_75),
        float(dens0), float(dens1), float(dens2), float(dens3), float(dens4), float(dens5), float(dens6), float(dens7), float(dens8), float(dens9),
        float(dec0), float(dec1), float(dec2), float(dec3), float(dec4), float(dec5), float(dec6), float(dec7), float(dec8),
        float(max_crown_diameter), float(clustering_degree), float(intensity_mean), float(intensity_std), 
        float(intensity_skewness), float(intensity_kurtosis), float(mean_pulse_widths), float(max_height),
        float(crown_height), float(crown_volume), float(segdens0), float(segdens1), float(segdens2), float(segdens3), 
        float(segdens4), float(segdens5), float(segdens6), float(segdens7), float(tree_height), float(highest_branch), 
        float(lowest_branch), float(longest_spread), float(longest_cross_spread), float(equivalent_crown_diameter),
        float(canopy_width_x), float(canopy_width_y), float(canopy_volume), float(point_density), float(lai),
        float(canopy_closure), float(crown_base_height), float(std_dev_height), float(height_kurtosis),
        float(height_skewness), float(crown_area), float(crown_perimeter), float(crown_volume_to_height_ratio),
        float(canopy_cover_fraction), float(stem_volume), float(canopy_base_height), float(fwhm), float(echo_width),
        float(surface_area), float(surface_to_volume_ratio), float(avg_nn_dist), float(fract_dimension),
        float(bb_dims), float(crown_shape_indices)
    ])
    feature_names = ["height_quantile_25", "height_quantile_50", "height_quantile_75", "dens0", "dens1", "dens2", "dens3", "dens4", "dens5", "dens6", "dens7", "dens8", "dens9", "dec0",
                     "dec1", "dec2", "dec3", "dec4", "dec5", "dec6", "dec7", "dec8", "max_crown_diameter", "clustering_degree", "intensity_mean", "intensity_std", "intensity_skewness", "intensity_kurtosis", "mean_pulse_widths",
                     "max_height", "crown_height", "crown_volume", "segdens0",
                     "segdens1", "segdens2", "segdens3", "segdens4", "segdens5", "segdens6", "segdens7", "tree_height", "highest_branch", "lowest_branch", "longest_spread", "longest_cross_spread",
                     "equivalent_crown_diameter", "canopy_width_x", "canopy_width_y", "canopy_volume", "point_density", "lai", "canopy_closure", "crown_base_height", "std_dev_height",
                     "height_kurtosis", "height_skewness", "crown_area", "crown_perimeter", "crown_volume_to_height_ratio", "canopy_cover_fraction", "stem_volume", "canopy_base_height", "fwhm", "echo_width",
                     "surface_area", "surface_to_volume_ratio", "avg_nn_dist", "fract_dimension", "bb_dims", "crown_shape_indices"]
    return metrics, feature_names

def compute_combined_metrics(points):
    """
    Generates the numerical features for a point cloud.

    Args:
    points: Array of las file point cloud points.

    Returns:
    metrics: List of numerical features for the point cloud.
    """
    metrics = []
    height_quantile_25 = compute_height_quantile(points, 25)
    height_quantile_50 = compute_height_quantile(points, 50)
    height_quantile_75 = compute_height_quantile(points, 75)
    dens0, dens1, dens2, dens3, dens4, dens5, dens6, dens7, dens8, dens9 = compute_point_density_normalized_height(points)
    dec0, dec1, dec2, dec3, dec4, dec5, dec6, dec7, dec8 = compute_height_density_deciles(points)
    max_crown_diameter = compute_maximum_crown_diameter(points)
    clustering_degree = compute_points_relative_clustering_degree(points)
    max_height = compute_maximum_height(points)
    crown_height = compute_crown_height(points)
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
    metrics.extend([
        float(height_quantile_25), float(height_quantile_50), float(height_quantile_75),
        float(dens0), float(dens1), float(dens2), float(dens3), float(dens4), float(dens5), float(dens6), float(dens7), float(dens8), float(dens9),
        float(dec0), float(dec1), float(dec2), float(dec3), float(dec4), float(dec5), float(dec6), float(dec7), float(dec8),
        float(max_crown_diameter), float(clustering_degree), float(max_height),
        float(crown_height), float(crown_volume), float(segdens0), float(segdens1), float(segdens2), float(segdens3), 
        float(segdens4), float(segdens5), float(segdens6), float(segdens7), float(tree_height), float(highest_branch), 
        float(lowest_branch), float(longest_spread), float(longest_cross_spread), float(equivalent_crown_diameter),
        float(canopy_width_x), float(canopy_width_y), float(canopy_volume), float(point_density), float(lai),
        float(canopy_closure), float(crown_base_height), float(std_dev_height), float(height_kurtosis),
        float(height_skewness), float(crown_area), float(crown_perimeter), float(crown_volume_to_height_ratio),
        float(canopy_cover_fraction), float(stem_volume), float(canopy_base_height), float(surface_area),
        float(surface_to_volume_ratio), float(avg_nn_dist), float(fract_dimension),
        float(bb_dims), float(crown_shape_indices)
    ])
    feature_names = ["height_quantile_25", "height_quantile_50", "height_quantile_75", "dens0", "dens1", "dens2", "dens3", "dens4", "dens5", "dens6", "dens7", "dens8", "dens9", "dec0",
                     "dec1", "dec2", "dec3", "dec4", "dec5", "dec6", "dec7", "dec8", "max_crown_diameter", "clustering_degree", "max_height", "crown_height", "crown_volume", "segdens0",
                     "segdens1", "segdens2", "segdens3", "segdens4", "segdens5", "segdens6", "segdens7", "tree_height", "highest_branch", "lowest_branch", "longest_spread", "longest_cross_spread",
                     "equivalent_crown_diameter", "canopy_width_x", "canopy_width_y", "canopy_volume", "point_density", "lai", "canopy_closure", "crown_base_height", "std_dev_height",
                     "height_kurtosis", "height_skewness", "crown_area", "crown_perimeter", "crown_volume_to_height_ratio", "canopy_cover_fraction", "stem_volume", "canopy_base_height",
                     "surface_area", "surface_to_volume_ratio", "avg_nn_dist", "fract_dimension", "bb_dims", "crown_shape_indices"]
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
    print("\nClass Distribution Statistics:\n")
    print(df.to_string(index=False))

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
    print(f"Class distribution before balancing: {class_counts}")
    print(f"Downsampling all classes to {min_count} samples.")
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
    print(f"Class distribution after balancing: {balanced_counts}")
    return X_pc, X_metrics, X_img_1, X_img_2, y_pc

def resample_pointcloud(pointclouds, target_num_points, iteration):
    """
    Resample a point cloud to a target number of points using non-uniform grid sampling
    followed by farthest point sampling (FPS).

    Args:
        pointclouds (list of numpy.ndarray): List of input point clouds as Nx3 arrays.
        target_num_points (int): Target number of points.
        iteration (int): Index of the point cloud to resample.

    Returns:
        numpy.ndarray: Resampled point cloud with target_num_points points.
    """
    method = "NONE"
    pointcloud = pointclouds[iteration]
    num_points = len(pointcloud)
    
    if num_points > target_num_points:
        # Downsample using farthest point sampling
        sampled_pointcloud = farthest_point_sampling(pointcloud, target_num_points)
        method = "FPS"
    elif num_points < target_num_points:
        # Upsample by interpolating new points and combining with original points
        extra_points_needed = target_num_points - num_points
        interpolated_points = interpolate_points(pointcloud, extra_points_needed)
        sampled_pointcloud = np.vstack((pointcloud, interpolated_points))
        method = "INTERP"
    else:
        # No resampling needed
        sampled_pointcloud = pointcloud
        method = "NONE"

    logging.info(f"Resampled point cloud {iteration}/{len(pointclouds)} using {method} from {num_points} points to {len(sampled_pointcloud)} points")
    if len(sampled_pointcloud) < target_num_points:
        print("\033[31mPointcloud has an insufficient number of points!.\033[0m")
    else:
        print("\033[32mNumber of points is fine.\033[0m")
    return sampled_pointcloud

def farthest_point_sampling(pointcloud, num_samples):
    """
    Perform farthest point sampling (FPS) on a point cloud.

    Args:
        pointcloud (numpy.ndarray): Input point cloud as an Nx3 array.
        num_samples (int): Number of points to sample.

    Returns:
        numpy.ndarray: Sampled point cloud as a num_samplesx3 array.
    """
    num_points = pointcloud.shape[0]
    sampled_indices = [np.random.randint(0, num_points)]  # Start with a random point
    distances = np.linalg.norm(pointcloud - pointcloud[sampled_indices[0]], axis=1)
    for _ in range(1, num_samples):
        farthest_point_idx = np.argmax(distances)
        sampled_indices.append(farthest_point_idx)
        new_distances = np.linalg.norm(pointcloud - pointcloud[farthest_point_idx], axis=1)
        distances = np.minimum(distances, new_distances)
    return pointcloud[sampled_indices]

def interpolate_points(pointcloud, extra_points_needed):
    """
    Generate additional points by interpolating between randomly selected pairs of points.

    Args:
        pointcloud (numpy.ndarray): Input point cloud as an Nx3 array.
        extra_points_needed (int): Number of additional points to generate.

    Returns:
        numpy.ndarray: Interpolated points as a (extra_points_needed)x3 array.
    """
    indices = np.random.choice(len(pointcloud), size=(extra_points_needed, 2))
    alpha = np.random.rand(extra_points_needed, 1)
    interpolated_points = alpha * pointcloud[indices[:, 0]] + (1 - alpha) * pointcloud[indices[:, 1]]
    return interpolated_points

def generate_training_data(capsel, growsel, filtered_pointclouds, resampled_pointclouds, combined_metrics, images_frontal, images_sideways, sss_testsize, metrics_dir, rfe_threshold, feature_names):
    """
    Generates training data.

    Args:
    capsel: User-specfied acquisition method.
    growsel: User-specified leaf-condition.
    filtered_pointclouds: List of point cloud filepaths.
    resampled_pointclouds: List of resampled point cloud arrays.
    combined_metrics: Array of numerical features.
    images_frontal: List of frontal view image arrays.
    images_sideways: List of sideways view image arrays.
    sss_testsize: Train/test split ratio.
    metrics_dir: Savepath for numerical features.
    rfe_threshold: Threshold for feature importance.

    Returns:
    X_pc_train: Training point clouds.
    X_pc_val: Validation point clouds.
    X_metrics_train: Training numerical features.
    X_metrics_val: Validation numerical features.
    X_img_1_train: Training frontal images.
    X_img_1_val: Validation frontal images.
    X_img_2_train: Training sideways images.
    X_img_2_val: Validation sideways images.
    y_train: Training labels.
    y_val: Validation labels.
    num_classes: Number of classes in the dataset.
    onehot_to_label_dict: Disctionary to translate one-hot encoded labels to textual labels.
    """
    tree_labels = np.array(get_labels_for_trees(filtered_pointclouds))
    label_encoder = LabelEncoder()
    elimination_labels = label_encoder.fit_transform(tree_labels)
    numeric_tree_labels = elimination_labels.astype(int)
    onehot_to_label_dict = {numeric_tree_labels[i]: tree_labels[i] for i in range(len(tree_labels))}
    rfe_metrics = []
    for file in os.listdir(metrics_dir):
        if "training_rfe" in file:
            rfe_metrics.append(file)
        else:
            pass
    if len(rfe_metrics) > 0:
        rfe_metrics_path = main_utils.join_paths(metrics_dir, rfe_metrics[0])
        combined_eliminated_metrics = load_metrics_from_path(rfe_metrics_path)
        logging.debug("Loaded metrics of shape %s", combined_eliminated_metrics.shape)
    else:
        #combined_eliminated_metrics = perform_recursive_feature_elimination_with_threshold(capsel, growsel, combined_metrics, elimination_labels, metrics_dir, rfe_threshold, feature_names)
        combined_eliminated_metrics, features = perform_statistical_rfe(capsel, growsel, combined_metrics, elimination_labels, metrics_dir, feature_names, p_value_threshold=0.05)
        logging.debug("Metrics shape after Recursive Feature Elimination: %s", combined_eliminated_metrics.shape)
    logging.debug("Tree species to train on: %s", np.unique(tree_labels))
    logging.info("One-Hot encoding labels!")
    logging.info(f"Metrics shape: {combined_eliminated_metrics.shape}")
    encoder = OneHotEncoder(sparse_output=False)
    y_orig = encoder.fit_transform(tree_labels.reshape(-1, 1))
    num_classes = len(encoder.categories_[0])
    X_pc_orig = resampled_pointclouds
    X_metrics_orig = combined_eliminated_metrics
    X_img_1_orig = images_frontal
    X_img_2_orig = images_sideways
    X_pc, X_metrics, X_img_1, X_img_2, y = balance_classes(X_pc_orig, X_metrics_orig, X_img_1_orig, X_img_2_orig, y_orig, onehot_to_label_dict)
    logging.info("Performing Stratified-Shuffle-Split!")
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=42)
    for train_index_temp, pred_index in sss.split(X_pc, np.argmax(y, axis=1)):
        X_pc_train_temp, X_pc_pred = X_pc[train_index_temp], X_pc[pred_index]
        X_metrics_train_temp, X_metrics_pred = X_metrics[train_index_temp], X_metrics[pred_index]
        X_img_1_train_temp, X_img_1_pred = X_img_1[train_index_temp], X_img_1[pred_index]
        X_img_2_train_temp, X_img_2_pred = X_img_2[train_index_temp], X_img_2[pred_index]
        y_train_temp, y_pred = y[train_index_temp], y[pred_index]
    sss2 = StratifiedShuffleSplit(n_splits=5, test_size=sss_testsize, random_state=42)
    for train_index, val_index in sss2.split(X_pc_train_temp, np.argmax(y_train_temp, axis=1)):
        X_pc_train, X_pc_val = X_pc_train_temp[train_index], X_pc_train_temp[val_index]
        X_metrics_train, X_metrics_val = X_metrics_train_temp[train_index], X_metrics_train_temp[val_index]
        X_img_1_train, X_img_1_val = X_img_1_train_temp[train_index], X_img_1_train_temp[val_index]
        X_img_2_train, X_img_2_val = X_img_2_train_temp[train_index], X_img_2_train_temp[val_index]
        y_train, y_val = y_train_temp[train_index], y_train_temp[val_index]
    print_class_distribution(y_train, y_val, y_pred, onehot_to_label_dict)
    return X_pc_train, X_pc_val, X_pc_pred, X_metrics_train, X_metrics_val, X_metrics_pred, X_img_1_train, X_img_1_val, X_img_1_pred, X_img_2_train, X_img_2_val, X_img_2_pred, y_train, y_val, y_pred, num_classes, onehot_to_label_dict

def perform_statistical_rfe(capsel, growsel, X, y, metrics_dir, feature_names, p_value_threshold=0.5, correlation_threshold=0.85):
    """
    Perform Statistical Recursive Feature Elimination with redundancy checks.
    """
    logging.info("Performing Statistical Recursive Feature Elimination!")
    non_constant_mask = (X.std(axis=0) > 0)
    X = X[:, non_constant_mask]
    feature_names = [feature_names[i] for i in range(len(feature_names)) if non_constant_mask[i]]
    iteration = 0
    eliminated_feature_names = []
    while True:
        iteration += 1
        logging.info(f"Iteration {iteration}: Performing statistical test for feature significance.")
        f_scores, p_values = f_classif(X, y)
        logging.debug(f"Iteration {iteration}: F-Scores: {f_scores}")
        logging.debug(f"Iteration {iteration}: P-Values: {p_values}")
        features_to_eliminate = np.where(p_values > p_value_threshold)[0]
        features_to_keep = np.where(p_values <= p_value_threshold)[0]
        eliminated_names = [feature_names[i] for i in features_to_eliminate]
        retained_names = [feature_names[i] for i in features_to_keep]
        logging.info(f"Iteration {iteration}: Eliminated Features: {eliminated_names}")
        logging.info(f"Iteration {iteration}: Retained Features: {len(retained_names)}")
        logging.debug(f"Threshold: {p_value_threshold}")
        if len(features_to_eliminate) == 0:
            logging.info("No further features to eliminate. Stopping RFE.")
            break
        X = X[:, features_to_keep]
        feature_names = retained_names
        eliminated_feature_names.extend(eliminated_names)
    logging.info("Checking for redundant features via correlation.")
    corr_matrix = pd.DataFrame(X).corr(method='spearman')
    to_remove = set()
    for i in range(len(corr_matrix)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                if X[:, i].var() > X[:, j].var():
                    to_remove.add(j)
                else:
                    to_remove.add(i)
    to_keep = [i for i in range(X.shape[1]) if i not in to_remove]
    X = X[:, to_keep]
    feature_names = [feature_names[i] for i in to_keep]
    logging.info(f"Features Removed Due to Correlation: {len(to_remove)}")
    logging.info(f"Final number of features kept: {len(feature_names)}")
    logging.info(f"Features kept: {feature_names}")
    savename = f"training_rfe_generated_metrics_{capsel}_{growsel}.csv"
    metrics_path = os.path.join(metrics_dir, savename)
    save_metrics_to_csv_pandas(X, metrics_path)
    return X, feature_names

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
    print(f"Selected Features: {selected_feature_names}")
    print(f"Eliminated Features: {eliminated_feature_names}")
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
    return X_reduced