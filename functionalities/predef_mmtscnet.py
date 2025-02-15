import time
import datetime
import os
from functionalities import model_utils
from keras_tuner import BayesianOptimization, Objective

def get_closest_timestamp_for_config(timestamp, cap_sel, grow_sel, netpcsize, fwf_av, model_dir, num_classes):
    """
    Finds and loads tuned instance of MMTSCNet for given configuration.

    Args:
    timestamp: Timestamp in string format.
    cap_sel: User-specified acquisition selection.
    grow_sel: User-specified leaf-confition selection.
    netpcsize: Number of points to resample point clouds to.
    fwf_av: True/False - Presence of FWF data.
    model_dir: Save directory for tuned and trained model isntances.
    num_classes: Int - Number of classes to train on.

    Returns:
    closest_time: Timestamp for the last tuned model instace.
    """
    timestamp_format = "%Y%m%d-%H%M%S"
    target_time = datetime.datetime.strptime(timestamp, timestamp_format)
    closest_time = None
    min_time_diff = float("inf")
    for subfolder in os.listdir(model_dir):
        parts = subfolder.split("_")
        if (str(cap_sel) in subfolder and 
        str(grow_sel) in subfolder and 
        str(netpcsize) in subfolder and 
        str(fwf_av) in subfolder and 
        str(num_classes) in subfolder):
            try:
                current_timestamp_str = parts[-1]
                current_time = datetime.datetime.strptime(current_timestamp_str, timestamp_format)
                time_diff = abs((target_time - current_time).total_seconds())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_time = current_time
            except ValueError:
                continue
    return closest_time.strftime(timestamp_format) if closest_time else None

def get_hyperparams_for_config(num_classes, cap_sel, grow_sel, fwf_av, point_cloud_shape, image_shape, metrics_shape, netpcsize, model_dir):
    """
    Retrieves the best hyperparameters for a given model configuration using Bayesian Optimization.

    Args:
        num_classes (int): Number of classification categories.
        cap_sel (str): Acquisition selection criteria.
        grow_sel (str): Leaf-condition selection criteria.
        fwf_av (bool): Whether Full Waveform (FWF) data is available.
        point_cloud_shape (tuple): Shape of input point cloud data.
        image_shape (tuple): Shape of input image data.
        metrics_shape (tuple): Shape of input metrics data.
        netpcsize (int): Number of points to resample point clouds to.
        model_dir (str): Directory where model instances are stored.

    Returns:
        tuple: (Best hyperparameters, optimal learning rate)
    """
    os.chdir(model_dir)
    if fwf_av == True:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        closest_timestamp = get_closest_timestamp_for_config(timestamp, cap_sel, grow_sel, netpcsize, "fwf", model_dir, num_classes)
        dir_name = f'hp-tuning-fwf_{cap_sel}_{grow_sel}_{netpcsize}_{num_classes}_{closest_timestamp}'
        tuner = BayesianOptimization(
            model_utils.CombinedModel(point_cloud_shape, image_shape, metrics_shape, num_classes, netpcsize),
            objective=Objective("val_custom_metric", direction="max"),
            max_trials=7,
            max_retries_per_trial=3,
            max_consecutive_failed_trials=3,
            directory=dir_name,
            project_name='tree_classification'
        )
        tuner.reload()
        hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        optimal_learning_rate = hps.get('learning_rate')
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        closest_timestamp = get_closest_timestamp_for_config(timestamp, cap_sel, grow_sel, netpcsize, "fwf", model_dir)
        dir_name = f'hp-tuning_{cap_sel}_{grow_sel}_{netpcsize}_{num_classes}_{closest_timestamp}'
        tuner = BayesianOptimization(
            model_utils.CombinedModel(point_cloud_shape, image_shape, metrics_shape, num_classes, netpcsize),
            objective=Objective("val_custom_metric", direction="max"),
            max_trials=7,
            max_retries_per_trial=3,
            max_consecutive_failed_trials=3,
            directory=dir_name,
            project_name='tree_classification'
        )
        tuner.reload()
        hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        optimal_learning_rate = hps.get('learning_rate')
    return hps, optimal_learning_rate





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