import time
import datetime
import os
from functionalities import model_utils
from keras_tuner import BayesianOptimization, Objective

def get_closest_timestamp_for_config(timestamp, cap_sel, grow_sel, netpcsize, fwf_av, model_dir, num_classes):
    """
    Finds the folder with the closest timestamp that matches the given configuration parameters.
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
        tuner.get_best_hyperparameters(num_trials=1)[0]
        hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        optimal_learning_rate = hps.get('learning_rate')
    return hps, optimal_learning_rate