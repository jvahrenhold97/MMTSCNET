from functionalities import workspace_setup, preprocessing, model_utils, predef_mmtscnet
import numpy as np
import os
import tensorflow as tf
import logging
import gc
import time
from keras_tuner import BayesianOptimization, Objective
from keras.callbacks import LearningRateScheduler
import keras
from keras.losses import CategoricalCrossentropy

def extract_data(data_dir, work_dir, fwf_av, capsel, growsel):
    """
    Handles input/output operations before data preprocessing.

    This function:
    - Sets up the working directory.
    - Unzips all datasets into the appropriate directories.
    - Prepares the required file structure based on the presence of FWF data.
    - Organizes data for further processing by MMTSCNet.

    Args:
        data_dir (str): Path to the source data directory.
        work_dir (str): Path to the working directory.
        fwf_av (bool): Indicates if Full Waveform (FWF) data is available.
        capsel (str): Acquisition selection criteria.
        growsel (str): Leaf-condition selection criteria.

    Returns:
        list: A list of full paths to processed data files.
    """
    local_pathlist = workspace_setup.create_working_directory(work_dir, fwf_av)
    workspace_setup.unzip_all_datasets(data_dir, local_pathlist, fwf_av)
    if fwf_av == True:
        workspace_setup.create_fpcs(local_pathlist[2], local_pathlist[3])
        full_pathlist = workspace_setup.create_config_directory(local_pathlist, capsel, growsel, fwf_av)
        workspace_setup.extract_single_trees_from_fpc(local_pathlist[3], local_pathlist[1], full_pathlist[6], full_pathlist[7], capsel, growsel)
        return full_pathlist
    else:
        full_pathlist = workspace_setup.create_config_directory(local_pathlist, capsel, growsel, fwf_av)
        workspace_setup.copy_files_for_prediction(full_pathlist[1], full_pathlist[4], capsel, growsel)
        return full_pathlist
    
def preprocess_data(full_pathlist, ssstest, capsel, growsel, elimper, maxpcscale, netpcsize, netimgsize, fwf_av):
    """
    Handles data preprocessing for MMTSCNet, including filtering, augmentation, resampling, and feature extraction.

    This function:
    - Filters and removes underrepresented species.
    - Augments and rescales point cloud data.
    - Extracts metrics and generates training images.
    - Splits the dataset into training, validation, and prediction sets.

    Args:
        full_pathlist (list): Paths to all required input data.
        ssstest (float): Train-test split ratio.
        capsel (str): Acquisition selection criteria.
        growsel (str): Leaf-condition selection criteria.
        elimper (float): Threshold for removing underrepresented species.
        maxpcscale (float): Maximum augmentation scaling for point clouds.
        netpcsize (int): Number of points to resample point clouds to.
        netimgsize (int): Image size in pixels.
        fwf_av (bool): Indicates whether Full Waveform (FWF) data is available.

    Returns:
        tuple: Processed training, validation, and test datasets including point clouds, metrics, images, and labels.
    """
    if fwf_av == True:
        logging.info("Creating Test-Set and removing underrepresented species...")
        if workspace_setup.files_extracted(full_pathlist[10]) == 0:
            preprocessing.remove_insufficient_pointclouds_fwf(full_pathlist[6], full_pathlist[7], netpcsize)
            species_distribution = preprocessing.eliminate_unused_species_fwf(full_pathlist[6], full_pathlist[7], elimper, netpcsize)
            preprocessing.remove_height_outliers_fwf(full_pathlist[6], full_pathlist[7])
            logging.info("Species distribution: %s", species_distribution)
            preprocessing.remove_unmatched_files(full_pathlist[6], full_pathlist[7])
            preprocessing.move_pointclouds_to_preds_fwf(full_pathlist[6], full_pathlist[7], full_pathlist[10], full_pathlist[11])
        else:
            pointclouds = []
            for file in os.listdir(full_pathlist[6]):
                pointclouds.append(file)
            species_list = preprocessing.get_species_distribution(pointclouds)
            species_to_use, species_distribution = preprocessing.eliminate_underrepresented_species(species_list, 0.0)
            preprocessing.remove_unmatched_files(full_pathlist[6], full_pathlist[7])
        logging.info("Gathering point clouds...")
        unaugmented_regular_pointclouds = preprocessing.select_pointclouds(full_pathlist[6])
        unaugmented_fwf_pointclouds = preprocessing.select_pointclouds(full_pathlist[7])
        unaugmented_regular_pred_pointclouds = preprocessing.select_pointclouds(full_pathlist[10])
        unaugmented_fwf_pred_pointclouds = preprocessing.select_pointclouds(full_pathlist[11])
        logging.info("Augmenting point clouds...")
        preprocessing.augment_selection_fwf(unaugmented_regular_pointclouds, unaugmented_fwf_pointclouds, maxpcscale, full_pathlist[6], full_pathlist[7], species_distribution)
        preprocessing.augment_selection_fwf(unaugmented_regular_pred_pointclouds, unaugmented_fwf_pred_pointclouds, maxpcscale, full_pathlist[10], full_pathlist[11], species_distribution)
        logging.info("Resampling point clouds using FPS...")
        selected_pointclouds_augmented, selected_fwf_pointclouds_augmented, selected_images_augmented = preprocessing.get_user_specified_data_fwf(full_pathlist[6], full_pathlist[7], full_pathlist[8], capsel, growsel)
        selected_pointclouds_pred_augmented, selected_fwf_pointclouds_pred_augmented, selected_images_pred_augmented = preprocessing.get_user_specified_data_fwf(full_pathlist[10], full_pathlist[11], full_pathlist[12], capsel, growsel)
        resampled_pointclouds = preprocessing.resample_pointclouds_fps(selected_pointclouds_augmented, netpcsize)
        resampled_pointclouds_pred = preprocessing.resample_pointclouds_fps(selected_pointclouds_pred_augmented, netpcsize)
        logging.info("Generating metrics for point clouds...")
        combined_metrics_all, feature_names, eliminated_features = preprocessing.generate_metrics_for_selected_pointclouds_fwf(selected_pointclouds_augmented, selected_fwf_pointclouds_augmented, full_pathlist[9], capsel, growsel, [])
        combined_metrics_all_pred, feature_names_pred, elim_features = preprocessing.generate_metrics_for_selected_pointclouds_fwf(selected_pointclouds_pred_augmented, selected_fwf_pointclouds_pred_augmented, full_pathlist[13], capsel, growsel, eliminated_features)
        logging.info("Kept features: %s", feature_names)
        combined_metrics_all_cleaned, combined_metrics_all_pred_cleaned, dropped_cols = preprocessing.drop_nan_columns(combined_metrics_all, combined_metrics_all_pred)
        logging.info("Dropped columns indices: %s", dropped_cols)
        feature_names_cleaned = [name for i, name in enumerate(feature_names) if i not in dropped_cols]
        logging.info("New shape of combined_metrics_all: %s", combined_metrics_all_cleaned.shape)
        logging.info("New shape of combined_metrics_all_pred: %s", combined_metrics_all_pred_cleaned.shape)
        logging.info("Generating images...")
        max_height = preprocessing.get_maximum_unscaled_image_size(full_pathlist[6], full_pathlist[8])
        preprocessing.generate_colored_images(netimgsize, full_pathlist[6], full_pathlist[8], max_height)
        max_height_pred = preprocessing.get_maximum_unscaled_image_size(full_pathlist[10], full_pathlist[12])
        preprocessing.generate_colored_images(netimgsize, full_pathlist[10], full_pathlist[12], max_height_pred)
        selected_pointclouds_augmented, selected_fwf_pointclouds_augmented, selected_images_augmented = preprocessing.get_user_specified_data_fwf(full_pathlist[6], full_pathlist[7], full_pathlist[8], capsel, growsel)
        selected_pointclouds_pred_augmented, selected_fwf_pointclouds_pred_augmented, selected_images_pred_augmented = preprocessing.get_user_specified_data_fwf(full_pathlist[10], full_pathlist[11], full_pathlist[12], capsel, growsel)
        logging.info("Collecting image data...")
        images_frontal, images_sideways = preprocessing.match_images_with_pointclouds(selected_pointclouds_augmented, selected_images_augmented)
        images_frontal_pred, images_sideways_pred = preprocessing.match_images_with_pointclouds(selected_pointclouds_pred_augmented, selected_images_pred_augmented)
        images_front = np.asarray(images_frontal, dtype=np.float32)
        images_side = np.asarray(images_sideways, dtype=np.float32)
        images_front_pred = np.asarray(images_frontal_pred, dtype=np.float32)
        images_side_pred = np.asarray(images_sideways_pred, dtype=np.float32)
        print(images_front.shape, images_side.shape, images_front_pred.shape, images_side_pred.shape)
        logging.info("Creating final Training-, Validation- and Test-Set...")
        X_pc_train, X_pc_val, X_pc_pred, X_metrics_train, X_metrics_val, X_metrics_pred, X_img_1_train, X_img_1_val, X_img_1_pred, X_img_2_train, X_img_2_val, X_img_2_pred, y_train, y_val, y_pred, num_classes, label_dict = preprocessing.generate_training_data(capsel, growsel, selected_pointclouds_augmented, resampled_pointclouds, selected_pointclouds_pred_augmented, resampled_pointclouds_pred, combined_metrics_all_cleaned, combined_metrics_all_pred_cleaned, images_front, images_side, images_front_pred, images_side_pred, ssstest, full_pathlist[9], full_pathlist[13], 0.005, feature_names_cleaned)
        return X_pc_train, X_pc_val, X_pc_pred, X_metrics_train, X_metrics_val, X_metrics_pred, X_img_1_train, X_img_1_val, X_img_1_pred, X_img_2_train, X_img_2_val, X_img_2_pred, y_train, y_val, y_pred, num_classes, label_dict
    else:
        logging.info("Creating Test-Set and removing underrepresented species...")
        if workspace_setup.files_extracted(full_pathlist[7]) == 0:
            preprocessing.remove_insufficient_pointclouds(full_pathlist[4], netpcsize)
            species_distribution = preprocessing.eliminate_unused_species(full_pathlist[4], elimper, netpcsize)
            preprocessing.remove_height_outliers(full_pathlist[4])
            preprocessing.move_pointclouds_to_preds(full_pathlist[4], full_pathlist[7])
        else:
            pointclouds = []
            for file in os.listdir(full_pathlist[4]):
                pointclouds.append(file)
            species_list = preprocessing.get_species_distribution(pointclouds)
            species_to_use, species_distribution = preprocessing.eliminate_underrepresented_species(species_list, 0.0)
        logging.info("Gathering point clouds...")
        unaugmented_regular_pointclouds = preprocessing.select_pointclouds(full_pathlist[4])
        unaugmented_regular_pred_pointclouds = preprocessing.select_pointclouds(full_pathlist[7])
        logging.info("Augmenting point clouds...")
        preprocessing.augment_selection(unaugmented_regular_pointclouds, maxpcscale, full_pathlist[4], species_distribution)
        preprocessing.augment_selection(unaugmented_regular_pred_pointclouds, maxpcscale, full_pathlist[7], species_distribution)
        logging.info("Resampling point clouds using FPS...")
        selected_pointclouds_augmented, selected_images_augmented = preprocessing.get_user_specified_data(full_pathlist[4], full_pathlist[5], capsel, growsel)
        selected_pointclouds_pred_augmented, selected_images_pred_augmented = preprocessing.get_user_specified_data(full_pathlist[7], full_pathlist[8], capsel, growsel)
        resampled_pointclouds = preprocessing.resample_pointclouds_fps(selected_pointclouds_augmented, netpcsize)
        resampled_pointclouds_pred = preprocessing.resample_pointclouds_fps(selected_pointclouds_pred_augmented, netpcsize)
        logging.info("Generating metrics for point clouds...")
        combined_metrics_all, feature_names, eliminated_features = preprocessing.generate_metrics_for_selected_pointclouds(selected_pointclouds_augmented, full_pathlist[6], capsel, growsel, [])
        combined_metrics_all_pred, feature_names_pred, elim_features_pred = preprocessing.generate_metrics_for_selected_pointclouds(selected_pointclouds_pred_augmented, full_pathlist[9], capsel, growsel, eliminated_features)
        combined_metrics_all_cleaned, combined_metrics_all_pred_cleaned, dropped_cols = preprocessing.drop_nan_columns(combined_metrics_all, combined_metrics_all_pred)
        logging.info("Dropped columns indices: %s", dropped_cols)
        feature_names_cleaned = [name for i, name in enumerate(feature_names) if i not in dropped_cols]
        logging.info("New shape of combined_metrics_all: %s", combined_metrics_all_cleaned.shape)
        logging.info("New shape of combined_metrics_all_pred: %s", combined_metrics_all_pred_cleaned.shape)
        logging.info("Generating images...")
        max_height = preprocessing.get_maximum_unscaled_image_size(full_pathlist[4], full_pathlist[5])
        preprocessing.generate_colored_images(netimgsize, full_pathlist[4], full_pathlist[5], max_height)
        max_height_pred = preprocessing.get_maximum_unscaled_image_size(full_pathlist[7], full_pathlist[8])
        preprocessing.generate_colored_images(netimgsize, full_pathlist[7], full_pathlist[8], max_height_pred)
        selected_pointclouds_augmented, selected_images_augmented = preprocessing.get_user_specified_data(full_pathlist[4], full_pathlist[5], capsel, growsel)
        selected_pointclouds_pred_augmented, selected_images_pred_augmented = preprocessing.get_user_specified_data(full_pathlist[7], full_pathlist[8], capsel, growsel)
        logging.info("Collecting image data...")
        images_frontal, images_sideways = preprocessing.match_images_with_pointclouds(selected_pointclouds_augmented, selected_images_augmented)
        images_frontal_pred, images_sideways_pred = preprocessing.match_images_with_pointclouds(selected_pointclouds_pred_augmented, selected_images_pred_augmented)
        images_front = np.asarray(images_frontal, dtype=np.float32)
        images_side = np.asarray(images_sideways, dtype=np.float32)
        images_front_pred = np.asarray(images_frontal_pred, dtype=np.float32)
        images_side_pred = np.asarray(images_sideways_pred, dtype=np.float32)
        print(images_front.shape, images_side.shape, images_front_pred.shape, images_side_pred.shape)
        logging.info("Creating final Training-, Validation- and Test-Set...")
        X_pc_train, X_pc_val, X_pc_pred, X_metrics_train, X_metrics_val, X_metrics_pred, X_img_1_train, X_img_1_val, X_img_1_pred, X_img_2_train, X_img_2_val, X_img_2_pred, y_train, y_val, y_pred, num_classes, label_dict = preprocessing.generate_training_data(capsel, growsel, selected_pointclouds_augmented, resampled_pointclouds, selected_pointclouds_pred_augmented, resampled_pointclouds_pred, combined_metrics_all_cleaned, combined_metrics_all_pred_cleaned, images_front, images_side, images_front_pred, images_side_pred, ssstest, full_pathlist[6], full_pathlist[9], 0.005, feature_names_cleaned)
        return X_pc_train, X_pc_val, X_pc_pred, X_metrics_train, X_metrics_val, X_metrics_pred, X_img_1_train, X_img_1_val, X_img_1_pred, X_img_2_train, X_img_2_val, X_img_2_pred, y_train, y_val, y_pred, num_classes, label_dict
    
def perform_hp_tuning(model_dir, X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, bsize, netpcsize, netimgsize, num_classes, capsel, growsel, fwf_av):
    """
    Performs hyperparameter tuning for MMTSCNet using Bayesian Optimization.

    This function:
    - Configures GPU memory settings.
    - Prepares and normalizes training and validation data.
    - Checks for data integrity and label corruption.
    - Defines data generators and shuffles data.
    - Executes Bayesian Optimization to find the best hyperparameters.
    - Returns an untrained model with the best-found configuration.

    Args:
        model_dir (str): Directory to store tuning results.
        X_pc_train (ndarray): Training point cloud data.
        X_img_1_train (ndarray): Training frontal images.
        X_img_2_train (ndarray): Training sideways images.
        X_metrics_train (ndarray): Training numerical features.
        y_train (ndarray): Training labels.
        X_pc_val (ndarray): Validation point cloud data.
        X_img_1_val (ndarray): Validation frontal images.
        X_img_2_val (ndarray): Validation sideways images.
        X_metrics_val (ndarray): Validation numerical features.
        y_val (ndarray): Validation labels.
        bsize (int): Batch size for training.
        netpcsize (int): Number of points per point cloud.
        netimgsize (int): Image size in pixels.
        num_classes (int): Number of classification categories.
        capsel (str): Acquisition selection criteria.
        growsel (str): Leaf-condition selection criteria.
        fwf_av (bool): Indicates whether Full Waveform (FWF) data is available.

    Returns:
        tuple: 
            - untrained_model (tf.keras.Model): Model initialized with optimal hyperparameters.
            - optimal_learning_rate (float): Best learning rate found during tuning.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    point_cloud_shape = (netpcsize, 3)
    image_shape = (netimgsize, netimgsize, 3)
    metrics_shape = (X_metrics_train.shape[1],)
    batch_size = bsize
    num_hp_epochs = 8
    num_hp_trials = 5
    os.chdir(model_dir)
    tf.keras.backend.clear_session()
    X_img_1_train, X_img_2_train, X_pc_train, X_metrics_train = model_utils.normalize_data(X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train)
    X_img_1_val, X_img_2_val, X_pc_val, X_metrics_val = model_utils.normalize_data(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val)
    model_utils.check_data(X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train)
    model_utils.check_data(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val)
    corruption_found = model_utils.check_label_corruption(y_train)
    if not corruption_found:
        logging.info("No corruption found in one-hot encoded labels!")
    corruption_found = model_utils.check_label_corruption(y_val)
    if not corruption_found:
        logging.info("No corruption found in one-hot encoded labels!")
    train_gen = model_utils.DataGenerator(X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, batch_size)
    val_gen = model_utils.DataGenerator(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, batch_size)
    train_gen.on_epoch_end()
    val_gen.on_epoch_end()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if fwf_av == True:
        dir_name = f'hp-tuning-fwf_{capsel}_{growsel}_{netpcsize}_{num_classes}_{timestamp}'
    else:
        dir_name = f'hp-tuning_{capsel}_{growsel}_{netpcsize}_{num_classes}_{timestamp}'
    tuner = BayesianOptimization(
        model_utils.CombinedModel(point_cloud_shape, image_shape, metrics_shape, num_classes, netpcsize),
        objective=Objective("val_recall", direction="max"),
        max_trials=num_hp_trials,
        max_retries_per_trial=3,
        max_consecutive_failed_trials=3,
        directory=dir_name,
        project_name='tree_classification'
    )
    degrade_lr = LearningRateScheduler(model_utils.scheduler)
    logging.info(f"Commencing hyperparameter-tuning for {num_hp_trials} trials with {num_hp_epochs} epochs!")
    tuner.search(train_gen,
                epochs=num_hp_epochs,
                validation_data=val_gen,
                
                callbacks=[degrade_lr])
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    optimal_learning_rate = best_hyperparameters.get('learning_rate')
    logging.info(f"Optimal Learning Rate: {optimal_learning_rate}")
    combined_model = model_utils.CombinedModel(point_cloud_shape, image_shape, metrics_shape, num_classes, netpcsize)
    untrained_model = combined_model.get_untrained_model(best_hyperparameters)
    untrained_model.summary()
    gc.collect()
    keras.backend.clear_session()
    return untrained_model, optimal_learning_rate

def perform_training(model, bsz, X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, modeldir, label_dict, capsel, growsel, netpcsize, fwf_av, optimal_learning_rate):
    """
    Trains the MMTSCNet model with the provided data, applying early stopping and learning rate scheduling.

    This function:
    - Configures GPU memory settings.
    - Compiles the model with optimal hyperparameters.
    - Normalizes input data and validates integrity.
    - Defines data generators and shuffles datasets.
    - Trains the model using early stopping and learning rate scheduling.
    - Saves the trained model and plots training metrics.

    Args:
        model (tf.keras.Model): Untrained MMTSCNet model.
        bsz (int): Batch size for training.
        X_pc_train (ndarray): Training point cloud data.
        X_img_1_train (ndarray): Training frontal images.
        X_img_2_train (ndarray): Training sideways images.
        X_metrics_train (ndarray): Training numerical features.
        y_train (ndarray): Training labels.
        X_pc_val (ndarray): Validation point cloud data.
        X_img_1_val (ndarray): Validation frontal images.
        X_img_2_val (ndarray): Validation sideways images.
        X_metrics_val (ndarray): Validation numerical features.
        y_val (ndarray): Validation labels.
        modeldir (str): Directory to store model checkpoints and logs.
        label_dict (dict): Mapping of class indices to labels.
        capsel (str): Acquisition selection criteria.
        growsel (str): Leaf-condition selection criteria.
        netpcsize (int): Number of points per point cloud.
        fwf_av (bool): Indicates whether Full Waveform (FWF) data is available.
        optimal_learning_rate (float): Learning rate determined from hyperparameter tuning.

    Returns:
        tuple:
            - trained_model (tf.keras.Model): Fully trained model.
            - plot_path (str): Path to the training metrics plots.
    """
    tf.keras.utils.set_random_seed(812)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=optimal_learning_rate, clipnorm=0.5),
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy', tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="pr_curve", curve="PR"), tf.keras.metrics.PrecisionAtRecall(0.85, name="pr_at_rec"), tf.keras.metrics.RecallAtPrecision(0.85, name="rec_at_pr")]
    )
    model._name="MMTSCNet_V2"
    model.summary()
    os.chdir(modeldir)
    keras.backend.clear_session()
    X_img_1_train, X_img_2_train, X_pc_train, X_metrics_train = model_utils.normalize_data(X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train)
    X_img_1_val, X_img_2_val, X_pc_val, X_metrics_val = model_utils.normalize_data(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val)
    model_utils.check_data(X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train)
    model_utils.check_data(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val)
    corruption_found = model_utils.check_label_corruption(y_train)
    if not corruption_found:
        logging.info("No corruption found in one-hot encoded labels!")
    corruption_found = model_utils.check_label_corruption(y_val)
    if not corruption_found:
        logging.info("No corruption found in one-hot encoded labels!")
    train_gen = model_utils.DataGenerator(X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, bsz)
    val_gen = model_utils.DataGenerator(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, bsz)
    train_gen.on_epoch_end()
    val_gen.on_epoch_end()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)
    degrade_lr = tf.keras.callbacks.LearningRateScheduler(model_utils.scheduler)
    history = model.fit(
        train_gen,
        epochs=300,
        validation_data=val_gen,
        
        callbacks=[early_stopping, degrade_lr],
        verbose=1
    )
    plot_path = model_utils.plot_and_save_history(history, modeldir, capsel, growsel, netpcsize, fwf_av)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if fwf_av == True:
        model_file_path = f'trained-fwf_{capsel}_{growsel}_{str(netpcsize)}_{timestamp}'
    else:
        model_file_path = f'trained_{capsel}_{growsel}_{str(netpcsize)}_{timestamp}'
    if os.path.exists(model_file_path):
        os.remove(model_file_path)
    try:
        model.save(model_file_path, save_format="keras")
    except:
        pass
    model_utils.plot_best_epoch_metrics(history, plot_path)
    model.summary()
    model_path = model_utils.get_trained_model_folder(modeldir, capsel, growsel)
    trained_model = model_utils.load_trained_model_from_folder(model_path)
    keras.backend.clear_session()
    gc.collect()
    return trained_model, plot_path

def build_mmtscnet_with_optimal_hps(netpcsize, netimgsize, num_classes, cap_sel, grow_sel, fwf_av, X_metrics_train, modeldir):
    """
    Builds an untrained MMTSCNet model using the best-found hyperparameters.

    This function:
    - Defines input shapes based on network parameters.
    - Retrieves optimal hyperparameters and learning rate.
    - Initializes an untrained MMTSCNet model with the optimal configuration.

    Args:
        netpcsize (int): Number of points per point cloud.
        netimgsize (int): Image size in pixels.
        num_classes (int): Number of classification categories.
        cap_sel (str): Acquisition selection criteria.
        grow_sel (str): Leaf-condition selection criteria.
        fwf_av (bool): Indicates whether Full Waveform (FWF) data is available.
        X_metrics_train (ndarray): Training numerical feature data.
        modeldir (str): Directory where model configurations are stored.

    Returns:
        tuple:
            - untrained_model (tf.keras.Model): Model initialized with optimal hyperparameters.
            - optimal_lr (float): Best learning rate found for the configuration.
    """
    point_cloud_shape = (netpcsize, 3)
    image_shape = (netimgsize, netimgsize, 3)
    metrics_shape = (X_metrics_train.shape[1],)
    best_hyperparameters, optimal_lr = predef_mmtscnet.get_hyperparams_for_config(num_classes, cap_sel, grow_sel, fwf_av, point_cloud_shape, image_shape, metrics_shape, netpcsize, modeldir)
    combined_model = model_utils.CombinedModel(point_cloud_shape, image_shape, metrics_shape, num_classes, netpcsize)
    untrained_model = combined_model.get_untrained_model(best_hyperparameters)
    untrained_model.summary()
    return untrained_model, optimal_lr





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