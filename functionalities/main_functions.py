from functionalities import workspace_setup, preprocessing, model_utils
import numpy as np
import os
import tensorflow as tf
import logging
import gc
import time
from keras_tuner import BayesianOptimization, Objective
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

def extract_data(data_dir, work_dir, fwf_av, capsel, growsel):
    """
    Main utility function for i/o ops before preprocessing.

    Args:
    data_dir: User-specified source data directory.
    work_dir: User-specified working directory.
    fwf_av: True/False - Presence of FWF data.
    capsel: User-specified acquisition selection.
    growsel: User-specified leaf-confition selection.

    Return:
    full_pathlist: List of all paths available to MMTSCNet.
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
    Main utility function for the preprocessing.

    Args:
    full_pathlist: List of all paths used for MMTSCNet.
    ssstest: Train/Test split ratio.
    capsel: User-specified acquisition selection.
    growsel: User-specified leaf-confition selection.
    elimper: Threshold for the elimination of underrepresented species.
    maxpcscale: Maximum scaling to apply during augmentation.
    netpcsize: Number of points to resample point clouds to.
    netimgsize: Image size in Pixels (224).
    fwf_av: True/False - Presence of FWF data.

    Returns:
    X_pc_train: Training point clouds.
    X_pc_val: Validation point clouds.
    X_metrics_train: Training numerical features.
    X_metrics_val: Validation numerical features.
    X_img_1_train: Frontal validation images.
    X_img_1_val: Frontal validation images.
    X_img_2_train: Sideways training images.
    X_img_2_val: Sideways validation images.
    y_train: Training labels.
    y_val: Validation labels.
    num_classes: Number of classes present in the training dataset.
    label_dict: Dictionary to translate one-hot encoded labels to textual labels.
    """
    if fwf_av == True:
        unaugmented_regular_pointclouds = preprocessing.select_pointclouds(full_pathlist[6])
        unaugmented_fwf_pointclouds = preprocessing.select_pointclouds(full_pathlist[7])

        logging.info(f"{full_pathlist[6], full_pathlist[7], len(unaugmented_fwf_pointclouds), len(unaugmented_regular_pointclouds)}")
        preprocessing.augment_selection_fwf(unaugmented_regular_pointclouds, unaugmented_fwf_pointclouds, elimper, maxpcscale, full_pathlist[6], full_pathlist[7], netpcsize, capsel)
        preprocessing.generate_colored_images(netimgsize, full_pathlist[6], full_pathlist[8])
        selected_pointclouds_augmented, selected_fwf_pointclouds_augmented, selected_images_augmented = preprocessing.get_user_specified_data_fwf(full_pathlist[6], full_pathlist[7], full_pathlist[8], capsel, growsel)
        filtered_pointclouds, filtered_fwf_pointclouds, filtered_images = preprocessing.filter_data_for_selection_fwf(selected_pointclouds_augmented, selected_fwf_pointclouds_augmented, selected_images_augmented, elimper)
        logging.info("Resampling point clouds...")
        pointclouds_for_resampling = [preprocessing.load_point_cloud(file) for file in filtered_pointclouds]
        centered_points = preprocessing.center_point_cloud(pointclouds_for_resampling)
        resampled_pointclouds = np.array([preprocessing.resample_pointcloud(centered_points, netpcsize, i) for i in range(len(centered_points))])
        logging.info("Generating metrics for point clouds...")
        combined_metrics_all, feature_names = preprocessing.generate_metrics_for_selected_pointclouds_fwf(filtered_pointclouds, filtered_fwf_pointclouds, full_pathlist[9], capsel, growsel)
        images_frontal, images_sideways = preprocessing.match_images_with_pointclouds(filtered_pointclouds, filtered_images)
        images_front = np.asarray(images_frontal)
        images_side = np.asarray(images_sideways)
        X_pc_train, X_pc_val, X_pc_pred, X_metrics_train, X_metrics_val, X_metrics_pred, X_img_1_train, X_img_1_val, X_img_1_pred, X_img_2_train, X_img_2_val, X_img_2_pred, y_train, y_val, y_pred, num_classes, label_dict = preprocessing.generate_training_data(capsel, growsel, filtered_pointclouds, resampled_pointclouds, combined_metrics_all, images_front, images_side, ssstest, full_pathlist[9], 0.008, feature_names)
        return X_pc_train, X_pc_val, X_pc_pred, X_metrics_train, X_metrics_val, X_metrics_pred, X_img_1_train, X_img_1_val, X_img_1_pred, X_img_2_train, X_img_2_val, X_img_2_pred, y_train, y_val, y_pred, num_classes, label_dict
    else:
        unaugmented_regular_pointclouds = preprocessing.select_pointclouds(full_pathlist[4])
        preprocessing.augment_selection(unaugmented_regular_pointclouds, elimper, maxpcscale, full_pathlist[4], netpcsize, capsel)
        preprocessing.generate_colored_images(netimgsize, full_pathlist[4], full_pathlist[5])
        selected_pointclouds_augmented, selected_images_augmented = preprocessing.get_user_specified_data(full_pathlist[4], full_pathlist[5], capsel, growsel)
        filtered_pointclouds, filtered_images = preprocessing.filter_data_for_selection(selected_pointclouds_augmented, selected_images_augmented, elimper)
        logging.info("Resampling point clouds...")
        pointclouds_for_resampling = [preprocessing.load_point_cloud(file) for file in filtered_pointclouds]
        centered_points = preprocessing.center_point_cloud(pointclouds_for_resampling)
        resampled_pointclouds = np.array([preprocessing.resample_pointcloud(centered_points, netpcsize, i) for i in range(len(centered_points))])
        logging.info("Generating metrics for point clouds...")
        combined_metrics_all, feature_names = preprocessing.generate_metrics_for_selected_pointclouds(filtered_pointclouds, full_pathlist[6], capsel, growsel)
        images_frontal, images_sideways = preprocessing.match_images_with_pointclouds(filtered_pointclouds, filtered_images)
        images_front = np.asarray(images_frontal)
        images_side = np.asarray(images_sideways)
        X_pc_train, X_pc_val, X_pc_pred, X_metrics_train, X_metrics_val, X_metrics_pred, X_img_1_train, X_img_1_val, X_img_1_pred, X_img_2_train, X_img_2_val, X_img_2_pred, y_train, y_val, y_pred, num_classes, label_dict = preprocessing.generate_training_data(capsel, growsel, filtered_pointclouds, resampled_pointclouds, combined_metrics_all, images_front, images_side, ssstest, full_pathlist[6], 0.008, feature_names)
        return X_pc_train, X_pc_val, X_pc_pred, X_metrics_train, X_metrics_val, X_metrics_pred, X_img_1_train, X_img_1_val, X_img_1_pred, X_img_2_train, X_img_2_val, X_img_2_pred, y_train, y_val, y_pred, num_classes, label_dict
    
def perform_hp_tuning(model_dir, X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, bsize, netpcsize, netimgsize, num_classes, capsel, growsel, fwf_av):
    """
    Main utility function for the hyperparameter tuning process.

    Args:
    model_dir: Filepath to model saving destination.
    X_pc_train: Training point clouds.
    X_pc_val: Validation point clouds.
    X_metrics_train: Training numerical features.
    X_metrics_val: Validation numerical features.
    X_img_1_train: Frontal validation images.
    X_img_1_val: Frontal validation images.
    X_img_2_train: Sideways training images.
    X_img_2_val: Sideways validation images.
    y_train: Training labels.
    y_val: Validation labels.
    bsize: User-specified batch size.
    netpcsize: Number of points to resample point clouds to.
    netimgsize: Image size in Pixels (224).
    num_classes: Number of classes present in the training dataset.
    capsel: User-specified acquisition selection.
    growsel: User-specified leaf-confition selection.
    fwf_av: True/False - Presence of FWF data.

    Returns:
    untrained_model: Keras model instance of MMTSCNet with tuned hyperparameters.
    """
    # Initial definition of variables
    point_cloud_shape = (netpcsize, 3)
    image_shape = (netimgsize, netimgsize, 3)
    metrics_shape = (X_metrics_train.shape[1],)
    batch_size = bsize
    num_hp_epochs = 7
    num_hp_trials = 10
    os.chdir(model_dir)
    # Clear the backend to free up memory
    tf.keras.backend.clear_session()
    # Normalize the data and check for corrupted entries
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
    logging.info(f"Distribution in training data: {model_utils.get_class_distribution(y_train)}")
    logging.info(f"Distribution in testing data: {model_utils.get_class_distribution(y_val)}")
    # Define data generators to feed data during hyperparameter tuning
    train_gen = model_utils.DataGenerator(X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, batch_size)
    val_gen = model_utils.DataGenerator(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, batch_size)
    # Shuffling of the dataset
    train_gen.on_epoch_end()
    val_gen.on_epoch_end()
    # Define name for hyperparameter saving folder
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if fwf_av == True:
        dir_name = f'hp-tuning-fwf_{capsel}_{growsel}_{netpcsize}_{timestamp}'
    else:
        dir_name = f'hp-tuning_{capsel}_{growsel}_{netpcsize}_{timestamp}'
    # Definition of instance of the BayesianOptimization Keras tuner
    tuner = BayesianOptimization(
        model_utils.CombinedModel(point_cloud_shape, image_shape, metrics_shape, num_classes, netpcsize),
        # Validation custom metric as objective
        objective=Objective("val_custom_metric", direction="max"),
        max_trials=num_hp_trials,
        max_retries_per_trial=3,
        max_consecutive_failed_trials=3,
        directory=dir_name,
        project_name='tree_classification'
    )
    # Definition of learning rate schedule and Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.995, patience=5, min_lr=5e-7)
    degrade_lr = LearningRateScheduler(model_utils.scheduler)
    macro_f1_callback = model_utils.MacroF1ScoreCallback(validation_data=val_gen, batch_size=batch_size)
    custom_scoring_callback = model_utils.WeightedResultsCallback(validation_data=val_gen, batch_size=batch_size)
    logging.info(f"Commencing hyperparameter-tuning for {num_hp_trials} trials with {num_hp_epochs} epochs!")
    # Tuning for the data with class-weights
    tuner.search(train_gen,
                epochs=num_hp_epochs,
                validation_data=val_gen,
                class_weight=model_utils.generate_class_weights(y_train),
                callbacks=[reduce_lr, degrade_lr, macro_f1_callback, custom_scoring_callback])
    # Retrieve best hyperparameter configuration of the tuning process
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    optimal_learning_rate = best_hyperparameters.get('learning_rate')
    print(f"Optimal Learning Rate: {optimal_learning_rate}")
    # Create instance of MMTSCNet with optimal hyperparameters
    combined_model = model_utils.CombinedModel(point_cloud_shape, image_shape, metrics_shape, num_classes, netpcsize)
    untrained_model = combined_model.get_untrained_model(best_hyperparameters)
    untrained_model.summary()
    gc.collect()
    return untrained_model, optimal_learning_rate

def perform_training(model, bsz, X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, modeldir, label_dict, capsel, growsel, netpcsize, fwf_av, optimal_learning_rate):
    """
    Main utility function for the training process.

    Args:
    model: Untrained tuned model instance.
    bsz: User-specified batch size.
    X_pc_train: Training point clouds.
    X_pc_val: Validation point clouds.
    X_metrics_train: Training numerical features.
    X_metrics_val: Validation numerical features.
    X_img_1_train: Frontal validation images.
    X_img_1_val: Frontal validation images.
    X_img_2_train: Sideways training images.
    X_img_2_val: Sideways validation images.
    y_train: Training labels.
    y_val: Validation labels.
    model_dir: Target directory for model saving.
    label_dict: Dictionary to translate one-hot encoded labels to textual labels.
    netpcsize: Number of points to resample point clouds to.
    capsel: User-specified acquisition selection.
    growsel: User-specified leaf-confition selection.
    fwf_av: True/False - Presence of FWF data.

    Returns:
    trained_model: Keras model instance of the trained MMTSCNet.
    """
    y_pred_val = y_val
    tf.keras.utils.set_random_seed(812)
    # Compilation of the tuned model with learning rate and training matrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=optimal_learning_rate, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="pr_curve", curve="PR"), tf.keras.metrics.PrecisionAtRecall(0.85, name="pr_at_rec"), tf.keras.metrics.RecallAtPrecision(0.85, name="rec_at_pr")]
    )
    # Print model summary
    model._name="MMTSCNet_V2"
    model.summary()
    os.chdir(modeldir)
    # Cear backend to free up memory
    tf.keras.backend.clear_session()
    # Data normalization and corruption check
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
    # Definition of data generators for training
    train_gen = model_utils.DataGenerator(X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, bsz)
    val_gen = model_utils.DataGenerator(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, bsz)
    # Shuffling of the dataset
    train_gen.on_epoch_end()
    val_gen.on_epoch_end()
    # Callback definition
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.95, patience=3, min_lr=1e-6)
    degrade_lr = tf.keras.callbacks.LearningRateScheduler(model_utils.scheduler)
    macro_f1_callback = model_utils.MacroF1ScoreCallback(validation_data=val_gen, batch_size=bsz)
    custom_scoring_callback = model_utils.WeightedResultsCallback(validation_data=val_gen, batch_size=bsz)
    logging.info(f"Distribution in training data: {model_utils.get_class_distribution(y_train)}")
    logging.info(f"Distribution in testing data: {model_utils.get_class_distribution(y_val)}")
    # Model training setup with 300 epochs and early stopping
    history = model.fit(
        train_gen,
        epochs=150,
        validation_data=val_gen,
        class_weight=model_utils.generate_class_weights(y_train),
        callbacks=[early_stopping, reduce_lr, degrade_lr, macro_f1_callback, custom_scoring_callback],
        verbose=1
    )
    # Plot training metrics as graphs
    plot_path = model_utils.plot_and_save_history(history, modeldir, capsel, growsel, netpcsize, fwf_av)
    # Creation of save folder for model and data
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
    # Prediction on validation data
    predictions = model.predict([X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val], batch_size=8, verbose=1)
    # Translation of labels
    y_pred_real = model_utils.map_onehot_to_real(predictions, label_dict)
    y_true_real = model_utils.map_onehot_to_real(y_pred_val, label_dict)
    # Plotting of confusion matrix and training metrics
    model_utils.plot_conf_matrix(y_true_real, y_pred_real, modeldir, plot_path, label_dict, capsel, growsel, netpcsize)
    model_utils.plot_best_epoch_metrics(history, plot_path)
    model.summary()
    # I/O ops
    model_path = model_utils.get_trained_model_folder(modeldir, capsel, growsel)
    trained_model = model_utils.load_trained_model_from_folder(model_path)
    gc.collect()
    return trained_model