import tensorflow as tf
import logging
import sys
import time
from utils import main_utils
from functionalities import main_functions, model_utils
import multiprocessing as mp

def run_mmtscnet():
    """
    Main function used to preprocess data, tune and train MMTSCNet.

    """
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus)>0:
        args = main_utils.parse_arguments()
        log_level = logging.DEBUG if args.verbose else logging.INFO
        main_utils.setup_logging(log_level)
        logging.info("Starting MMTSCNET - This may take a while!")
        data_dir, work_dir, model_dir, elim_per, max_pcscale, sss_test, cap_sel, grow_sel, bsize, img_size, pc_size = main_utils.validate_inputs(args.datadir, args.workdir, args.modeldir, args.elimper, args.maxpcscale, args.ssstest, args.capsel, args.growsel, args.batchsize, args.numpoints)
        fwf_av = main_utils.are_fwf_pointclouds_available(data_dir)
        try:
            logging.info("Creating Working environment...")
            workspace_paths = main_functions.extract_data(data_dir, work_dir, fwf_av, cap_sel, grow_sel)
            logging.info("Preprocessing data...")
            X_pc_train, X_pc_val, X_pc_pred, X_metrics_train, X_metrics_val, X_metrics_pred, X_img_1_train, X_img_1_val, X_img_1_pred, X_img_2_train, X_img_2_val, X_img_2_pred, y_train, y_val, y_pred, num_classes, label_dict = main_functions.preprocess_data(workspace_paths, sss_test, cap_sel, grow_sel, elim_per, max_pcscale, pc_size, img_size, fwf_av)
            if args.inference and grow_sel != "ALL" and cap_sel not in ["ALL", "TLS"]:
                untrained_model, optimal_learning_rate = main_functions.build_mmtscnet_with_optimal_hps(pc_size, img_size, num_classes, cap_sel, grow_sel, fwf_av, X_metrics_train)
            else:
                logging.info("Commencing hyperparameter-tuning...")
                untrained_model, optimal_learning_rate = main_functions.perform_hp_tuning(model_dir, X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, bsize, pc_size, img_size, num_classes, cap_sel, grow_sel, fwf_av)
            logging.info("Training MMTSCNet...")
            trained_model, plot_path = main_functions.perform_training(untrained_model, bsize, X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, model_dir, label_dict, cap_sel, grow_sel, pc_size, fwf_av, optimal_learning_rate)
            logging.info("Training finished, predicting...")
            model_utils.predict_for_data(trained_model, X_pc_val, X_metrics_val, X_img_1_val, X_img_2_val, y_val ,X_pc_pred, X_metrics_pred, X_img_1_pred, X_img_2_pred, y_pred, label_dict, model_dir, cap_sel, grow_sel, pc_size, plot_path)
        except Exception as e:
            logging.exception("An error occurred: %s", e)
            time.sleep(3)
            sys.exit(1)
    else:
        logging.exception("No NVIDIA GPU detected! Can't run MMTSCNet!")
        time.sleep(3)
        sys.exit(1)

if __name__ == "__main__":
    """
    Runs the program.

    """
    mp.set_start_method('spawn')
    run_mmtscnet()