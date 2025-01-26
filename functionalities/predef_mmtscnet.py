from keras_tuner import HyperParameters

def get_hyperparams_for_config(num_classes, cap_sel, grow_sel, fwf_av):
    best_hyperparameters = HyperParameters()
    if num_classes == 4:
        if cap_sel == "ALS":
            if fwf_av == True:
                # === Point Cloud Extractor ===
                best_hyperparameters.Fixed('pce_depth', 3)
                best_hyperparameters.Fixed('pce_units', 512)
                best_hyperparameters.Fixed('pce_dropout_rate', 0.05)
                best_hyperparameters.Fixed('pce_regularization', 0.0001)
                best_hyperparameters.Fixed('pce_neighbors', 32)
                best_hyperparameters.Fixed('pce_msg_radius_1', 0.02)
                best_hyperparameters.Fixed('pce_msg_radius_2', 0.3)
                best_hyperparameters.Fixed('pce_msg_radius_3', 0.7)
                # === T-Net for Point Cloud Transformation ===
                best_hyperparameters.Fixed('tnet_units', 128)
                best_hyperparameters.Fixed('tnet_regularization', 0.00001)
                best_hyperparameters.Fixed('tnet_dropout', 0.05)
                # === Metrics Model (MLP for tabular data) ===
                best_hyperparameters.Fixed('metrics_units', 768)
                best_hyperparameters.Fixed('metrics_dropout_rate', 0.1)
                best_hyperparameters.Fixed('metrics_regularization', 0.0001)
                # === Classification Head ===
                best_hyperparameters.Fixed('projection_units', 256)
                best_hyperparameters.Fixed('clss_depth', 3)
                best_hyperparameters.Fixed('clss_units', 480)
                best_hyperparameters.Fixed('clss_dropout_rate', 0.1)
                best_hyperparameters.Fixed('clss_regularization', 0.0001)
                # === Learning Rate ===
                best_hyperparameters.Fixed('learning_rate', 5e-5)
            else:
                # === Point Cloud Extractor ===
                best_hyperparameters.Fixed('pce_depth', 3)
                best_hyperparameters.Fixed('pce_units', 512)
                best_hyperparameters.Fixed('pce_dropout_rate', 0.05)
                best_hyperparameters.Fixed('pce_regularization', 0.0001)
                best_hyperparameters.Fixed('pce_neighbors', 32)
                best_hyperparameters.Fixed('pce_msg_radius_1', 0.02)
                best_hyperparameters.Fixed('pce_msg_radius_2', 0.3)
                best_hyperparameters.Fixed('pce_msg_radius_3', 0.7)
                # === T-Net for Point Cloud Transformation ===
                best_hyperparameters.Fixed('tnet_units', 128)
                best_hyperparameters.Fixed('tnet_regularization', 0.00001)
                best_hyperparameters.Fixed('tnet_dropout', 0.05)
                # === Metrics Model (MLP for tabular data) ===
                best_hyperparameters.Fixed('metrics_units', 768)
                best_hyperparameters.Fixed('metrics_dropout_rate', 0.1)
                best_hyperparameters.Fixed('metrics_regularization', 0.0001)
                # === Classification Head ===
                best_hyperparameters.Fixed('projection_units', 256)
                best_hyperparameters.Fixed('clss_depth', 3)
                best_hyperparameters.Fixed('clss_units', 480)
                best_hyperparameters.Fixed('clss_dropout_rate', 0.1)
                best_hyperparameters.Fixed('clss_regularization', 0.0001)
                # === Learning Rate ===
                best_hyperparameters.Fixed('learning_rate', 5e-5)
    else:
        if cap_sel == "ALS":
            if fwf_av == True:
                # === Point Cloud Extractor ===
                best_hyperparameters.Fixed('pce_depth', 3)
                best_hyperparameters.Fixed('pce_units', 512)
                best_hyperparameters.Fixed('pce_dropout_rate', 0.05)
                best_hyperparameters.Fixed('pce_regularization', 0.0001)
                best_hyperparameters.Fixed('pce_neighbors', 32)
                best_hyperparameters.Fixed('pce_msg_radius_1', 0.02)
                best_hyperparameters.Fixed('pce_msg_radius_2', 0.3)
                best_hyperparameters.Fixed('pce_msg_radius_3', 0.7)
                # === T-Net for Point Cloud Transformation ===
                best_hyperparameters.Fixed('tnet_units', 128)
                best_hyperparameters.Fixed('tnet_regularization', 0.00001)
                best_hyperparameters.Fixed('tnet_dropout', 0.05)
                # === Metrics Model (MLP for tabular data) ===
                best_hyperparameters.Fixed('metrics_units', 768)
                best_hyperparameters.Fixed('metrics_dropout_rate', 0.1)
                best_hyperparameters.Fixed('metrics_regularization', 0.0001)
                # === Classification Head ===
                best_hyperparameters.Fixed('projection_units', 256)
                best_hyperparameters.Fixed('clss_depth', 3)
                best_hyperparameters.Fixed('clss_units', 480)
                best_hyperparameters.Fixed('clss_dropout_rate', 0.1)
                best_hyperparameters.Fixed('clss_regularization', 0.0001)
                # === Learning Rate ===
                best_hyperparameters.Fixed('learning_rate', 5e-5)
            else:
                # === Point Cloud Extractor ===
                best_hyperparameters.Fixed('pce_depth', 3)
                best_hyperparameters.Fixed('pce_units', 512)
                best_hyperparameters.Fixed('pce_dropout_rate', 0.05)
                best_hyperparameters.Fixed('pce_regularization', 0.0001)
                best_hyperparameters.Fixed('pce_neighbors', 32)
                best_hyperparameters.Fixed('pce_msg_radius_1', 0.02)
                best_hyperparameters.Fixed('pce_msg_radius_2', 0.3)
                best_hyperparameters.Fixed('pce_msg_radius_3', 0.7)
                # === T-Net for Point Cloud Transformation ===
                best_hyperparameters.Fixed('tnet_units', 128)
                best_hyperparameters.Fixed('tnet_regularization', 0.00001)
                best_hyperparameters.Fixed('tnet_dropout', 0.05)
                # === Metrics Model (MLP for tabular data) ===
                best_hyperparameters.Fixed('metrics_units', 768)
                best_hyperparameters.Fixed('metrics_dropout_rate', 0.1)
                best_hyperparameters.Fixed('metrics_regularization', 0.0001)
                # === Classification Head ===
                best_hyperparameters.Fixed('projection_units', 256)
                best_hyperparameters.Fixed('clss_depth', 3)
                best_hyperparameters.Fixed('clss_units', 480)
                best_hyperparameters.Fixed('clss_dropout_rate', 0.1)
                best_hyperparameters.Fixed('clss_regularization', 0.0001)
                # === Learning Rate ===
                best_hyperparameters.Fixed('learning_rate', 5e-5)
        else:
            if grow_sel == "LEAF-ON":
                if fwf_av == True:
                    # === Point Cloud Extractor ===
                    best_hyperparameters.Fixed('pce_depth', 3)
                    best_hyperparameters.Fixed('pce_units', 512)
                    best_hyperparameters.Fixed('pce_dropout_rate', 0.05)
                    best_hyperparameters.Fixed('pce_regularization', 0.0001)
                    best_hyperparameters.Fixed('pce_neighbors', 32)
                    best_hyperparameters.Fixed('pce_msg_radius_1', 0.02)
                    best_hyperparameters.Fixed('pce_msg_radius_2', 0.3)
                    best_hyperparameters.Fixed('pce_msg_radius_3', 0.7)
                    # === T-Net for Point Cloud Transformation ===
                    best_hyperparameters.Fixed('tnet_units', 128)
                    best_hyperparameters.Fixed('tnet_regularization', 0.00001)
                    best_hyperparameters.Fixed('tnet_dropout', 0.05)
                    # === Metrics Model (MLP for tabular data) ===
                    best_hyperparameters.Fixed('metrics_units', 768)
                    best_hyperparameters.Fixed('metrics_dropout_rate', 0.1)
                    best_hyperparameters.Fixed('metrics_regularization', 0.0001)
                    # === Classification Head ===
                    best_hyperparameters.Fixed('projection_units', 256)
                    best_hyperparameters.Fixed('clss_depth', 3)
                    best_hyperparameters.Fixed('clss_units', 480)
                    best_hyperparameters.Fixed('clss_dropout_rate', 0.1)
                    best_hyperparameters.Fixed('clss_regularization', 0.0001)
                    # === Learning Rate ===
                    best_hyperparameters.Fixed('learning_rate', 5e-5)
                else:
                    # === Point Cloud Extractor ===
                    best_hyperparameters.Fixed('pce_depth', 3)
                    best_hyperparameters.Fixed('pce_units', 512)
                    best_hyperparameters.Fixed('pce_dropout_rate', 0.05)
                    best_hyperparameters.Fixed('pce_regularization', 0.0001)
                    best_hyperparameters.Fixed('pce_neighbors', 32)
                    best_hyperparameters.Fixed('pce_msg_radius_1', 0.02)
                    best_hyperparameters.Fixed('pce_msg_radius_2', 0.3)
                    best_hyperparameters.Fixed('pce_msg_radius_3', 0.7)
                    # === T-Net for Point Cloud Transformation ===
                    best_hyperparameters.Fixed('tnet_units', 128)
                    best_hyperparameters.Fixed('tnet_regularization', 0.00001)
                    best_hyperparameters.Fixed('tnet_dropout', 0.05)
                    # === Metrics Model (MLP for tabular data) ===
                    best_hyperparameters.Fixed('metrics_units', 768)
                    best_hyperparameters.Fixed('metrics_dropout_rate', 0.1)
                    best_hyperparameters.Fixed('metrics_regularization', 0.0001)
                    # === Classification Head ===
                    best_hyperparameters.Fixed('projection_units', 256)
                    best_hyperparameters.Fixed('clss_depth', 3)
                    best_hyperparameters.Fixed('clss_units', 480)
                    best_hyperparameters.Fixed('clss_dropout_rate', 0.1)
                    best_hyperparameters.Fixed('clss_regularization', 0.0001)
                    # === Learning Rate ===
                    best_hyperparameters.Fixed('learning_rate', 5e-5)
            else:
                if fwf_av == True:
                    # === Point Cloud Extractor ===
                    best_hyperparameters.Fixed('pce_depth', 3)
                    best_hyperparameters.Fixed('pce_units', 512)
                    best_hyperparameters.Fixed('pce_dropout_rate', 0.05)
                    best_hyperparameters.Fixed('pce_regularization', 0.0001)
                    best_hyperparameters.Fixed('pce_neighbors', 32)
                    best_hyperparameters.Fixed('pce_msg_radius_1', 0.02)
                    best_hyperparameters.Fixed('pce_msg_radius_2', 0.3)
                    best_hyperparameters.Fixed('pce_msg_radius_3', 0.7)
                    # === T-Net for Point Cloud Transformation ===
                    best_hyperparameters.Fixed('tnet_units', 128)
                    best_hyperparameters.Fixed('tnet_regularization', 0.00001)
                    best_hyperparameters.Fixed('tnet_dropout', 0.05)
                    # === Metrics Model (MLP for tabular data) ===
                    best_hyperparameters.Fixed('metrics_units', 768)
                    best_hyperparameters.Fixed('metrics_dropout_rate', 0.1)
                    best_hyperparameters.Fixed('metrics_regularization', 0.0001)
                    # === Classification Head ===
                    best_hyperparameters.Fixed('projection_units', 256)
                    best_hyperparameters.Fixed('clss_depth', 3)
                    best_hyperparameters.Fixed('clss_units', 480)
                    best_hyperparameters.Fixed('clss_dropout_rate', 0.1)
                    best_hyperparameters.Fixed('clss_regularization', 0.0001)
                    # === Learning Rate ===
                    best_hyperparameters.Fixed('learning_rate', 5e-5)
                else:
                    # === Point Cloud Extractor ===
                    best_hyperparameters.Fixed('pce_depth', 3)
                    best_hyperparameters.Fixed('pce_units', 512)
                    best_hyperparameters.Fixed('pce_dropout_rate', 0.05)
                    best_hyperparameters.Fixed('pce_regularization', 0.0001)
                    best_hyperparameters.Fixed('pce_neighbors', 32)
                    best_hyperparameters.Fixed('pce_msg_radius_1', 0.02)
                    best_hyperparameters.Fixed('pce_msg_radius_2', 0.3)
                    best_hyperparameters.Fixed('pce_msg_radius_3', 0.7)
                    # === T-Net for Point Cloud Transformation ===
                    best_hyperparameters.Fixed('tnet_units', 128)
                    best_hyperparameters.Fixed('tnet_regularization', 0.00001)
                    best_hyperparameters.Fixed('tnet_dropout', 0.05)
                    # === Metrics Model (MLP for tabular data) ===
                    best_hyperparameters.Fixed('metrics_units', 768)
                    best_hyperparameters.Fixed('metrics_dropout_rate', 0.1)
                    best_hyperparameters.Fixed('metrics_regularization', 0.0001)
                    # === Classification Head ===
                    best_hyperparameters.Fixed('projection_units', 256)
                    best_hyperparameters.Fixed('clss_depth', 3)
                    best_hyperparameters.Fixed('clss_units', 480)
                    best_hyperparameters.Fixed('clss_dropout_rate', 0.1)
                    best_hyperparameters.Fixed('clss_regularization', 0.0001)
                    # === Learning Rate ===
                    best_hyperparameters.Fixed('learning_rate', 5e-5)
    return best_hyperparameters