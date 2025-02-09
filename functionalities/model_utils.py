import tensorflow as tf
from keras.layers import Input, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate, Reshape, ReLU, Add, Activation, LayerNormalization, Attention, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Multiply, Lambda
from keras.models import Model
from keras.regularizers import L1L2
from keras.applications import DenseNet121, EfficientNetV2S
from keras.callbacks import Callback
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from keras.optimizers import Adam
from keras.utils import Sequence
from keras_tuner import HyperModel, HyperParameters
import os
import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import logging
import pandas as pd
from keras.backend import sigmoid
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers import Layer
from keras.engine.input_spec import InputSpec
from keras import backend as K
from keras.losses import CategoricalCrossentropy

# Logging setup for tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Setup for mixed precision processing
from keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

class HybridPooling(Layer):
    """
    Combines Global Average Pooling and Global Max Pooling to preserve both
    dominant and fine-grained features.
    """
    def call(self, inputs):
        avg_pooled = GlobalAveragePooling1D()(inputs)
        max_pooled = GlobalMaxPooling1D()(inputs)
        return tf.concat([avg_pooled, max_pooled], axis=-1)

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

class GroupNormalization(Layer):
    """Group normalization layer

    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization. GN's computation is independent of batch sizes,
    and its accuracy is stable in a wide range of batch sizes

    # Arguments
        groups: Integer, the number of groups for Group Normalization.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
        - [GitHub Repository](https://github.com/titu1994/Keras-Group-Normalization/blob/master/group_norm.py)
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        tensor_input_shape = K.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = K.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = K.stack(group_shape)
        inputs = K.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        group_reduction_axes = group_reduction_axes[2:]

        mean = K.mean(inputs, axis=group_reduction_axes, keepdims=True)
        variance = K.var(inputs, axis=group_reduction_axes, keepdims=True)

        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = K.reshape(inputs, group_shape)
        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        outputs = K.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

keras.utils.generic_utils.get_custom_objects().update({'swish': Activation(swish)})
keras.utils.generic_utils.get_custom_objects().update({'GroupNormalization': GroupNormalization})
keras.utils.generic_utils.get_custom_objects().update({'HybridPooling': HybridPooling})

def scheduler(epoch, lr):
    """
    Creation of the learning rate scheduler.

    Args:
    epoch: Current epoch number.
    lr: Current learning rate.

    Returns:
    lr: Learning rate to be applied during the next epoch.
    """
    if epoch <= 5:
        return lr * (1.1 if lr < 1e-4 else 1.002)
    elif epoch > 5 and lr >= 1e-7:
        return lr * 0.98
    else:
        return lr
        
def check_if_model_is_created(modeldir):
    """
    Checks if a trained model has been saved at the specified location.

    Args:
    modeldir: Filepath to search in.

    Returns:
    True/False
    """
    files_list =  []
    for file in os.listdir(modeldir):
        if "trained" in file:
            files_list.append(file)
        else:
            pass
    if len(files_list)>0:
        return True
    else:
        return False
    
def check_if_tuned_model_is_created(modeldir):
    """
    Checks if a tuned model has been saved at the specified location.

    Args:
    modeldir: Filepath to search in.

    Returns:
    True/False
    """
    files_list =  []
    for file in os.listdir(modeldir):
        if "tuning" in file:
            files_list.append(file)
        else:
            pass
    if len(files_list)>0:
        return True
    else:
        return False

def get_tuned_model_folder(modeldir, capsel, growsel):
    """
    Retrieves the filepath for the most recently created model instance.

    Args:
    modeldir: Filepath to search in.
    capsel: User-specified acquisition selection.
    growsel: User-specified leaf-condition.

    Returns:
    most_recent_path: Path to the most recently created model.
    """
    most_recent_file = None
    most_recent_time = None
    for file in os.listdir(modeldir):
        if file.lower().endswith(".tf") or file.lower().endswith(".keras") or file.lower().endswith(".h5"):
            pass
        elif "trained" in file:
            pass
        elif capsel in file and growsel in file:
            date = file.split("_")[4]
            filetime = datetime.datetime.strptime(date, "%Y%m%d-%H%M%S")
            if most_recent_time is None or filetime > most_recent_time:
                most_recent_file = file
                most_recent_time = filetime
    most_recent_path = os.path.join(modeldir + "/" + most_recent_file)
    return most_recent_path

def get_trained_model_folder(modeldir, capsel, growsel):
    """
    Retrieves the filepath for the most recently created model instance.

    Args:
    modeldir: Filepath to search in.
    capsel: User-specified acquisition selection.
    growsel: User-specified leaf-condition.

    Returns:
    most_recent_path: Path to the most recently created model.
    """
    most_recent_file = None
    most_recent_time = None
    for file in os.listdir(modeldir):
        if file.lower().endswith(".tf") or file.lower().endswith(".keras") or file.lower().endswith(".h5"):
            pass
        elif "trained" in file and capsel in file and growsel in file:
            date = file.split("_")[4].split(".")[0]
            filetime = datetime.datetime.strptime(date, "%Y%m%d-%H%M%S")
            if most_recent_time is None or filetime > most_recent_time:
                most_recent_file = file
                most_recent_time = filetime
        else:
            pass
    most_recent_path = os.path.join(modeldir + "/" + most_recent_file)
    return most_recent_path

def load_trained_model_from_folder(model_path):
    """
    Loads a trained Keras model from a specified path.

    Args:
    model_path: Filepath fo the trained model instance.

    Returns:
    model: Trained model instance.
    """
    model = keras.models.load_model(model_path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=7.5e-6),
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy', tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="pr_curve", curve="PR"), tf.keras.metrics.PrecisionAtRecall(0.85, name="pr_at_rec"), tf.keras.metrics.RecallAtPrecision(0.85, name="rec_at_pr")]
    )
    return model

def load_tuned_model_from_folder(model_path):
    """
    Loads a tuned Keras model from a specified path.

    Args:
    model_path: Filepath fo the tuned model instance.

    Returns:
    model: Tuned model instance.
    """
    custom_objects = {
        'HyperParameters': HyperParameters,
        'PointCloudExtractor': PointCloudExtractor,
        'EnhancedMetricsModel': EnhancedMetricsModel,
        'CNNModel': EfficientNetModel,
        'CombinedModel': CombinedModel
    }
    model = keras.models.load_model(model_path, custom_objects)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=7.5e-6),
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy', tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="pr_curve", curve="PR"), tf.keras.metrics.PrecisionAtRecall(0.85, name="pr_at_rec"), tf.keras.metrics.RecallAtPrecision(0.85, name="rec_at_pr")]
    )
    return model

def find_most_recent_file(directory, capsel, growsel):
    """
    Checks a folder for the most recently created file based on its timestamp.

    Args:
    directory: Directory to check the contained files for.
    capsel: User-specified acquisition selection.
    growsel: User-specified leaf-condition.

    Returns:
    filepath: Filepath of the most recently created file.
    """
    files = []
    for file in os.listdir(directory):
        if ".laz" in file and "on" in file:
            files.append(file)
        elif ".laz" in file and "off" in file:
            files.append(file)
    filepath = os.path.join(directory, files[0])
    return filepath
    
def create_label_mapping(onehot_to_label_dict):
    """
    Maps one-hot and textual labels.

    Args:
    onehot_to_label_dict: Dictionary to translate one-hot encoded labels to textual labels.

    Returns:
    label_to_int: Dictionary to translate textual labels to one-hot.
    int_to_label: Dictionary to translate one-hot to textual labels.
    """
    label_to_int = {label: idx for idx, label in enumerate(onehot_to_label_dict.values())}
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    return label_to_int, int_to_label

def predict_for_data(trained_model, X_pc_val, X_metrics_val, X_img_1_val, X_img_2_val, y_val, X_pc_pred, X_metrics_pred, X_img_1_pred, X_img_2_pred, y_pred, label_dict, modeldir, capsel, growsel, netpcsize, plot_path):
    """
    Predicts for data with a trained instance of MMTSCNet.

    Args:
    pretrained_model: A pretrained instance of MMTSCNet.
    X_pc: Point clouds.
    X_metrics: Numerical features.
    X_img_1: Frontal images.
    X_img_2: Sideways images.
    onehot_to_label_dict: Dictionary to translate one-hot encoded labels to textual labels.
    filtered_pointclouds: Array of point cloud file paths.
    las_unzipped_path: Filepath of unzipped las point clouds.
    model_dir: Savepath for trained models.
    capsel: User-specified acquisition method.
    growsel: User-specified leaf-condition.
    netpcsize: Resampling target number of points.

    Returns:
    (Saved): A classified plot point cloud.
    """
    K.clear_session()

    X_img_1_pred, X_img_2_pred, X_pc_pred, X_metrics_pred = normalize_data(
        X_pc_pred, X_img_1_pred, X_img_2_pred, X_metrics_pred
    )
    check_data(X_pc_pred, X_img_1_pred, X_img_2_pred, X_metrics_pred, y_pred)
    pred_results = np.zeros_like(y_pred)
    for _ in range(10):
        predictions = trained_model.predict(
            [X_pc_pred, X_img_1_pred, X_img_2_pred, X_metrics_pred], batch_size=8, verbose=1
        )
        pred_results += predictions
    pred_results /= 10
    y_pred_real = map_onehot_to_real(pred_results, label_dict)
    y_true_real = map_onehot_to_real(y_pred, label_dict)   
    plot_conf_matrix(
        y_true_real, y_pred_real, "PRED", plot_path, label_dict, capsel, growsel, netpcsize
    ) 
    X_img_1_val, X_img_2_val, X_pc_val, X_metrics_val = normalize_data(
        X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val
    )
    check_data(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val)
    val_results = np.zeros_like(y_val)
    for _ in range(10):
        predictions = trained_model.predict(
            [X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val], batch_size=8, verbose=1
        )
        val_results += predictions
    val_results /= 10
    y_val_real = map_onehot_to_real(val_results, label_dict)
    y_true_real = map_onehot_to_real(y_val, label_dict)
    plot_conf_matrix(
        y_true_real, y_val_real, "VAL", plot_path, label_dict, capsel, growsel, netpcsize
    )
        
def check_label_corruption(one_hot_labels):
    """
    Check for corruption in one-hot encoded labels.

    Args:
    one_hot_labels: Array of one-hot encoded labels.

    Returns:
    True/False: True if corruption is found, False otherwise.
    """
    # Check for NaN values
    if np.isnan(one_hot_labels).any():
        return True
    # Check if each label vector has exactly one element set to 1
    if not np.all(np.sum(one_hot_labels, axis=1) == 1):
        return True
    # Check if any label vector has all elements set to 0
    if not np.all(np.any(one_hot_labels, axis=1)):
        return True
    # No corruption found
    return False

def plot_and_save_history(history, checkpoint_dir, capsel, growsel, netpcsize, fwf_av):
    """
    Plot training metrics to graphs.

    Args:
    history: Keras object with training metrics saved.
    checkpoint_dir: Directory to save plots in.
    capsel: User-specified acquisition method.
    growsel: User-specified leaf-condition.
    netpcsize: Resampling target number of points.
    fwf_av: True/False - Presence of FWF data.
    """
    # Create a directory for plots if it doesn't exist
    if fwf_av == True:
        plot_path = os.path.join("plots_fwf_" + capsel + "_" + growsel + "_" + str(netpcsize))
    else:
        plot_path = os.path.join("plots_" + capsel + "_" + growsel + "_" + str(netpcsize))
    plots_dir = os.path.join(checkpoint_dir, plot_path)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(plots_dir + "/" + str(capsel) + "_" + growsel + "_" + str(netpcsize) + "_loss.png"))
    plt.close()
    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(plots_dir + "/" + str(capsel) + "_" + growsel + "_" + str(netpcsize) + "_accuracy.png"))
    plt.close()
    # Plot training & validation precision values
    plt.figure()
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('Model Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(plots_dir + "/" + str(capsel) + "_" + growsel + "_" + str(netpcsize) + "_precision.png"))
    plt.close()
    # Plot training & validation recall values
    plt.figure()
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('Model Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(plots_dir + "/" + str(capsel) + "_" + growsel + "_" + str(netpcsize) + "_recall.png"))
    plt.close()
    # Plot training & validation AUC values
    plt.figure()
    plt.plot(history.history['pr_curve'])
    plt.plot(history.history['val_pr_curve'])
    plt.title('Model area under PR-Curve')
    plt.ylabel('Area under PR-Curve')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(plots_dir + "/" + str(capsel) + "_" + growsel + "_" + str(netpcsize) + "_aucpr.png"))
    plt.close()
    # Plot training & validation AUC values
    plt.figure()
    plt.plot(history.history['pr_at_rec'])
    plt.plot(history.history['val_pr_at_rec'])
    plt.title('Precision at Recall (0.85)')
    plt.ylabel('Precision at Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(plots_dir + "/" + str(capsel) + "_" + growsel + "_" + str(netpcsize) + "_pr_at_rec.png"))
    plt.close()
    # Plot training & validation AUC values
    plt.figure()
    plt.plot(history.history['rec_at_pr'])
    plt.plot(history.history['val_rec_at_pr'])
    plt.title('Recall at Precision (0.85)')
    plt.ylabel('Recall at Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(plots_dir + "/" + str(capsel) + "_" + growsel + "_" + str(netpcsize) + "_rec_at_pr.png"))
    plt.close()
    return plots_dir

def plot_conf_matrix(true_labels, predicted_labels, name, plot_path, label_dict, capsel, growsel, netpcsize):
    """
    Plot confusion matrix.

    Args:
    true_labels: Labels extracted from the source dataset.
    predicted_labels: Labels predicted by MMTSCNet.
    modeldir: Savepath for the model.
    plot_path: Savepath for the plots.
    label_dict: Dictionary to translate one-hot encoded labels to textual labels.
    capsel: User-specified acquisition method.
    growsel: User-specified leaf-condition.
    netpcsize: Resampling target number of points.
    """
    min_length = min(len(true_labels), len(predicted_labels))
    true_labels = true_labels[:min_length]
    predicted_labels = predicted_labels[:min_length]
    logging.debug("Length of true_labels: %d, Length of predicted_labels: %d", len(true_labels), len(predicted_labels))
    # Get the unique labels from the true labels
    unique_labels = sorted(set(true_labels) | set(predicted_labels))
    # Create the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='twilight', cbar=False, xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(plot_path, f'conf-matrix_{capsel}_{growsel}_{str(netpcsize)}_{name}.png'))
    plt.close()

def plot_best_epoch_metrics(history, modeldir):
    """
    Plot a table of training metrics in columns.

    Args:
        history: Keras object with training metrics saved.
        modeldir: Save path for model.
    """
    # Extract the epoch with the best validation accuracy
    best_epoch = np.argmax(history.history['val_accuracy'])
    # Extract metrics for the best epoch
    best_metrics = {metric: values[best_epoch] for metric, values in history.history.items()}
    # Convert the metrics to a DataFrame and transpose it for column layout
    metrics_df = pd.DataFrame(best_metrics, index=[f'Epoch {best_epoch}']).transpose()
    # Plot the metrics as a table
    fig, ax = plt.subplots(figsize=(10, len(metrics_df) * 0.5))  # Adjust figure size for better readability
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=metrics_df.values,
        rowLabels=metrics_df.index,
        colLabels=metrics_df.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(metrics_df.columns))))
    # Save the table as an image
    plt.savefig(os.path.join(modeldir, 'best_epoch_metrics.png'))
    plt.close()

def map_onehot_to_real(onehot_lbls, onehot_to_text_dict):
    """
    Maps one-hot encoded labels to textual labels.

    Args:
    onehot_lbls: Array of one-hot encoded labels.
    onehot_to_text_dict: Dictionary to translate one-hot encoded labels to textual labels.

    Returns:
    np.array(text_labels): An array of the translated textual labels.
    """
    if onehot_lbls.ndim == 1:
        # Predictions are already class labels
        predicted_classes = onehot_lbls
        text_labels = [onehot_to_text_dict[idx] for idx in predicted_classes]
        return np.array(text_labels)
    elif onehot_lbls.ndim == 2:
        # Predictions are probabilities for each class
        predicted_classes = np.argmax(onehot_lbls, axis=1)
        text_labels = [onehot_to_text_dict[idx] for idx in predicted_classes]
        return np.array(text_labels)
    else:
        raise ValueError("Unexpected prediction shape: {}".format(onehot_lbls.shape))

def check_data(X_train, X_img_1, X_img_2, X_metrics, y_train):
    """
    Checks data for corruption.

    Args:
    X_train: Point clouds.
    X_img_1: Frontal images.
    X_img_2: Sideways images.
    X_metrics: Numerical features.
    y_train: Labels.
    """
    assert not np.isnan(X_train).any(), "Training pointclouds contain NaN values"
    assert not np.isnan(X_img_1).any(), "Training images (first set) contain NaN values"
    assert not np.isnan(X_img_2).any(), "Training images (second set) contain NaN values"
    assert not np.isnan(X_metrics).any(), "Training metrics contain NaN values"
    assert not np.isnan(y_train).any(), "Training labels contain NaN values"
    assert not np.isinf(X_train).any(), "Training pointclouds contain infinite values"
    assert not np.isinf(X_img_1).any(), "Training images (first set) contain infinite values"
    assert not np.isinf(X_img_2).any(), "Training images (second set) contain infinite values"
    assert not np.isinf(X_metrics).any(), "Training metrics contain infinite values"
    assert not np.isinf(y_train).any(), "Training labels contain infinite values"
    assert np.min(X_train) >= 0 and np.max(X_train) <= 1, "Training pointclouds are not normalized"
    assert np.min(X_img_1) >= 0 and np.max(X_img_1) <= 1, "Training images (first set) are not normalized"
    assert np.min(X_img_2) >= 0 and np.max(X_img_2) <= 1, "Training images (second set) are not normalized"
    assert np.min(X_metrics) >= 0 and np.max(X_metrics) <= 1, "Training metrics are not normalized"
    
def normalize_data(X_pc, X_img_1, X_img_2, X_metrics):
    """
    Normalizes input data to common conditions while preventing NaN values.
    """
    X_img_1 = X_img_1 / 255.0
    X_img_2 = X_img_2 / 255.0
    # Point cloud normalization
    scaler_pc = MinMaxScaler()
    X_pc = scaler_pc.fit_transform(X_pc.reshape(-1, X_pc.shape[-1])).reshape(X_pc.shape)
    # Metrics normalization with safeguard
    scaler_metrics = MinMaxScaler()
    min_vals = np.min(X_metrics, axis=0)
    max_vals = np.max(X_metrics, axis=0)
    # Identify constant columns
    constant_columns = np.where(min_vals == max_vals)[0]
    # Replace constant columns with zeros to prevent division by zero
    for col in constant_columns:
        X_metrics[:, col] = np.mean(X_metrics[:, col])
    # Apply scaling
    X_metrics = scaler_metrics.fit_transform(X_metrics.reshape(-1, X_metrics.shape[-1])).reshape(X_metrics.shape)
    # Clipping
    X_pc = np.clip(X_pc, 0, 1)
    X_metrics = np.clip(X_metrics, 0, 1)
    X_img_1 = np.clip(X_img_1, 0, 1)
    X_img_2 = np.clip(X_img_2, 0, 1)
    return X_img_1, X_img_2, X_pc, X_metrics

def generate_class_weights(y_train):
    """
    Generates class-weights to combat bias.

    Args:
    y_train: Labels.

    Returns:
    dict(enumerate(class_weights)): Class weights for each individual class.
    """
    y_train_int = np.argmax(y_train, axis=1)
    # Get the unique classes
    classes = np.unique(y_train_int)
    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_int)
    # Create a dictionary to pass to the fit method
    return dict(enumerate(class_weights))

class DataGenerator(Sequence):
    """
    Generates datasets in batches.

    Args:
    Sequence: Data.

    Returns:
    Batched data: A batch of data.
    """
    def __init__(self, X_pc, X_img_f, X_img_s, X_metrics, y, batch_size):
        self.X_pc = np.array(X_pc)
        self.X_img_f = np.array(X_img_f)
        self.X_img_s = np.array(X_img_s)
        self.X_metrics = np.array(X_metrics)
        self.y = np.array(y)
        self.batch_size = batch_size
        self.indices = np.arange(len(y))
        self.on_epoch_end()
    def __len__(self):
        # Calculate batch indices
        with tf.device('/CPU:0'):
            return int(np.floor(len(self.y) / float(self.batch_size)))
    def __getitem__(self, index):
        # Compile batch of data on CPU
        with tf.device('/CPU:0'):
            batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
            X_pc_batch = self.X_pc[batch_indices]
            X_img_f_batch = self.X_img_f[batch_indices]
            X_img_s_batch = self.X_img_s[batch_indices]
            X_metrics_batch = self.X_metrics[batch_indices]
            y_batch = self.y[batch_indices]
            return [X_pc_batch, X_img_f_batch, X_img_s_batch, X_metrics_batch], y_batch
    def on_epoch_end(self):
        # Shuffle data on CPU
        with tf.device('/CPU:0'):
            np.random.shuffle(self.indices)       

class PointCloudExtractor(tf.keras.layers.Layer):
    """
    Optimized Point Cloud Feature Extractor with:
    - Multi-Scale Grouping (MSG) with Ball Query
    - Residual Connections (if shapes match)
    - Improved T-Net with Orthogonality Regularization
    - Farthest Point Sampling (FPS)
    """
    def __init__(self, num_points, hp, **kwargs):
        super(PointCloudExtractor, self).__init__(**kwargs)
        self.num_points = num_points
        self.hp = hp

    def build(self, input_shape):
        # === Hyperparameter Configuration ===
        num_conv1d = self.hp.Choice('pce_depth', [2, 3])
        hp_units = self.hp.Choice('pce_units', [128, 256])
        hp_dropout_rate = self.hp.Float('pce_dropout_rate', min_value=0.05, max_value=0.2, step=0.025)
        hp_regularizer = self.hp.Float('pce_regularization', min_value=0.00001, max_value=0.0001, step=0.00001)
        hp_neighbors = self.hp.Int('pce_neighbors', min_value=8, max_value=32, step=8)

        # === Multi-Scale Grouping Radii ===
        self.radii = [
            self.hp.Float('pce_msg_radius_1', min_value=0.005, max_value=0.2, step=0.005),
            self.hp.Float('pce_msg_radius_2', min_value=0.05, max_value=0.4, step=0.005),
            self.hp.Float('pce_msg_radius_3', min_value=0.1, max_value=0.75, step=0.005),
            self.hp.Float('pce_msg_radius_4', min_value=0.3, max_value=0.95, step=0.005),
            self.hp.Float('pce_msg_radius_5', min_value=0.6, max_value=1.1, step=0.005)
        ]
        self.msg_neighbors = hp_neighbors

        # === Feature Extraction Convolutions ===
        self.conv1 = Conv1D(hp_units, 1, padding="same", kernel_regularizer=L1L2(l1=hp_regularizer, l2=hp_regularizer), name="pce_conv_01")
        self.norm1 = GroupNormalization(groups=16, name="pce_lnorm_01")
        self.act1 = Activation('swish', name="pce_swish_01")
        self.dropout1 = Dropout(hp_dropout_rate, name="pce_dropout_01")

        self.conv_blocks = []
        for i in range(num_conv1d):
            filters = max(32, (hp_units // (i + 1)))
            filters = filters - (filters % 16)
            conv = Conv1D(filters, 1, padding='same', kernel_regularizer=L1L2(l1=hp_regularizer*(i + 0.5), l2=hp_regularizer*(i + 1)), name=f"pce_conv_b_{i}")
            norm = GroupNormalization(groups=16, name=f"pce_lnorm_b_{i}")
            act = Activation('swish', name=f"pce_swish_b_{i}")
            dropout = Dropout(hp_dropout_rate, name=f"pce_dropout_b_{i}")
            self.conv_blocks.append((conv, norm, act, dropout))

        self.residual_conv = Conv1D(hp_units, 1, padding="same", name="pce_conv_res_01")
        self.global_pool = HybridPooling(name="pce_hybpool_01")
        self.global_norm = GroupNormalization(groups=16, name="pce_lnorm_01")

        self.global_feature_mix = Dense(hp_units, activation='swish', name="pce_feature_mix")

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        point_cloud_transformed = inputs

        # === Compute Pairwise Distances for Ball Query ===
        distances = tf.norm(
            tf.expand_dims(point_cloud_transformed, axis=2) - tf.expand_dims(point_cloud_transformed, axis=1), axis=-1
        )  

        grouped_features = []
        for radius in self.radii:
            num_neighbors = self.msg_neighbors

            #indices = tfg.nn.knn.knn(point_cloud_transformed, k=num_neighbors)
            within_radius_mask = tf.cast(distances <= radius, tf.float32)
            indices = tf.math.top_k(within_radius_mask, k=num_neighbors)[1]
            grouped_feature = tf.gather(point_cloud_transformed, indices, batch_dims=1)
            grouped_features.append(grouped_feature)

        # === Feature Aggregation with Learnable Mixing ===
        features = tf.concat(grouped_features, axis=-1)
        features = self.global_feature_mix(features)

        # === Residual Connection ===
        residual = self.residual_conv(features)

        # === Convolution Block ===
        features = self.conv1(features)
        features = self.norm1(features)
        features = self.act1(features)
        features = self.dropout1(features)

        # === Residual Connection ===
        residual = tf.reshape(residual, tf.shape(features))
        features = tf.add(features, residual)

        # === Additional Convolutions ===
        for conv, norm, act, dropout in self.conv_blocks:
            features = conv(features)
            features = norm(features)
            features = act(features)
            features = dropout(features)

        batch_size = tf.shape(features)[0]
        num_points = tf.shape(features)[1]
        new_features_dim = features.shape[2] * features.shape[3]
        features = tf.reshape(features, (batch_size, num_points, new_features_dim))

        return self.global_pool(features)
    
class EfficientNetModel(tf.keras.layers.Layer):
    """
    MMTSCNet Image Processor with Partial Fine-Tuning.
    """
    def __init__(self, img_input_shape, **kwargs):
        super(EfficientNetModel, self).__init__(**kwargs)
        self.img_input_shape = img_input_shape

    def build(self, input_shape):
        base_model = EfficientNetV2S(include_top=False, input_shape=self.img_input_shape, pooling='avg')
        # Step 1: Freeze Entire DenseNet
        for layer in base_model.layers:
            layer.trainable = False
        # Step 2: Unfreeze Last 50 Layers
        for layer in base_model.layers[-20:]:
            layer.trainable = True
        self.base_model = base_model

        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(0.2)
    
    def call(self, inputs):
        x = self.base_model(inputs)  # Extract Features
        x = self.layer_norm(x)  # Use LayerNormalization
        return self.dropout(x)

    def get_config(self):
        config = super(EfficientNetModel, self).get_config()
        config.update({'img_input_shape': self.img_input_shape})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class EnhancedMetricsModel(tf.keras.layers.Layer):
    """
    Improved Metrics-MLP with:
    - Corrected Skip Connections (ResNet-style)
    - Swish Activation for better gradients
    - Layer Normalization for stable training
    - Adaptive Dropout
    """
    def __init__(self, hp, **kwargs):
        super(EnhancedMetricsModel, self).__init__(**kwargs)
        self.hp = hp

    def build(self, input_shape):
        # Hyperparameter Configuration
        units = self.hp.Choice('metrics_units', values=[128, 256, 512])
        dropout_rate = self.hp.Float('metrics_dropout_rate', min_value=0.2, max_value=0.35, step=0.025)
        regularization = self.hp.Float('metrics_regularization', min_value=0.00001, max_value=0.002, step=0.00001)

        # === Define Layers in build() ===
        self.dense1 = Dense(units, activation=None, kernel_regularizer=L1L2(l1=regularization/16, l2=regularization/8), name="metrics_dense_1")
        self.norm1 = LayerNormalization(name="metrics_norm_1")
        self.act1 = Activation('swish', name="metrics_swish_1")
        self.dropout1 = Dropout(dropout_rate, name="metrics_dropout_1")

        self.dense2 = Dense(units // 2, activation=None, kernel_regularizer=L1L2(l1=regularization/8, l2=regularization/4), name="metrics_dense_2")
        self.norm2 = LayerNormalization(name="metrics_norm_2")
        self.act2 = Activation('swish', name="metrics_swish_2")
        self.dropout2 = Dropout(dropout_rate/2, name="metrics_dropout_2")

        self.dense3 = Dense(units // 4, activation=None, kernel_regularizer=L1L2(l1=regularization/4, l2=regularization/2), name="metrics_dense_3")
        self.norm3 = LayerNormalization(name="metrics_norm_3")
        self.act3 = Activation('swish', name="metrics_swish_3")
        self.dropout3 = Dropout(dropout_rate/4, name="metrics_dropout_3")

        self.dense4 = Dense(units // 8, activation=None, kernel_regularizer=L1L2(l1=regularization/2, l2=regularization), name="metrics_dense_4")
        self.norm4 = LayerNormalization(name="metrics_norm_4")
        self.act4 = Activation('swish', name="metrics_swish_4")
        self.dropout4 = Dropout(dropout_rate/8, name="metrics_dropout_4")

        # === Projection Layers for Skip Connections ===
        self.proj1 = Dense(units, activation=None, name="metrics_proj_1")
        self.proj2 = Dense(units // 2, activation=None, name="metrics_proj_2")
        self.proj3 = Dense(units // 4, activation=None, name="metrics_proj_3")

        super(EnhancedMetricsModel, self).build(input_shape)

    def call(self, inputs):
        """
        Applies the model with skip connections and shape matching.
        """
        x = inputs

        # === First Layer ===
        shortcut = x
        x = self.dense1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        if K.int_shape(shortcut)[-1] != K.int_shape(x)[-1]:
            shortcut = self.proj1(shortcut)
        x = Add(name="metrics_residual_add_1")([x, shortcut])

        # === Second Layer ===
        shortcut = x
        x = self.dense2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        if K.int_shape(shortcut)[-1] != K.int_shape(x)[-1]:
            shortcut = self.proj2(shortcut)
        x = Add(name="metrics_residual_add_2")([x, shortcut])

        # === Third Layer ===
        shortcut = x
        x = self.dense3(x)
        x = self.norm3(x)
        x = self.act3(x)
        x = self.dropout3(x)

        if K.int_shape(shortcut)[-1] != K.int_shape(x)[-1]:
            shortcut = self.proj3(shortcut)
        x = Add(name="metrics_residual_add_3")([x, shortcut])

        # === Fourth Layer ===
        x = self.dense4(x)
        x = self.norm4(x)
        x = self.act4(x)
        x = self.dropout4(x)

        return x

    def get_config(self):
        config = super(EnhancedMetricsModel, self).get_config()
        config.update({'hp': self.hp})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
def residual_dense_block(x, units, dropout_rate, regularizer_value, name_prefix):
    """
    Implements a residual dense block with proper skip connection handling.
    Ensures skip connection dimensions always match the output.
    """
    if units <= 32:
        num_groups = 1
    elif units <= 64:
        num_groups = 4
    elif units <= 128:
        num_groups = 8
    elif units <= 256:
        num_groups = 16
    else:
        num_groups = 32
    adaptive_dropout = dropout_rate * (1 + (units / 256))  
    adaptive_regularization = regularizer_value * (1 + (units / 512))
    shortcut = x  
    x = Dense(units, activation=None, kernel_regularizer=L1L2(l1=adaptive_regularization/1.5, l2=adaptive_regularization), name=f"{name_prefix}_dense")(x)
    x = GroupNormalization(groups=num_groups, name=f"{name_prefix}_gnorm")(x)
    x = Activation('swish', name=f"{name_prefix}_swish")(x)
    x = Dropout(adaptive_dropout, name=f"{name_prefix}_dropout")(x)
    input_dim = K.int_shape(shortcut)[-1]
    if input_dim != units:
        shortcut = Dense(units, activation=None, name=f"{name_prefix}_shortcut_projection")(shortcut)  
    return Add(name=f"{name_prefix}_residual_add")([x, shortcut])

def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal loss function for handling class imbalance.

    Args:
    alpha: Scaling factor for rare classes (0.25 default for more weight on rare classes).
    gamma: Focusing parameter (2.0 default for more focus on misclassified examples).

    Returns:
    A loss function that can be used in model.compile().
    """
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)  # Avoid log(0)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)  # Reduce easy samples' impact
        return tf.reduce_sum(weight * cross_entropy, axis=-1)

    return loss

class CombinedModel(HyperModel):
    """
    MMTSCNet architecture definition.

    """
    def __init__(self, point_cloud_shape, image_shape, metrics_shape, num_classes, num_points, **kwargs):
        super(CombinedModel, self).__init__(**kwargs)
        self.point_cloud_shape = point_cloud_shape
        self.image_shape = image_shape
        self.metrics_shape = metrics_shape
        self.num_classes = num_classes
        self.num_points = num_points

    def build(self, hp):
        # === Input Definition ===
        pointnet_input = tf.keras.Input(shape=self.point_cloud_shape, name='pointnet_input')
        image_input_1 = tf.keras.Input(shape=self.image_shape, name='image_input_1')
        image_input_2 = tf.keras.Input(shape=self.image_shape, name='image_input_2')
        metrics_input = tf.keras.Input(shape=self.metrics_shape, name='metrics_input')

        image_input_1 = tf.cast(image_input_1, dtype=tf.float32)
        image_input_2 = tf.cast(image_input_2, dtype=tf.float32)

        # === Branches ===
        pointnet_branch = PointCloudExtractor(self.num_points, hp)(pointnet_input)
        image_branch_1 = EfficientNetModel(self.image_shape)(image_input_1)
        image_branch_2 = EfficientNetModel(self.image_shape)(image_input_2)
        metrics_branch = EnhancedMetricsModel(hp)(metrics_input)

        # === Hyperparameter Setup ===
        projection_units = hp.Choice('projection_units', [64, 128, 192])
        num_dense = hp.Choice('clss_depth', [2, 3])  
        units_dense = hp.Choice('clss_units', [128, 256, 512])
        dropout_clss = hp.Float('clss_dropout_rate', min_value=0.2, max_value=0.35, step=0.025)
        regularizer_value_clss = hp.Float('clss_regularization', min_value=0.00001, max_value=0.002, step=0.00001)

        # Compute attention-based feature weights
        fusion_features = Concatenate()([pointnet_branch, image_branch_1, image_branch_2, metrics_branch])
        attention_scores = Dense(128, activation="swish", kernel_regularizer=L1L2(l1=regularizer_value_clss/1.5, l2=regularizer_value_clss))(fusion_features)

        # Competitive Attention: Use softmax instead of independent sigmoid scaling
        context_vector = Dense(4, activation=None)(Concatenate()([attention_scores, fusion_features]))
        attention_weights = Activation("softmax", name="attention_weights")(context_vector)  # Apply softmax

        # Split into modality-specific attention scores
        pointnet_weight, image1_weight, image2_weight, metrics_weight = tf.split(attention_weights, 4, axis=-1)

        # === Apply Balanced Weights to Feature Extractors ===
        pointnet_branch = Multiply()([pointnet_branch, pointnet_weight])
        image_branch_1 = Multiply()([image_branch_1, image1_weight])
        image_branch_2 = Multiply()([image_branch_2, image2_weight])
        metrics_branch = Multiply()([metrics_branch, metrics_weight])

        # Apply learned weights to extracted features (not layer size!)
        pointnet_branch = pointnet_branch * (pointnet_weight + 1)
        image_branch_1 = image_branch_1 * (image1_weight + 1)
        image_branch_2 = image_branch_2 * (image2_weight + 1)
        metrics_branch = metrics_branch * (metrics_weight + 1)

        # === Projection Layers (Fixed neuron count) ===
        pointnet_branch = Dense(projection_units, activation=None, name="pce_projection")(pointnet_branch)
        image_branch_1 = Dense(projection_units, activation=None, name="img1_projection")(image_branch_1)
        image_branch_2 = Dense(projection_units, activation=None, name="img2_projection")(image_branch_2)
        metrics_branch = Dense(projection_units, activation=None, name="metrics_projection")(metrics_branch)

        # Concatenation
        x = Concatenate(name="concat_all")([pointnet_branch, image_branch_1, image_branch_2, metrics_branch])

        for i in range(1, num_dense + 1):
            units = max(16, (units_dense // (2**i)) - ((units_dense // (2**i)) % 8))
            x = residual_dense_block(x, units, dropout_clss, regularizer_value_clss, f"clss_block_{i}")

        output = Dense(self.num_classes, activation='softmax', name='output')(x)

        model = Model(inputs=[pointnet_input, image_input_1, image_input_2, metrics_input], outputs=output)
        # Learning rate setup
        initial_learning_rate = hp.Choice('learning_rate', [1e-4, 7.5e-5, 5e-5, 2.5e-5, 1e-5])
        model.compile(optimizer=Adam(learning_rate=initial_learning_rate, clipnorm=0.5),
                      loss=focal_loss(alpha=0.25, gamma=2.0),
                      metrics=['accuracy', tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="pr_curve", curve="PR"), tf.keras.metrics.PrecisionAtRecall(0.85, name="pr_at_rec"), tf.keras.metrics.RecallAtPrecision(0.85, name="rec_at_pr")])
        
        trainable_params = np.sum([K.count_params(w) for w in model.trainable_weights])
        param_memory = (trainable_params * 3.0) / (1024 * 1024)
        activation_memory = param_memory * 2.0
        gradient_memory = param_memory
        optimizer_memory = param_memory * 2.0
        total_memory = param_memory + activation_memory + gradient_memory + optimizer_memory
        logging.info("Trainable Parameters: %s - Estimated VRAM usage: %s MB", trainable_params, total_memory)

        return model

    def get_untrained_model(self, best_hyperparameters):
        return self.build(best_hyperparameters)

    def get_config(self):
        config = super(CombinedModel, self).get_config()
        config.update({
            'point_cloud_shape': self.point_cloud_shape,
            'image_shape': self.image_shape,
            'metrics_shape': self.metrics_shape,
            'num_classes': self.num_classes,
            'num_points': self.num_points
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)