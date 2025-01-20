import tensorflow as tf
import numpy as np
from keras_tuner import HyperParameters
from keras.layers import Conv1D, BatchNormalization, ReLU, Dropout, GlobalMaxPooling1D, Dense, Reshape
from keras.regularizers import L1L2
from keras.applications import DenseNet121

# =======================
# ✅ Dummy Hyperparameters
# =======================
class DummyHyperParameters:
    def Choice(self, name, values):
        return np.random.choice(values)

    def Float(self, name, min_value, max_value, step):
        return np.random.uniform(min_value, max_value)

    def Int(self, name, min_value, max_value, step):
        return np.random.randint(min_value, max_value + 1)

# =======================
# ✅ DenseNetModel
# =======================
class DenseNetModel(tf.keras.layers.Layer):
    """
    MMTSCNet image processor (DenseNet121).
    """
    def __init__(self, img_input_shape, model_name, **kwargs):
        super(DenseNetModel, self).__init__(**kwargs)
        self.img_input_shape = img_input_shape
        self._model_name = model_name

    def build(self, input_shape):
        self.model = DenseNet121(include_top=False, input_shape=self.img_input_shape, pooling='avg')

    def call(self, inputs):
        x = self.model(inputs)
        return x

    def get_config(self):
        config = super(DenseNetModel, self).get_config()
        config.update({
            'img_input_shape': self.img_input_shape,
            'model_name': self._model_name
        })
        return config

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        return cls(**config)

# =======================
# ✅ PointCloudExtractor
# =======================
class PointCloudExtractor(tf.keras.layers.Layer):
    """
    Optimized Point Cloud Extractor with Multi-Scale Grouping (MSG) and Residual Connections.
    """
    def __init__(self, num_points, hp, **kwargs):
        super(PointCloudExtractor, self).__init__(**kwargs)
        self.num_points = num_points
        self.hp = hp

    def build(self, input_shape):
        num_conv1d = self.hp.Choice('pce_depth', [1, 2, 3, 4, 5])
        hp_units = self.hp.Choice('mmtsc_units', values=[256, 512, 1024])
        hp_dropout_rate = self.hp.Float('mmtsc_dropout_rate', min_value=0.025, max_value=0.2, step=0.025)
        hp_regularizer_value = self.hp.Float('mmtsc_regularization', min_value=1e-7, max_value=0.0005, step=1e-7)

        self.conv1 = Conv1D(hp_units, 1, padding="same", kernel_regularizer=L1L2(l1=hp_regularizer_value, l2=hp_regularizer_value))
        self.norm1 = BatchNormalization()
        self.relu1 = ReLU()
        self.dropout1 = Dropout(hp_dropout_rate)

        self.conv_blocks = []
        for i in range(num_conv1d):
            filters = hp_units // (i + 1)
            conv = Conv1D(filters, 1, padding='same', kernel_regularizer=L1L2(l1=hp_regularizer_value, l2=hp_regularizer_value))
            norm = BatchNormalization()
            relu = ReLU()
            dropout = Dropout(hp_dropout_rate)
            self.conv_blocks.append((conv, norm, relu, dropout))

        self.residual_conv = Conv1D(hp_units, 1, padding="same")
        self.maxp2 = GlobalMaxPooling1D()

    def call(self, inputs):
        features = self.conv1(inputs)
        features = self.norm1(features)
        features = self.relu1(features)
        features = self.dropout1(features)

        residual = self.residual_conv(features)
        residual = tf.reshape(residual, tf.shape(features))
        features = tf.add(features, residual)

        for conv, norm, relu, dropout in self.conv_blocks:
            features = conv(features)
            features = norm(features)
            features = relu(features)
            features = dropout(features)

        net = self.maxp2(features)
        return net

    def get_config(self):
        config = super(PointCloudExtractor, self).get_config()
        config.update({'num_points': self.num_points, 'hp': self.hp})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# =======================
# ✅ Testing Function
# =======================
def test_models():
    batch_size = 8
    num_points = 2048
    num_features = 3  # (X, Y, Z)
    img_shape = (224, 224, 3)

    test_pc_input = tf.random.uniform((batch_size, num_points, num_features))
    test_img_input = tf.random.uniform((batch_size, *img_shape))

    hp = DummyHyperParameters()

    try:
        # 🔹 Test PointCloudExtractor
        print("🚀 Testing PointCloudExtractor...")
        pce = PointCloudExtractor(num_points=num_points, hp=hp)
        pce.build(test_pc_input.shape)
        pc_output = pce(test_pc_input)
        print(f"✅ PointCloudExtractor Output Shape: {pc_output.shape}")

        # 🔹 Test DenseNetModel
        print("🚀 Testing DenseNetModel...")
        dnet = DenseNetModel(img_input_shape=img_shape, model_name="DenseNet_Test")
        dnet.build(test_img_input.shape)
        img_output = dnet(test_img_input)
        print(f"✅ DenseNetModel Output Shape: {img_output.shape}")

        return True

    except Exception as e:
        print("❌ Test Failed: Error encountered")
        print(e)
        return False

# Run the test
if __name__ == "__main__":
    test_models()
