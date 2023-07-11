import tensorflow as tf
from trainer import get_initializer


def baseline_ann(input_shape, output_shape, output_activation, model_config):
    if len(input_shape) == 1:
        input_shape = (input_shape[0],)

    if "kernel_initializer" in model_config:
        kernel_initializer = get_initializer(
            model_config["kernel_initializer"])
    else:
        kernel_initializer = 'glorot_uniform'

    if "bias_initializer" in model_config:
        bias_initializer = get_initializer(model_config["bias_initializer"])
    else:
        bias_initializer = "zeros"

    inputs = tf.keras.Input(shape=input_shape, name="input_layer")
    x = tf.keras.layers.Dense(64, activation="relu", kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer, name="dense_1")(inputs)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer, name="dense_2")(x)
    outputs = tf.keras.layers.Dense(
        output_shape, activation=output_activation, name="predictions")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def baseline_cnn(input_shape, output_shape, output_activation, model_config):
    if len(input_shape) == 1:
        input_shape = (input_shape[0],)

    if "kernel_initializer" in model_config:
        kernel_initializer = get_initializer(
            model_config["kernel_initializer"])
    else:
        kernel_initializer = 'glorot_uniform'

    if "bias_initializer" in model_config:
        bias_initializer = get_initializer(model_config["bias_initializer"])
    else:
        bias_initializer = "zeros"

    inputs = tf.keras.Input(shape=input_shape, name="input_layer")
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(
        output_shape, activation=output_activation, name="predictions")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


class ModelBuilder(object):
    def __init__(self, model_id, input_shape, output_shape, output_activation, model_config):
        self.model_id = model_id
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.output_activation = output_activation
        self.model_config = model_config

    def get_model(self):
        if self.model_id == "baseline-ann":
            model = baseline_ann(
                self.input_shape, self.output_shape, self.output_activation, self.model_config)
        elif self.model_id == "baseline-cnn":
            model = baseline_cnn(
                self.input_shape, self.output_shape, self.output_activation, self.model_config)
        else:
            raise ValueError(f"Unknown model_id {self.model_id}")
        return model