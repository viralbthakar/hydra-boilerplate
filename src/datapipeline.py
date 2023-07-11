import numpy as np
import tensorflow as tf


def load_mnist(flatten=False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    if flatten:
        x_train = np.reshape(x_train, (-1, 784))
        x_test = np.reshape(x_test, (-1, 784))
        return (x_train, y_train), (x_test, y_test)
    else:
        return (x_train, y_train), (x_test, y_test)


def load_cifar10(flatten=False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if flatten:
        x_train = np.reshape(x_train, (-1, 3072))
        x_test = np.reshape(x_test, (-1, 3072))
        return (x_train, y_train), (x_test, y_test)
    else:
        return (x_train, y_train), (x_test, y_test)


class DatasetBuilder(object):
    def __init__(self, dataset_id, flatten):
        self.dataset_id = dataset_id
        self.flatten = flatten

    def get_dataset(self):
        if self.dataset_id == "mnist":
            (x_train, y_train), (x_test, y_test) = load_mnist(flatten=self.flatten)
            return (x_train, y_train), (x_test, y_test)
        elif self.dataset_id == "cifar10":
            (x_train, y_train), (x_test, y_test) = load_cifar10(
                flatten=self.flatten)
            return (x_train, y_train), (x_test, y_test)
        else:
            raise ValueError(f"Unknown dataset_id {self.dataset_id}")


class DatapipelineBuilder(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def create_datapipeline(self, batch_size=32):
        datapipeline = tf.data.Dataset.from_tensor_slices((self.X, self.Y))
        datapipeline = datapipeline.shuffle(
            buffer_size=1024, reshuffle_each_iteration=True)
        datapipeline = datapipeline.batch(batch_size=batch_size)
        datapipeline = datapipeline.prefetch(1)
        return datapipeline