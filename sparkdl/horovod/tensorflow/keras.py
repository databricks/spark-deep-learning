# Copyright 2018 Databricks, Inc.
#
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-instance-attributes
# pylint: disable=logging-format-interpolation
# pylint: disable=invalid-name

import time
from tensorflow import keras

from sparkdl.horovod import log_to_driver

__all__ = ["LogCallback"]


class LogCallback(keras.callbacks.Callback):
    """
    A simple HorovodRunner log callback that streams event logs to notebook cell output.
    """

    def __init__(self, per_batch_log=False):
        """
        :param per_batch_log: whether to output logs per batch, default: False.
        """
        raise NotImplementedError()

    def on_epoch_begin(self, epoch, logs=None):
        raise NotImplementedError()

    def on_batch_end(self, batch, logs=None):
        raise NotImplementedError()

    def on_epoch_end(self, epoch, logs=None):
        raise NotImplementedError()
