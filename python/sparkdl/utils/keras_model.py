#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import shutil
import tempfile

import keras
from keras.models import load_model as _load_keras_hdf5_model

__all__ = ['model_to_bytes', 'bytes_to_model', 'bytes_to_h5file',
           'is_valid_loss_function', 'is_valid_optimizer']


def model_to_bytes(model):
    """
    Serialize the Keras model to HDF5 and load the file as bytes.
    This saves the Keras model to a temp file as an intermediate step.
    :return: str containing the model data
    """
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "model.h5")
    try:
        model.save(temp_path)
        with open(temp_path, mode='rb') as fin:
            file_bytes = fin.read()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return file_bytes


def bytes_to_h5file(modelBytes):
    """
    Dump HDF5 file content bytes to a local file
    :return: path to the file
    """
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "model.h5")
    with open(temp_path, mode='wb') as fout:
        fout.write(modelBytes)
    return temp_path


def bytes_to_model(modelBytes, remove_temp_path=True):
    """
    Convert a Keras model from a byte string to a Keras model instance.
    This saves the Keras model to a temp file as an intermediate step.
    """
    temp_path = bytes_to_h5file(modelBytes)
    try:
        model = _load_keras_hdf5_model(temp_path)
    finally:
        if remove_temp_path:
            temp_dir = os.path.dirname(temp_path)
            shutil.rmtree(temp_dir, ignore_errors=True)
    return model


def _get_loss_function(identifier):
    """
    Retrieves a Keras loss function instance.
    :param: identifier str, name of the loss function
    :return: A Keras loss function instance if the identifier is valid
    """
    return keras.losses.get(identifier)


def is_valid_loss_function(identifier):
    """ Check if a named loss function is supported in Keras """
    try:
        _loss = _get_loss_function(identifier)
        return _loss is not None
    except ValueError:
        return False


def _get_optimizer(identifier):
    """
    Retrieves a Keras Optimizer instance.
    :param: identifier str, name of the optimizer
    :return: A Keras optimizer instance if the identifier is valid
    """
    return keras.optimizers.get(identifier)


def is_valid_optimizer(identifier):
    """ Check if a named optimizer is supported in Keras """
    try:
        _optim = _get_optimizer(identifier)
        return _optim is not None
    except ValueError:
        return False
