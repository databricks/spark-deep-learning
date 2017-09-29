# Copyright 2017 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# pylint: disable=wrong-spelling-in-docstring,invalid-name,import-error

""" SparkDLTypeConverters
Type conversion utilities for definition Spark Deep Learning related MLlib `Params`.
"""

import six

import tensorflow as tf

from pyspark.ml.param import TypeConverters

import sparkdl.utils.keras_model as kmutil

__all__ = ['SparkDLTypeConverters']

class SparkDLTypeConverters(object):
    """
    .. note:: DeveloperApi

    Factory methods for type conversion functions for :py:func:`Param.typeConverter`.
    These methods are similar to :py:class:`spark.ml.param.TypeConverters`.
    They provide support for the `Params` types introduced in Spark Deep Learning Pipelines.
    """

    @staticmethod
    def toTFGraph(value):
        """
        Convert a value to a :py:obj:`tf.Graph` object, if possible.
        """
        if not isinstance(value, tf.Graph):
            raise TypeError("Could not convert %s to tf.Graph" % type(value))
        return value

    @staticmethod
    def toTFInputGraph(value):
        if isinstance(value, TFInputGraph):
            return value
        else:
            raise TypeError("Could not convert %s to TFInputGraph" % type(value))

    @staticmethod
    def asColumnToTensorNameMap(value):
        """
        Convert a value to a column name to :py:obj:`tf.Tensor` name mapping
        as a sorted list of string pairs, if possible.
        """
        if not isinstance(value, dict):
            err_msg = "Could not convert [type {}] {} to column name to tf.Tensor name mapping"
            raise TypeError(err_msg.format(type(value), value))

        # Conversion logic after quick type check
        strs_pair_seq = []
        for _maybe_col_name, _maybe_tnsr_name in value.items():
            # Check if the non-tensor value is of string type
            _check_is_str(_maybe_col_name)
            # Check if the tensor name looks like a tensor name
            _check_is_tensor_name(_maybe_tnsr_name)
            strs_pair_seq.append((_maybe_col_name, _maybe_tnsr_name))

        return sorted(strs_pair_seq)

    @staticmethod
    def asTensorNameToColumnMap(value):
        """
        Convert a value to a :py:obj:`tf.Tensor` name to column name mapping
        as a sorted list of string pairs, if possible.
        """
        if not isinstance(value, dict):
            err_msg = "Could not convert [type {}] {} to tf.Tensor name to column name mapping"
            raise TypeError(err_msg.format(type(value), value))

        # Conversion logic after quick type check
        strs_pair_seq = []
        for _maybe_tnsr_name, _maybe_col_name in value.items():
            # Check if the non-tensor value is of string type
            _check_is_str(_maybe_col_name)
            # Check if the tensor name looks like a tensor name
            _check_is_tensor_name(_maybe_tnsr_name)
            strs_pair_seq.append((_maybe_tnsr_name, _maybe_col_name))

        return sorted(strs_pair_seq)

    @staticmethod
    def toTFHParams(value):
        """ Convert a value to a :py:obj:`tf.contrib.training.HParams` object, if possible. """
        if not isinstance(value, tf.contrib.training.HParams):
            raise TypeError("Could not convert %s to TensorFlow HParams" % type(value))

        return value

    @staticmethod
    def toTFTensorName(value):
        """ Convert a value to a :py:obj:`tf.Tensor` name, if possible. """
        if isinstance(value, tf.Tensor):
            return value.name
        try:
            _maybe_tnsr_name = TypeConverters.toString(value)
            _check_is_tensor_name(_maybe_tnsr_name)
            return _maybe_tnsr_name
        except Exception as exc:
            err_msg = "Could not convert [type {}] {} to tf.Tensor name. {}"
            raise TypeError(err_msg.format(type(value), value, exc))

    @staticmethod
    def buildCheckList(supportedList):
        """
        Create a converter that try to check if a value is part of the supported list.

        :param supportedList: list, containing supported objects.
        :return: a converter that try to convert a value if it is part of the `supportedList`.
        """

        def converter(value):
            """ Implementing the conversion logic """
            if value not in supportedList:
                err_msg = "[type {}] {} is not in the supported list: {}"
                raise TypeError(err_msg.format(type(value), str(value), supportedList))

            return value

        return converter

    @staticmethod
    def toKerasLoss(value):
        """ Convert a value to a name of Keras loss function, if possible """
        # return early in for clarify as well as less indentation
        if not kmutil.is_valid_loss_function(value):
            err_msg = "Named loss not supported in Keras: [type {}] {}"
            raise ValueError(err_msg.format(type(value), value))

        return value

    @staticmethod
    def toKerasOptimizer(value):
        """ Convert a value to a name of Keras optimizer, if possible """
        if not kmutil.is_valid_optimizer(value):
            err_msg = "Named optimizer not supported in Keras: [type {}] {}"
            raise TypeError(err_msg.format(type(value), value))

        return value


def _check_is_tensor_name(_maybe_tnsr_name):
    """ Check if the input is a valid tensor name """
    if not isinstance(_maybe_tnsr_name, six.string_types):
        err_msg = "expect tensor name to be of string type, but got [type {}]"
        raise TypeError(err_msg.format(type(_maybe_tnsr_name)))

    # The check is taken from TensorFlow's NodeDef protocol buffer.
    # https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/core/framework/node_def.proto#L21-L25
    try:
        _, src_idx = _maybe_tnsr_name.split(":")
        _ = int(src_idx)
    except Exception:
        err_msg = "Tensor name must be of type <op_name>:<index>, but got {}"
        raise TypeError(err_msg.format(_maybe_tnsr_name))

    return _maybe_tnsr_name


def _check_is_str(_maybe_col_name):
    """ Check if the given colunm name is a valid column name """
    # We only check if the column name candidate is a string type
    if not isinstance(_maybe_col_name, six.string_types):
        err_msg = 'expect string type but got type {} for {}'
        raise TypeError(err_msg.format(type(_maybe_col_name), _maybe_col_name))
    return _maybe_col_name
