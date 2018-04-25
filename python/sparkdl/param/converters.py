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

""" SparkDLTypeConverters

Type conversion utilities for defining MLlib `Params` used in Spark Deep Learning Pipelines.

.. note:: We follow the convention of MLlib to name these utilities "converters",
          but most of them act as type checkers that return the argument if it is
          the desired type and raise `TypeError` otherwise.
"""

import six
import tensorflow as tf

from sparkdl.graph.input import TFInputGraph
import sparkdl.utils.keras_model as kmutil

__all__ = ['SparkDLTypeConverters']


class SparkDLTypeConverters(object):
    """
    .. note:: DeveloperApi

    Methods for type conversion functions for :py:func:`Param.typeConverter`.
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
        Convert a value to a column name to :py:class:`tf.Tensor` name mapping
        as a sorted list (in lexicographical order) of string pairs, if possible.
        """
        if not isinstance(value, dict):
            err_msg = "Could not convert [type {}] {} to column name to tf.Tensor name mapping"
            raise TypeError(err_msg.format(type(value), value))

        strs_pair_seq = []
        for _maybe_col_name, _maybe_tnsr_name in value.items():
            _check_is_str(_maybe_col_name)
            _check_is_tensor_name(_maybe_tnsr_name)
            strs_pair_seq.append((_maybe_col_name, _maybe_tnsr_name))

        return sorted(strs_pair_seq)

    @staticmethod
    def asTensorNameToColumnMap(value):
        """
        Convert a value to a :py:class:`tf.Tensor` name to column name mapping
        as a sorted list (in lexicographical order) of string pairs, if possible.
        """
        if not isinstance(value, dict):
            err_msg = "Could not convert [type {}] {} to tf.Tensor name to column name mapping"
            raise TypeError(err_msg.format(type(value), value))

        strs_pair_seq = []
        for _maybe_tnsr_name, _maybe_col_name in value.items():
            _check_is_str(_maybe_col_name)
            _check_is_tensor_name(_maybe_tnsr_name)
            strs_pair_seq.append((_maybe_tnsr_name, _maybe_col_name))

        return sorted(strs_pair_seq)

    @staticmethod
    def toTFHParams(value):
        """
        Check that the given value is a :py:class:`tf.contrib.training.HParams` object,
        and return it. Raise an error otherwise.
        """
        if not isinstance(value, tf.contrib.training.HParams):
            raise TypeError("Could not convert %s to TensorFlow HParams" % type(value))

        return value

    @staticmethod
    def toTFTensorName(value):
        """
        Check if a value is a valid :py:class:`tf.Tensor` name and return it.
        Raise an error otherwise.
        """
        if isinstance(value, tf.Tensor):
            return value.name
        try:
            _check_is_tensor_name(value)
            return value
        except Exception as exc:
            err_msg = "Could not convert [type {}] {} to tf.Tensor name. {}"
            raise TypeError(err_msg.format(type(value), value, exc))

    @staticmethod
    def buildSupportedItemConverter(supportedList):
        """
        Create a "converter" that try to check if a value is part of the supported list of values.

        :param supportedList: list, containing supported objects.
        :return: a converter that try to check if a value is part of the `supportedList` and
        return it.
                 Raise an error otherwise.
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
        """
        Check if a value is a valid Keras loss function name and return it.
        Otherwise raise an error.
        """
        # return early in for clarify as well as less indentation
        if not kmutil.is_valid_loss_function(value):
            err_msg = "Named loss not supported in Keras: [type {}] {}"
            raise ValueError(err_msg.format(type(value), value))

        return value

    @staticmethod
    def toKerasOptimizer(value):
        """
        Check if a value is a valid name of Keras optimizer and return it.
        Otherwise raise an error.
        """
        if not kmutil.is_valid_optimizer(value):
            err_msg = "Named optimizer not supported in Keras: [type {}] {}"
            raise TypeError(err_msg.format(type(value), value))

        return value

    @staticmethod
    def toChannelOrder(value):
        if not value in ('L', 'RGB', 'BGR'):
            raise ValueError("""Unsupported channel order. Expected one of ('L', 'RGB',
            'BGR') but got '%s'""" % value)
        return value


def _check_is_tensor_name(_maybe_tnsr_name):
    """ Check if the input is a valid tensor name or raise a `TypeError` otherwise. """
    if not isinstance(_maybe_tnsr_name, six.string_types):
        err_msg = "expect tensor name to be of string type, but got [type {}]"
        raise TypeError(err_msg.format(type(_maybe_tnsr_name)))

    # The check is taken from TensorFlow's NodeDef protocol buffer.
    #   Each input is "node:src_output" with "node" being a string name and
    #   "src_output" indicating which output tensor to use from "node". If
    #   "src_output" is 0 the ":0" suffix can be omitted.  Regular inputs
    #   may optionally be followed by control inputs that have the format
    #   "^node".
    # Reference:
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/node_def.proto
    # https://stackoverflow.com/questions/36150834/how-does-tensorflow-name-tensors
    try:
        _, src_idx = _maybe_tnsr_name.split(":")
        _ = int(src_idx)
    except Exception:
        err_msg = "Tensor name must be of type <op_name>:<index>, but got {}"
        raise TypeError(err_msg.format(_maybe_tnsr_name))


def _check_is_str(_maybe_str):
    """ Check if the value is a valid string type or raise a `TypeError` otherwise. """
    # We only check if the column name candidate is a string type
    if not isinstance(_maybe_str, six.string_types):
        err_msg = 'expect string type but got type {} for {}'
        raise TypeError(err_msg.format(type(_maybe_str), _maybe_str))
