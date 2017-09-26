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

import six

import tensorflow as tf

from pyspark.ml.param import TypeConverters

import sparkdl.graph.utils as tfx
import sparkdl.utils.keras_model as kmutil

__all__ = ['SparkDLTypeConverters']


class SparkDLTypeConverters(object):
    """
    .. note:: DeveloperApi

    Factory methods for common type conversion functions for :py:func:`Param.typeConverter`.
    These methods are similar to :py:class:`spark.ml.param.TypeConverters`.
    They provide support for the `Params` types introduced in Spark Deep Learning Pipelines.
    """
    @staticmethod
    def toTFGraph(value):
        if isinstance(value, tf.Graph):
            return value
        else:
            raise TypeError("Could not convert %s to TensorFlow Graph" % type(value))

    @staticmethod
    def asColumnToTensorNameMap(value):
        """
        Convert a value to a column name to :py:obj:`tf.Tensor` name mapping
        as a sorted list of string pairs, if possible.
        """
        if isinstance(value, dict):
            strs_pair_seq = []
            for _maybe_col_name, _maybe_tnsr_name in value.items():
                # Check if the non-tensor value is of string type
                _col_name = _get_strict_col_name(_maybe_col_name)
                # Check if the tensor name is actually valid
                _tnsr_name = _get_strict_tensor_name(_maybe_tnsr_name)
                strs_pair_seq.append((_col_name, _tnsr_name))

            return sorted(strs_pair_seq)

        err_msg = "Could not convert [type {}] {} to column name to tf.Tensor name mapping"
        raise TypeError(err_msg.format(type(value), value))

    @staticmethod
    def asTensorNameToColumnMap(value):
        """
        Convert a value to a :py:obj:`tf.Tensor` name to column name mapping
        as a sorted list of string pairs, if possible.
        """
        if isinstance(value, dict):
            strs_pair_seq = []
            for _maybe_tnsr_name, _maybe_col_name in value.items():
                # Check if the non-tensor value is of string type
                _col_name = _get_strict_col_name(_maybe_col_name)
                # Check if the tensor name is actually valid
                _tnsr_name = _get_strict_tensor_name(_maybe_tnsr_name)
                strs_pair_seq.append((_tnsr_name, _col_name))

            return sorted(strs_pair_seq)

        err_msg = "Could not convert [type {}] {} to tf.Tensor name to column name mapping"
        raise TypeError(err_msg.format(type(value), value))

    @staticmethod
    def toTFHParams(value):
        """ Convert a value to a :py:obj:`tf.contrib.training.HParams` object, if possible. """
        if isinstance(value, tf.contrib.training.HParams):
            return value
        else:
            raise TypeError("Could not convert %s to TensorFlow HParams" % type(value))

    @staticmethod
    def toStringOrTFTensor(value):
        """ Convert a value to a str or a :py:obj:`tf.Tensor` object, if possible. """
        if isinstance(value, tf.Tensor):
            return value
        try:
            return TypeConverters.toString(value)
        except Exception as exc:
            err_msg = "Could not convert [type {}] {} to tf.Tensor or str. {}"
            raise TypeError(err_msg.format(type(value), value, exc))

    @staticmethod
    def supportedNameConverter(supportedList):
        """
        Create a converter that try to check if a value is part of the supported list.

        :param supportedList: list, containing supported objects.
        :return: a converter that try to convert a value if it is part of the `supportedList`.
        """
        def converter(value):
            if value in supportedList:
                return value
            err_msg = "[type {}] {} is not in the supported list: {}"
            raise TypeError(err_msg.format(type(value), str(value), supportedList))

        return converter

    @staticmethod
    def toKerasLoss(value):
        """ Convert a value to a name of Keras loss function, if possible """
        if kmutil.is_valid_loss_function(value):
            return value
        err_msg = "Named loss not supported in Keras: [type {}] {}"
        raise ValueError(err_msg.format(type(value), value))

    @staticmethod
    def toKerasOptimizer(value):
        """ Convert a value to a name of Keras optimizer, if possible """
        if kmutil.is_valid_optimizer(value):
            return value
        err_msg = "Named optimizer not supported in Keras: [type {}] {}"
        raise TypeError(err_msg.format(type(value), value))


def _get_strict_tensor_name(_maybe_tnsr_name):
    """ Check if the input is a valid tensor name """
    try:
        assert isinstance(_maybe_tnsr_name, six.string_types), \
            "must provide a strict tensor name as input, but got {}".format(type(_maybe_tnsr_name))
        assert tfx.as_tensor_name(_maybe_tnsr_name) == _maybe_tnsr_name, \
            "input {} must be a valid tensor name".format(_maybe_tnsr_name)
    except Exception as exc:
        err_msg = "Can NOT convert [type {}] {} to tf.Tensor name: {}"
        raise TypeError(err_msg.format(type(_maybe_tnsr_name), _maybe_tnsr_name, exc))
    else:
        return _maybe_tnsr_name


def _get_strict_col_name(_maybe_col_name):
    """ Check if the given colunm name is a valid column name """
    # We only check if the column name candidate is a string type
    if not isinstance(_maybe_col_name, six.string_types):
        err_msg = 'expect string type but got type {} for {}'
        raise TypeError(err_msg.format(type(_maybe_col_name), _maybe_col_name))
    return _maybe_col_name
