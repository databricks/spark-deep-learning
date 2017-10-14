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
from sparkdl.graph.input import TFInputGraph
import sparkdl.utils.keras_model as kmutil

__all__ = ['SparkDLTypeConverters']

def _try_convert_tf_tensor_mapping(value, is_key_tf_tensor=True):
    if isinstance(value, dict):
        strs_pair_seq = []
        for k, v in value.items():
            try:
                if is_key_tf_tensor:
                    _pair = (tfx.as_tensor_name(k), v)
                else:
                    _pair = (k, tfx.as_tensor_name(v))
            except:
                err_msg = "Can NOT convert {} (type {}) to tf.Tensor name"
                _not_tf_op = k if is_key_tf_tensor else v
                raise TypeError(err_msg.format(_not_tf_op, type(_not_tf_op)))

            str_val = v if is_key_tf_tensor else k
            if not isinstance(str_val, six.string_types):
                err_msg = 'expect string type for {}, but got {}'
                raise TypeError(err_msg.format(str_val, type(str_val)))

            strs_pair_seq.append(_pair)

        return sorted(strs_pair_seq)

    if is_key_tf_tensor:
        raise TypeError("Could not convert %s to tf.Tensor name to str mapping" % type(value))
    else:
        raise TypeError("Could not convert %s to str to tf.Tensor name mapping" % type(value))


class SparkDLTypeConverters(object):
    @staticmethod
    def toTFGraph(value):
        if isinstance(value, tf.Graph):
            return value
        else:
            raise TypeError("Could not convert %s to TensorFlow Graph" % type(value))

    @staticmethod
    def toTFInputGraph(value):
        if isinstance(value, TFInputGraph):
            return value
        else:
            raise TypeError("Could not convert %s to TFInputGraph" % type(value))

    @staticmethod
    def asColumnToTensorNameMap(value):
        return _try_convert_tf_tensor_mapping(value, is_key_tf_tensor=False)

    @staticmethod
    def asTensorNameToColumnMap(value):
        return _try_convert_tf_tensor_mapping(value, is_key_tf_tensor=True)

    @staticmethod
    def toTFHParams(value):
        if isinstance(value, tf.contrib.training.HParams):
            return value
        else:
            raise TypeError("Could not convert %s to TensorFlow HParams" % type(value))

    @staticmethod
    def toStringOrTFTensor(value):
        if isinstance(value, tf.Tensor):
            return value
        else:
            try:
                return TypeConverters.toString(value)
            except TypeError:
                raise TypeError("Could not convert %s to tensorflow.Tensor or str" % type(value))

    @staticmethod
    def supportedNameConverter(supportedList):
        def converter(value):
            if value in supportedList:
                return value
            else:
                raise TypeError("%s %s is not in the supported list." % type(value), str(value))

        return converter

    @staticmethod
    def toKerasLoss(value):
        if kmutil.is_valid_loss_function(value):
            return value
        raise ValueError(
            "Named loss not supported in Keras: {} type({})".format(value, type(value)))

    @staticmethod
    def toKerasOptimizer(value):
        if kmutil.is_valid_optimizer(value):
            return value
        raise TypeError(
            "Named optimizer not supported in Keras: {} type({})".format(value, type(value)))
