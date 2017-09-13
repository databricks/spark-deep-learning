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
"""
Some parts are copied from pyspark.ml.param.shared and some are complementary
to pyspark.ml.param. The copy is due to some useful pyspark fns/classes being
private APIs.
"""

from functools import wraps
import six

import keras
import tensorflow as tf

from pyspark.ml.param import Param, Params, TypeConverters

from sparkdl.graph.builder import GraphFunction, IsolatedSession
import sparkdl.graph.utils as tfx
from sparkdl.graph.input import TFInputGraph, TFInputGraphBuilder

########################################################
# Copied from PySpark for backward compatibility. First in Apache Spark version 2.1.1.
########################################################

def keyword_only(func):
    """
    A decorator that forces keyword arguments in the wrapped method
    and saves actual input keyword arguments in `_input_kwargs`.

    .. note:: Should only be used to wrap a method where first arg is `self`
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if len(args) > 0:
            raise TypeError("Method %s forces keyword arguments." % func.__name__)
        self._input_kwargs = kwargs
        return func(self, **kwargs)

    return wrapper


class HasInputCol(Params):
    """
    Mixin for param inputCol: input column name.
    """

    inputCol = Param(
        Params._dummy(), "inputCol", "input column name.", typeConverter=TypeConverters.toString)

    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    def getInputCol(self):
        """
        Gets the value of inputCol or its default value.
        """
        return self.getOrDefault(self.inputCol)


class HasOutputCol(Params):
    """
    Mixin for param outputCol: output column name.
    """

    outputCol = Param(
        Params._dummy(), "outputCol", "output column name.", typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasOutputCol, self).__init__()
        self._setDefault(outputCol=self.uid + '__output')

    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)

    def getOutputCol(self):
        """
        Gets the value of outputCol or its default value.
        """
        return self.getOrDefault(self.outputCol)


########################################################
# New in sparkdl: TensorFlow Specific Parameters
########################################################

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
    def asColumnToTensorMap(value):
        if isinstance(value, dict):
            strs_pair_seq = [(k, tfx.as_op_name(v)) for k, v in value.items()]
            return sorted(strs_pair_seq)
        raise TypeError("Could not convert %s to TensorFlow Tensor" % type(value))

    @staticmethod
    def asTensorToColumnMap(value):
        if isinstance(value, dict):
            strs_pair_seq = [(tfx.as_op_name(k), v) for k, v in value.items()]
            return sorted(strs_pair_seq)
        raise TypeError("Could not convert %s to TensorFlow Tensor" % type(value))

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


class HasOutputMapping(Params):
    """
    Mixin for param outputMapping: ordered list of ('outputTensorOpName', 'outputColName') pairs
    """
    outputMapping = Param(
        Params._dummy(),
        "outputMapping",
        "Mapping output :class:`tf.Operation` names to DataFrame column names",
        typeConverter=SparkDLTypeConverters.asTensorToColumnMap)

    def setOutputMapping(self, value):
        # NOTE(phi-dbq): due to the nature of TensorFlow import modes, we can only derive the
        #                serializable TFInputGraph object once the inputMapping and outputMapping
        #                parameters are provided.
        raise NotImplementedError(
            "Please use the Transformer's constructor to assign `outputMapping` field.")

    def getOutputMapping(self):
        return self.getOrDefault(self.outputMapping)


class HasInputMapping(Params):
    """
    Mixin for param inputMapping: ordered list of ('inputColName', 'inputTensorOpName') pairs
    """
    inputMapping = Param(
        Params._dummy(),
        "inputMapping",
        "Mapping input DataFrame column names to :class:`tf.Operation` names",
        typeConverter=SparkDLTypeConverters.asColumnToTensorMap)

    def setInputMapping(self, value):
        # NOTE(phi-dbq): due to the nature of TensorFlow import modes, we can only derive the
        #                serializable TFInputGraph object once the inputMapping and outputMapping
        #                parameters are provided.
        raise NotImplementedError(
            "Please use the Transformer's constructor to assigne `inputMapping` field.")

    def getInputMapping(self):
        return self.getOrDefault(self.inputMapping)


class HasTFInputGraph(Params):
    """
    Mixin for param tfInputGraph: a serializable object derived from a TensorFlow computation graph.
    """
    tfInputGraph = Param(
        Params._dummy(),
        "tfInputGraph",
        "A serializable object derived from a TensorFlow computation graph",
        typeConverter=SparkDLTypeConverters.toTFInputGraph)

    def __init__(self):
        super(HasTFInputGraph, self).__init__()
        self._setDefault(tfInputGraph=None)

    def setTFInputGraph(self, value):
        # NOTE(phi-dbq): due to the nature of TensorFlow import modes, we can only derive the
        #                serializable TFInputGraph object once the inputMapping and outputMapping
        #                parameters are provided.
        raise NotImplementedError(
            "Please use the Transformer's constructor to assign `tfInputGraph` field.")

    def getTFInputGraph(self):
        return self.getOrDefault(self.tfInputGraph)


class HasTFHParams(Params):
    """
    Mixin for TensorFlow model hyper-parameters
    """
    tfHParams = Param(
        Params._dummy(),
        "hparams",
        "instance of :class:`tf.contrib.training.HParams`, a key-value map-like object",
        typeConverter=SparkDLTypeConverters.toTFHParams)

    def setTFHParams(self, value):
        return self._set(tfHParam=value)

    def getTFHParams(self):
        return self.getOrDefault(self.tfHParams)
