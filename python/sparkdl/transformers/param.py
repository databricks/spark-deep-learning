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

import keras
import tensorflow as tf

from pyspark.ml.param import Param, Params, TypeConverters


# From pyspark

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

    inputCol = Param(Params._dummy(), "inputCol", "input column name.", typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasInputCol, self).__init__()

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

    outputCol = Param(Params._dummy(), "outputCol", "output column name.", typeConverter=TypeConverters.toString)

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


# New in sparkdl

class SparkDLTypeConverters(object):

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
    def toTFGraph(value):
        # TODO: we may want to support tf.GraphDef in the future instead of tf.Graph since user
        # is less likely to mess up using GraphDef vs Graph (e.g. constants vs variables).
        if isinstance(value, tf.Graph):
            return value
        else:
            raise TypeError("Could not convert %s to tensorflow.Graph type" % type(value))

    @staticmethod
    def supportedNameConverter(supportedList):
        def converter(value):
            if value in supportedList:
                return value
            else:
                raise TypeError("%s %s is not in the supported list." % type(value), str(value))
