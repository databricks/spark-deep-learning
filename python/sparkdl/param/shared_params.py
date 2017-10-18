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

import tensorflow as tf

from pyspark.ml.param import Param, Params, TypeConverters

import sparkdl.utils.keras_model as kmutil


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


class KafkaParam(Params):
    kafkaParam = Param(Params._dummy(), "kafkaParam", "kafka", typeConverter=TypeConverters.identity)

    def __init__(self):
        super(KafkaParam, self).__init__()

    def setKafkaParam(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(kafkaParam=value)

    def getKafkaParam(self):
        """
        Gets the value of inputCol or its default value.
        """
        return self.getOrDefault(self.kafkaParam)


class FitParam(Params):
    fitParam = Param(Params._dummy(), "fitParam", "hyper parameter when training",
                     typeConverter=TypeConverters.identity)

    def __init__(self):
        super(FitParam, self).__init__()

    def setFitParam(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(fitParam=value)

    def getFitParam(self):
        """
        Gets the value of inputCol or its default value.
        """
        return self.getOrDefault(self.fitParam)


class MapFnParam(Params):
    mapFnParam = Param(Params._dummy(), "mapFnParam", "Tensorflow func", typeConverter=TypeConverters.identity)

    def __init__(self):
        super(MapFnParam, self).__init__()

    def setMapFnParam(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(mapFnParam=value)

    def getMapFnParam(self):
        """
        Gets the value of inputCol or its default value.
        """
        return self.getOrDefault(self.mapFnParam)


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


class HasEmbeddingSize(Params):
    """
    Mixin for param embeddingSize
    """

    embeddingSize = Param(Params._dummy(), "embeddingSize", "word embedding size",
                          typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasEmbeddingSize, self).__init__()

    def setEmbeddingSize(self, value):
        return self._set(embeddingSize=value)

    def getEmbeddingSize(self):
        return self.getOrDefault(self.embeddingSize)


class RunningMode(Params):
    """
    Mixin for param RunningMode
        * TFoS
        * Normal
    """

    runningMode = Param(Params._dummy(), "runningMode", "based on TFoS or Normal which is used to "
                                                        "hyper parameter tuning",
                        typeConverter=TypeConverters.toString)

    def __init__(self):
        super(RunningMode, self).__init__()

    def setRunningMode(self, value):
        return self._set(runningMode=value)

    def getRunningMode(self):
        return self.getOrDefault(self.runningMode)


class HasSequenceLength(Params):
    """
    Mixin for param sequenceLength
    """

    sequenceLength = Param(Params._dummy(), "sequenceLength", "sequence length",
                           typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasSequenceLength, self).__init__()

    def setSequenceLength(self, value):
        return self._set(sequenceLength=value)

    def getSequenceLength(self):
        return self.getOrDefault(self.sequenceLength)


class HasOutputCol(Params):
    """
    Mixin for param outputCol: output column name.
    """

    outputCol = Param(Params._dummy(),
                      "outputCol", "output column name.", typeConverter=TypeConverters.toString)

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


############################################
# New in sparkdl
############################################

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


class HasOutputNodeName(Params):
    # TODO: docs
    outputNodeName = Param(Params._dummy(), "outputNodeName",
                           "name of the graph element/node corresponding to the output",
                           typeConverter=TypeConverters.toString)

    def setOutputNodeName(self, value):
        return self._set(outputNodeName=value)

    def getOutputNodeName(self):
        return self.getOrDefault(self.outputNodeName)


class HasLabelCol(Params):
    """
    When training Keras image models in a supervised learning setting,
    users will provide a :py:obj:`DataFrame` column with the labels.

    .. note:: The Estimator expect this columnd to contain data directly usable for the Keras model.
              This usually means that the labels are already encoded in one-hot format.
              Please consider adding a :py:obj:`OneHotEncoder` to transform the label column.
    """
    labelCol = Param(Params._dummy(), "labelCol",
                     "name of the column storing the training data labels",
                     typeConverter=TypeConverters.toString)

    def setLabelCol(self, value):
        return self._set(labelCol=value)

    def getLabelCol(self):
        return self.getOrDefault(self.labelCol)


class HasKerasModel(Params):
    """
    This parameter allows users to provide Keras model file
    """
    # TODO: add an option to allow user to use Keras Model object
    modelFile = Param(Params._dummy(), "modelFile",
                      "HDF5 file containing the Keras model (architecture and weights)",
                      typeConverter=TypeConverters.toString)

    kerasFitParams = Param(Params._dummy(), "kerasFitParams",
                           "dict with parameters passed to Keras model fit method")

    def __init__(self):
        super(HasKerasModel, self).__init__()
        self._setDefault(kerasFitParams={'verbose': 1})

    def setModelFile(self, value):
        return self._set(modelFile=value)

    def getModelFile(self):
        return self.getOrDefault(self.modelFile)

    def setKerasFitParams(self, value):
        return self._set(kerasFitParams=value)

    def getKerasFitParams(self):
        return self.getOrDefault(self.kerasFitParams)


class HasKerasOptimizer(Params):
    # TODO: docs
    kerasOptimizer = Param(Params._dummy(), "kerasOptimizer",
                           "Name of the optimizer for training a Keras model",
                           typeConverter=SparkDLTypeConverters.toKerasOptimizer)

    def __init__(self):
        super(HasKerasOptimizer, self).__init__()
        # NOTE(phi-dbq): This is the recommended optimizer as of September 2017.
        self._setDefault(kerasOptimizer='adam')

    def setKerasOptimizer(self, value):
        return self._set(kerasOptimizer=value)

    def getKerasOptimizer(self):
        return self.getOrDefault(self.kerasOptimizer)


class HasKerasLoss(Params):
    # TODO: docs
    kerasLoss = Param(Params._dummy(), "kerasLoss",
                      "Name of the loss (objective function) for training a Keras model",
                      typeConverter=SparkDLTypeConverters.toKerasLoss)

    def seKerasLoss(self, value):
        return self._set(kerasLoss=value)

    def getKerasLoss(self):
        return self.getOrDefault(self.kerasLoss)
