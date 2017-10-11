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

from keras.applications.imagenet_utils import decode_predictions
import numpy as np

from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters

import sparkdl.graph.utils as tfx
from sparkdl.image.imageIO import resizeImage
import sparkdl.transformers.keras_applications as keras_apps
from sparkdl.param import (
    keyword_only, HasInputCol, HasOutputCol, SparkDLTypeConverters)
from sparkdl.transformers.tf_text import TFTextTransformer

SUPPORTED_MODELS = ["CNN", "LSTM"]


class DeepTextFeaturizer(Transformer, HasInputCol, HasOutputCol):
    """
    todo
    """
    modelName = Param(Params._dummy(), "modelName", "A deep learning model name")

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, modelName=None):
        """
        __init__(self, inputCol=None, outputCol=None, modelName=None)
        """
        super(DeepTextFeaturizer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, modelName=None):
        """
        setParams(self, inputCol=None, outputCol=None, modelName=None)
        """
        kwargs = self._input_kwargs
        self._set(**kwargs)
        return self

    def setModelName(self, value):
        return self._set(modelName=value)

    def getModelName(self):
        return self.getOrDefault(self.modelName)

    def _transform(self, dataset):
        transformer = _NamedTextTransformer(inputCol=self.getInputCol(),
                                            outputCol=self.getOutputCol(),
                                            modelName=self.getModelName(), featurize=True)
        return transformer.transform(dataset)


class _NamedTextTransformer(Transformer, HasInputCol, HasOutputCol):
    modelName = Param(Params._dummy(), "modelName", "A deep learning model name",
                      typeConverter=SparkDLTypeConverters.supportedNameConverter(SUPPORTED_MODELS))
    featurize = Param(Params._dummy(), "featurize",
                      "If true, output features. If false, output predictions. Either way the output is a vector.",
                      typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, modelName=None, featurize=False):
        """
        __init__(self, inputCol=None, outputCol=None, modelName=None, featurize=False)
        """
        super(_NamedTextTransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        self._inputTensorName = None
        self._outputTensorName = None
        self._outputMode = None

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, modelName=None, featurize=False):
        """
        setParams(self, inputCol=None, outputCol=None, modelName=None, featurize=False)
        """
        kwargs = self._input_kwargs
        self._set(**kwargs)
        return self

    def setModelName(self, value):
        return self._set(modelName=value)

    def getModelName(self):
        return self.getOrDefault(self.modelName)

    def setFeaturize(self, value):
        return self._set(featurize=value)

    def getFeaturize(self):
        return self.getOrDefault(self.featurize)

    def _transform(self, dataset):
        modelGraphSpec = _buildTFGraphForName(self.getModelName(), self.getFeaturize())
        inputCol = self.getInputCol()
        resizedCol = "__sdl_textResized"
        tfTransformer = TFTextTransformer(inputCol=resizedCol,
                                          outputCol=self.getOutputCol(),
                                          graph=modelGraphSpec["graph"],
                                          inputTensor=modelGraphSpec["inputTensorName"],
                                          outputTensor=modelGraphSpec["outputTensorName"],
                                          outputMode=modelGraphSpec["outputMode"])
        resizeUdf = resizeImage(modelGraphSpec["inputTensorSize"])
        result = tfTransformer.transform(dataset.withColumn(resizedCol, resizeUdf(inputCol)))
        return result.drop(resizedCol)


def _buildTFGraphForName(name, featurize):
    """
    Currently only supports pre-trained models from the Keras applications module.
    """
    modelData = keras_apps.getKerasApplicationModel(name).getModelData(featurize)
    sess = modelData["session"]
    outputTensorName = modelData["outputTensorName"]
    graph = tfx.strip_and_freeze_until([outputTensorName], sess.graph, sess, return_graph=True)
    modelData["graph"] = graph

    return modelData
