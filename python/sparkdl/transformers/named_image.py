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
from pyspark.sql.functions import udf
from pyspark.sql.types import (ArrayType, FloatType, StringType, StructField, StructType)

import sparkdl.graph.utils as tfx
from sparkdl.image.imageIO import resizeImage
import sparkdl.transformers.keras_applications as keras_apps
from sparkdl.param import (
    keyword_only, HasInputCol, HasOutputCol, SparkDLTypeConverters)
from sparkdl.transformers.tf_image import TFImageTransformer


SUPPORTED_MODELS = ["InceptionV3", "Xception", "ResNet50"]


class DeepImagePredictor(Transformer, HasInputCol, HasOutputCol):
    """
    Applies the model specified by its popular name to the image column in DataFrame.
    The input image column should be 3-channel SpImage.
    The output is a MLlib Vector.
    """

    modelName = Param(Params._dummy(), "modelName", "A deep learning model name",
                      typeConverter=SparkDLTypeConverters.supportedNameConverter(SUPPORTED_MODELS))
    decodePredictions = Param(Params._dummy(), "decodePredictions",
                              "If true, output predictions in the (class, description, probability) format",
                              typeConverter=TypeConverters.toBoolean)
    topK = Param(Params._dummy(), "topK", "How many classes to return if decodePredictions is True",
                 typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, modelName=None, decodePredictions=False,
                 topK=5):
        """
        __init__(self, inputCol=None, outputCol=None, modelName=None, decodePredictions=False,
                 topK=5)
        """
        super(DeepImagePredictor, self).__init__()
        self._setDefault(decodePredictions=False)
        self._setDefault(topK=5)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, modelName=None, decodePredictions=False,
                  topK=5):
        """
        setParams(self, inputCol=None, outputCol=None, modelName=None, decodePredictions=False,
                  topK=5)
        """
        kwargs = self._input_kwargs
        self._set(**kwargs)
        return self

    def setModelName(self, value):
        return self._set(modelName=value)

    def getModelName(self):
        return self.getOrDefault(self.modelName)

    def _transform(self, dataset):
        transformer = _NamedImageTransformer(inputCol=self.getInputCol(),
                                             outputCol=self._getIntermediateOutputCol(),
                                             modelName=self.getModelName(), featurize=False)
        transformed = transformer.transform(dataset)
        if self.getOrDefault(self.decodePredictions):
            return self._decodeOutputAsPredictions(transformed)
        else:
            return transformed.withColumnRenamed(
                self._getIntermediateOutputCol(), self.getOutputCol())

    def _decodeOutputAsPredictions(self, df):
        # If we start having different weights than imagenet, we'll need to
        # move this logic to individual model building in NamedImageTransformer.
        # Also, we could put the computation directly in the main computation
        # graph or use a scala UDF for potentially better performance.
        topK = self.getOrDefault(self.topK)
        def decode(predictions):
            pred_arr = np.expand_dims(np.array(predictions), axis=0)
            decoded = decode_predictions(pred_arr, top=topK)[0]
            # convert numpy dtypes to python native types
            return [(t[0], t[1], t[2].item()) for t in decoded]
        decodedSchema = ArrayType(
            StructType([StructField("class", StringType(), False),
                        StructField("description", StringType(), False),
                        StructField("probability", FloatType(), False)]))
        decodeUDF = udf(decode, decodedSchema)
        interim_output = self._getIntermediateOutputCol()
        return (
            df.withColumn(self.getOutputCol(), decodeUDF(df[interim_output]))
              .drop(interim_output)
        )

    def _getIntermediateOutputCol(self):
        return "__tmp_" + self.getOutputCol()


# TODO: give an option to take off multiple layers so it can be used in tuning
#       (could be the name of the layer or int for how many to take off).
class DeepImageFeaturizer(Transformer, HasInputCol, HasOutputCol):
    """
    Applies the model specified by its popular name, with its prediction layer(s) chopped off,
    to the image column in DataFrame. The output is a MLlib Vector so that DeepImageFeaturizer
    can be used in a MLlib Pipeline.
    The input image column should be 3-channel SpImage.
    """

    modelName = Param(Params._dummy(), "modelName", "A deep learning model name",
                      typeConverter=SparkDLTypeConverters.supportedNameConverter(SUPPORTED_MODELS))

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, modelName=None):
        """
        __init__(self, inputCol=None, outputCol=None, modelName=None)
        """
        super(DeepImageFeaturizer, self).__init__()
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
        transformer = _NamedImageTransformer(inputCol=self.getInputCol(),
                                             outputCol=self.getOutputCol(),
                                             modelName=self.getModelName(), featurize=True)
        return transformer.transform(dataset)


class _NamedImageTransformer(Transformer, HasInputCol, HasOutputCol):
    """
    For internal use only. NamedImagePredictor and NamedImageFeaturizer are the recommended classes
    to use.

    Applies the model specified by its popular name to the image column in DataFrame. There are
    two output modes: predictions or the featurization from the model. In either case the output
    is a MLlib Vector.
    """

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
        super(_NamedImageTransformer, self).__init__()
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
        resizedCol = "__sdl_imagesResized"
        tfTransformer = TFImageTransformer(inputCol=resizedCol,
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
