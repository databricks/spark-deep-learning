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
import py4j

from pyspark import SparkContext
from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.util import JavaMLReadable, JavaMLWritable, JavaMLReader
from pyspark.ml.wrapper import JavaTransformer
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType

import sparkdl.graph.utils as tfx
from sparkdl.image.imageIO import createResizeImageUDF
import sparkdl.transformers.keras_applications as keras_apps
from sparkdl.param import keyword_only, HasInputCol, HasOutputCol, SparkDLTypeConverters
from sparkdl.transformers.tf_image import TFImageTransformer


# If this list of supported models is expanded, update the list in the README
# section for DeepImageFeaturizer.
SUPPORTED_MODELS = ["InceptionV3", "Xception", "ResNet50", "VGG16", "VGG19"]


class DeepImagePredictor(Transformer, HasInputCol, HasOutputCol):
    """
    Applies the model specified by its popular name to the image column in DataFrame.
    The input image column should be 3-channel SpImage.
    The output is a MLlib Vector.
    """

    modelName = Param(
        Params._dummy(),
        "modelName",
        "A deep learning model name",
        typeConverter=SparkDLTypeConverters.buildSupportedItemConverter(SUPPORTED_MODELS))
    decodePredictions = Param(
        Params._dummy(),
        "decodePredictions",
        "If true, output predictions in the (class, description, probability) format",
        typeConverter=TypeConverters.toBoolean)
    topK = Param(
        Params._dummy(),
        "topK",
        "How many classes to return if decodePredictions is True",
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
        transformer = _NamedImageTransformer(
            inputCol=self.getInputCol(),
            outputCol=self._getIntermediateOutputCol(),
            modelName=self.getModelName(),
            featurize=False)
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
            StructType([
                StructField("class", StringType(), False),
                StructField("description", StringType(), False),
                StructField("probability", FloatType(), False)
            ]))
        decodeUDF = udf(decode, decodedSchema)
        interim_output = self._getIntermediateOutputCol()
        return df \
            .withColumn(self.getOutputCol(), decodeUDF(df[interim_output])) \
            .drop(interim_output)

    def _getIntermediateOutputCol(self):
        return "__tmp_" + self.getOutputCol()


def _getScaleHintList():
    featurizer = SparkContext.getOrCreate()._jvm.com.databricks.sparkdl.DeepImageFeaturizer
    if isinstance(featurizer, py4j.java_gateway.JavaPackage):
        # do not see DeepImageFeaturizer, possibly running without spark
        # instead of failing return empty list
        return []
    return dict(featurizer.scaleHintsJava()).keys()


class _LazyScaleHintConverter:  # pylint: disable=too-few-public-methods
    _sizeHintConverter = None

    def __call__(self, value):
        if not self._sizeHintConverter:
            self._sizeHintConverter = SparkDLTypeConverters.buildSupportedItemConverter(
                _getScaleHintList())
        return self._sizeHintConverter(value)


class _DeepImageFeaturizerReader(JavaMLReader):

    def __init__(self, clazz):  # pylint: disable=useless-super-delegation
        super(_DeepImageFeaturizerReader, self).__init__(clazz)

    @classmethod
    def _java_loader_class(cls, clazz):
        return "com.databricks.sparkdl.DeepImageFeaturizer"


class DeepImageFeaturizer(JavaTransformer, JavaMLReadable, JavaMLWritable):
    """
    Applies the model specified by its popular name, with its prediction layer(s) chopped off,
    to the image column in DataFrame. The output is a MLlib Vector so that DeepImageFeaturizer
    can be used in a MLlib Pipeline.
    The input image column should be ImageSchema.
    """
    # NOTE: We are not inheriting from HasInput/HasOutput to mirror the scala side. It also helps
    # us to avoid issue with serialization/deserialization - default values based on uid do not
    # always get set to correct value on the python side after deserialization. Default values do
    # not get reset to the jvm side value unless they param value is not set.
    # See pyspark.ml.wrapper.JavaParams._transfer_params_from_java
    inputCol = Param(
        Params._dummy(),
        "inputCol",
        "input column name.",
        typeConverter=TypeConverters.toString)
    outputCol = Param(
        Params._dummy(),
        "outputCol",
        "output column name.",
        typeConverter=TypeConverters.toString)
    modelName = Param(
        Params._dummy(),
        "modelName",
        "A deep learning model name",
        typeConverter=SparkDLTypeConverters.buildSupportedItemConverter(
            SUPPORTED_MODELS))
    scaleHint = Param(
        Params._dummy(),
        "scaleHint",
        "Hint which algorhitm to use for image "
        "resizing",
        typeConverter=_LazyScaleHintConverter())

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, modelName=None,
                 scaleHint="SCALE_AREA_AVERAGING"):
        """
        __init__(self, inputCol=None, outputCol=None, modelName=None,
                 scaleHint="SCALE_AREA_AVERAGING")
        """
        super(DeepImageFeaturizer, self).__init__()
        self._java_obj = self._new_java_obj("com.databricks.sparkdl.DeepImageFeaturizer", self.uid)
        self._setDefault(scaleHint="SCALE_AREA_AVERAGING")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, modelName=None,
                  scaleHint="SCALE_AREA_AVERAGING"):
        """
        setParams(self, inputCol=None, outputCol=None, modelName=None,
                  scaleHint="SCALE_AREA_AVERAGING")
        """
        kwargs = self._input_kwargs
        self._set(**kwargs)
        self._transfer_params_to_java()
        return self

    def setInputCol(self, value):
        return self._set(inputCol=value)

    def getInputCol(self):
        return self.getOrDefault(self.inputCol)

    def setOutputCol(self, value):
        return self._set(outputCol=value)

    def getOutputCol(self):
        return self.getOrDefault(self.outputCol)

    def setModelName(self, value):
        return self._set(modelName=value)

    def getModelName(self):
        return self.getOrDefault(self.modelName)

    def setScaleHint(self, value):
        return self._set(scaleHint=value)

    def getScaleHint(self):
        return self.getOrDefault(self.scaleHint)

    @classmethod
    def read(cls):
        return _DeepImageFeaturizerReader(cls)

    @classmethod
    def _from_java(cls, java_stage):
        """
        Given a Java object, create and return a Python wrapper of it.
        Used for ML persistence.
        """
        res = DeepImageFeaturizer()
        res._java_obj = java_stage
        res._resetUid(java_stage.uid())
        res._transfer_params_from_java()
        return res


class _NamedImageTransformer(Transformer, HasInputCol, HasOutputCol):
    """
    For internal use only. NamedImagePredictor and NamedImageFeaturizer are the recommended classes
    to use.

    Applies the model specified by its popular name to the image column in DataFrame. There are
    two output modes: predictions or the featurization from the model. In either case the output
    is a MLlib Vector.
    """

    modelName = Param(
        Params._dummy(),
        "modelName",
        "A deep learning model name",
        typeConverter=SparkDLTypeConverters.buildSupportedItemConverter(SUPPORTED_MODELS))
    featurize = Param(
        Params._dummy(),
        "featurize",
        "If true, output features, else, output predictions. Either way the output is a vector.",
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
        tfTransformer = TFImageTransformer(
            channelOrder='BGR',
            inputCol=resizedCol,
            outputCol=self.getOutputCol(),
            graph=modelGraphSpec["graph"],
            inputTensor=modelGraphSpec["inputTensorName"],
            outputTensor=modelGraphSpec["outputTensorName"],
            outputMode=modelGraphSpec["outputMode"])
        resizeUdf = createResizeImageUDF(modelGraphSpec["inputTensorSize"])
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
