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

import keras.backend as K
from keras.models import load_model

from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql.functions import udf

import sparkdl.graph.utils as tfx
from sparkdl.image import imageIO
from sparkdl.transformers.keras_utils import KSessionWrap
from sparkdl.transformers.param import (
    keyword_only, HasInputCol, HasOutputCol, SparkDLTypeConverters)
from sparkdl.transformers.tf_image import TFImageTransformer, OUTPUT_MODES
import sparkdl.transformers.utils as utils


class KerasImageFileTransformer(Transformer, HasInputCol, HasOutputCol):
    """
    Applies the Tensorflow-backed Keras model (specified by a file name) to
    images (specified by the URI in the inputCol column) in the DataFrame.

    Restrictions of the current API:
      * see TFImageTransformer.
      * Only supports Tensorflow-backed Keras models (no Theano).
    """

    modelFile = Param(Params._dummy(), "modelFile",
                      "h5py file containing the Keras model (architecture and weights)",
                      typeConverter=TypeConverters.toString)
    # TODO :add a lambda type converter e.g  callable(mylambda)
    imageLoader = Param(Params._dummy(), "imageLoader",
                        "Function containing the logic for loading and pre-processing images. " +
                        "The function should take in a URI string and return a 4-d numpy.array " +
                        "with shape (batch_size (1), height, width, num_channels).")
    outputMode = Param(Params._dummy(), "outputMode",
                       "How the output column should be formatted. 'vector' for a 1-d MLlib " +
                       "Vector of floats. 'image' to format the output to work with the image " +
                       "tools in this package.",
                       typeConverter=SparkDLTypeConverters.supportedNameConverter(OUTPUT_MODES))

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, modelFile=None, imageLoader=None,
                 outputMode="vector"):
        """
        __init__(self, inputCol=None, outputCol=None, modelFile=None, imageLoader=None,
                 outputMode="vector")
        """
        super(KerasImageFileTransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        self._inputTensor = None
        self._outputTensor = None

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, modelFile=None, imageLoader=None,
                  outputMode="vector"):
        """
        setParams(self, inputCol=None, outputCol=None, modelFile=None, imageLoader=None,
                  outputMode="vector")
        """
        kwargs = self._input_kwargs
        self._set(**kwargs)
        return self

    def setModelFile(self, value):
        return self._set(modelFile=value)

    def getModelFile(self):
        return self.getOrDefault(self.modelFile)

    def _transform(self, dataset):
        graph = self._loadTFGraph()
        image_df = self._loadImages(dataset)

        assert self._inputTensor is not None, "self._inputTensor must be set"
        assert self._outputTensor is not None, "self._outputTensor must be set"

        transformer = TFImageTransformer(inputCol=self._loadedImageCol(),
                                         outputCol=self.getOutputCol(), graph=graph,
                                         inputTensor=self._inputTensor,
                                         outputTensor=self._outputTensor,
                                         outputMode=self.getOrDefault(self.outputMode))
        return transformer.transform(image_df).drop(self._loadedImageCol())

    def _loadTFGraph(self):
        with KSessionWrap() as (sess, g):
            assert K.backend() == "tensorflow", \
                "Keras backend is not tensorflow but KerasImageTransformer only supports " + \
                "tensorflow-backed Keras models."
            with g.as_default():
                K.set_learning_phase(0)  # Testing phase
                model = load_model(self.getModelFile())
                out_op_name = tfx.op_name(g, model.output)
                self._inputTensor = model.input.name
                self._outputTensor = model.output.name
                return tfx.strip_and_freeze_until([out_op_name], g, sess, return_graph=True)

    def _loadedImageCol(self):
        return "__sdl_img"

    def _loadImages(self, dataset):
        """
        Load image files specified in dataset as image format specified in sparkdl.image.imageIO.
        """
        # plan 1: udf(loader() + convert from np.array to imageSchema) -> call TFImageTransformer
        # plan 2: udf(loader()) ... we don't support np.array as a dataframe column type...
        loader = self.getOrDefault(self.imageLoader)

        def load(uri):
            img = loader(uri)
            return imageIO.imageArrayToStruct(img)
        load_udf = udf(load, imageIO.imageSchema)
        return dataset.withColumn(self._loadedImageCol(), load_udf(dataset[self.getInputCol()]))
