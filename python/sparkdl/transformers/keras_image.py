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
from pyspark.ml.param import Params, TypeConverters

import sparkdl.graph.utils as tfx
from sparkdl.transformers.keras_utils import KSessionWrap
from sparkdl.param import (
    keyword_only, HasInputCol, HasOutputCol,
    CanLoadImage, HasKerasModel, HasOutputMode)
from sparkdl.transformers.tf_image import TFImageTransformer


class KerasImageFileTransformer(Transformer, HasInputCol, HasOutputCol,
                                CanLoadImage, HasKerasModel, HasOutputMode):
    """
    Applies the Tensorflow-backed Keras model (specified by a file name) to
    images (specified by the URI in the inputCol column) in the DataFrame.

    Restrictions of the current API:
      * see TFImageTransformer.
      * Only supports Tensorflow-backed Keras models (no Theano).
    """
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

    def _transform(self, dataset):
        graph = self._loadTFGraph()
        image_df = self.loadImagesInternal(dataset, self.getInputCol())

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
