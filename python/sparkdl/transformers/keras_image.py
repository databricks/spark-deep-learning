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

from pyspark.ml import Transformer
from sparkdl.param import CanLoadImage, HasInputCol, HasKerasModel, HasOutputCol, HasOutputMode, \
    keyword_only
from sparkdl.transformers.keras_utils import KSessionWrap
from sparkdl.transformers.tf_image import TFImageTransformer

# pylint: disable=duplicate-code


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
        with KSessionWrap() as (sess, keras_graph):
            graph, inputTensorName, outputTensorName = self._loadTFGraph(sess=sess,
                                                                         graph=keras_graph)
            image_df = self.loadImagesInternal(dataset, self.getInputCol())
            transformer = TFImageTransformer(channelOrder='RGB', inputCol=self._loadedImageCol(),
                                             outputCol=self.getOutputCol(), graph=graph,
                                             inputTensor=inputTensorName,
                                             outputTensor=outputTensorName,
                                             outputMode=self.getOrDefault(self.outputMode))
            return transformer.transform(image_df).drop(self._loadedImageCol())
