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
from sparkdl.graph.input import TFInputGraph
from sparkdl.param import HasInputCol, HasKerasModel, HasOutputCol, keyword_only
from sparkdl.transformers.keras_utils import KSessionWrap
from .tf_tensor import TFTransformer

# pylint: disable=duplicate-code


class KerasTransformer(Transformer, HasInputCol, HasOutputCol, HasKerasModel):
    """
    Applies a Tensorflow-backed Keras model (specified by a file name) to
    a column of arrays (where each array corresponds to a Tensor) in a DataFrame.
    Produces an output column of arrays.

    Restrictions of the current API:
      * See TFTransformer
      * Only supports Keras models with a single input tensor & a single output tensor, where
        the input & output tensors must have at most 2 dimensions.
      * Only supports Tensorflow-backed Keras models (no Theano or CNTK).
    """
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, modelFile=None):
        """
        __init__(self, inputCol=None, outputCol=None, modelFile=None)
        """
        super(KerasTransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, modelFile=None):
        """
        setParams(self, inputCol=None, outputCol=None, modelFile=None)
        """
        kwargs = self._input_kwargs
        self._set(**kwargs)
        return self

    def _transform(self, dataset):
        with KSessionWrap() as (sess, keras_graph):
            tfGraph, inputTensorName, outputTensorName = self._loadTFGraph(sess=sess,
                                                                           graph=keras_graph)
            inputGraph = TFInputGraph.fromGraph(graph=tfGraph, sess=sess,
                                                feed_names=[inputTensorName],
                                                fetch_names=[outputTensorName])
        # Create TFTransformer & use it to apply the loaded Keras model graph to our dataset
        transformer = TFTransformer(tfInputGraph=inputGraph,
                                    inputMapping={self.getInputCol(): inputTensorName},
                                    outputMapping={outputTensorName: self.getOutputCol()})
        return transformer.transform(dataset)
