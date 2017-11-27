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
from sparkdl.param import keyword_only, HasKerasModel, HasInputCol, HasOutputCol
from tf_tensor import TFTransformer
from sparkdl.transformers.keras_utils import KSessionWrap
from sparkdl.graph.input import TFInputGraph


class KerasTransformer(HasInputCol, HasOutputCol, HasKerasModel):
    """
    Applies the Tensorflow-backed Keras model (specified by a file name) to
    a column of arrays in a DataFrame.

    Restrictions of the current API:
      * See TFTransformer
      * Only supports Tensorflow-backed Keras models (no Theano).
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
        """ TODO(sid) actually add implementation here. """
        # Load Keras model as a TF graph from disk. Note that _loadGraph also
        # sets the _inputTensor and _outputTensor members to the names of the input/output
        # tensors of the loaded model
        with KSessionWrap() as (sess, keras_graph):
            tfGraph = self._loadTFGraph(sess=sess, graph=keras_graph)
            inputGraph = TFInputGraph.fromGraph(graph=inputGraph, sess=sess,
                                                feed_names=[self._inputTensor],
                                                fetch_names=[self._outputTensor])
        transformer = TFTransformer(graph=graph,
                                    inputMapping={self._inputTensor: self.getInputCol()},
                                    outputMapping={self._outputTensor: self.getOutputCol()})
        return transformer.transform(dataset)
