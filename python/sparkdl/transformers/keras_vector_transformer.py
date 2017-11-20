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

class KerasVectorTransformer(HasInputCol, HasOutputCol, HasKerasModel):
    """
    Applies the Tensorflow-backed Keras model (specified by a file name) to
    a column of vectors in a DataFrame.

    Restrictions of the current API:
      * see TFTransformer. TODO(sid): See TFTransformer?
      * Only supports Tensorflow-backed Keras models (no Theano).
    """
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, modelFile=None):
        """
        __init__(self, inputCol=None, outputCol=None, modelFile=None)
        """
        super(KerasVectorTransformer, self).__init__()
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


    def transform(self, dataframe):
        """ TODO(sid) actually add implementation here. """
        graph = self._loadTFGraph()
        transformer = TFTransformer(graph=graph)
        return transformer.transform(dataframe)


