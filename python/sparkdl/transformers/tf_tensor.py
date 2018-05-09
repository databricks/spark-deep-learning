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
from __future__ import absolute_import, division, print_function

import logging
import tensorflow as tf
# pylint: disable=no-name-in-module
from tensorflow.python.tools import optimize_for_inference_lib as infr_opt
# pylint: enable=no-name-in-module
import tensorframes as tfs  # pylint: disable=import-error

from pyspark.ml import Transformer
from pyspark.sql.types import DoubleType

import sparkdl.graph.utils as tfx
from sparkdl.param import keyword_only, HasInputMapping, HasOutputMapping, HasTFInputGraph, \
    HasTFHParams

__all__ = ['TFTransformer']

logger = logging.getLogger('sparkdl')


class TFTransformer(Transformer, HasTFInputGraph, HasTFHParams, HasInputMapping, HasOutputMapping):
    """
    Applies the TensorFlow graph to the array column in DataFrame.

    Restrictions of the current API:

    We assume that
    - All the inputs of the graphs have a "minibatch" dimension (i.e. an unknown leading
      dimension) in the tensor shapes.
    - Input DataFrame has an array column where all elements have the same length
    - The transformer is expected to work on blocks of data at the same time.
    """

    @keyword_only
    def __init__(self, tfInputGraph=None, inputMapping=None, outputMapping=None, tfHParms=None):
        """
        __init__(self, tfInputGraph=None, inputMapping=None, outputMapping=None, tfHParms=None)
        """
        super(TFTransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, tfInputGraph=None, inputMapping=None, outputMapping=None, tfHParms=None):
        """
        setParams(self, tfInputGraph=None, inputMapping=None, outputMapping=None, tfHParms=None)
        """
        super(TFTransformer, self).__init__()
        kwargs = self._input_kwargs
        # Further conanonicalization, e.g. converting dict to sorted str pairs happens here
        return self._set(**kwargs)

    def _get_placeholder_types(self, user_graph_def):
        """ Returns a list of placeholder type enums for the input nodes """
        user_graph = tf.Graph()
        with user_graph.as_default():
            # Load user-specified graph into memory, then get the data type of each input node
            tf.import_graph_def(user_graph_def, name="")
            res = []
            for _, tensor_name in self.getInputMapping():
                placeholder_type = tfx.get_tensor(tensor_name, user_graph).dtype.as_datatype_enum
                res.append(placeholder_type)
        return res

    def _optimize_for_inference(self):
        graph_def = self.getTFInputGraph().graph_def
        # Get data types of input placeholders
        placeholder_types = self._get_placeholder_types(graph_def)
        # Strip away graph nodes not used in computing the tensors with the specified output names
        input_names = [tfx.op_name(tnsr_name) for _, tnsr_name in self.getInputMapping()]
        output_names = [tfx.op_name(tnsr_name) for tnsr_name, _ in self.getOutputMapping()]
        return infr_opt.optimize_for_inference(graph_def,
                                               input_names,
                                               output_names,
                                               placeholder_types)

    def _transform(self, dataset):
        if any([field.dataType == DoubleType() for field in dataset.schema]):
            logger.warning("Detected DoubleType columns in dataframe passed to transform(). In "
                           "Deep Learning Pipelines 1.0 and above, DoubleType columns can only be "
                           "fed to input tensors of type tf.float64. To feed dataframe data to "
                           "tensors of other types (e.g. tf.float32, tf.int32, tf.int64), use the "
                           "corresponding Spark SQL data types (FloatType, IntegerType, LongType).")

        graph_def = self._optimize_for_inference()
        input_mapping = self.getInputMapping()
        output_mapping = self.getOutputMapping()

        graph = tf.Graph()
        with tf.Session(graph=graph):
            analyzed_df = tfs.analyze(dataset)
            out_tnsr_op_names = [tfx.op_name(tnsr_name) for tnsr_name, _ in output_mapping]
            # Load graph
            tf.import_graph_def(graph_def=graph_def, name='', return_elements=out_tnsr_op_names)
            # Feed dict maps from placeholder name to DF column name
            feed_dict = {tfx.op_name(tnsr_name): col_name for col_name, tnsr_name in input_mapping}
            fetches = [tfx.get_tensor(tnsr_name, graph) for tnsr_name in out_tnsr_op_names]
            out_df = tfs.map_blocks(fetches, analyzed_df, feed_dict=feed_dict)
            # We still have to rename output columns
            for tnsr_name, new_colname in output_mapping:
                old_colname = tfx.op_name(tnsr_name, graph)
                if old_colname != new_colname:
                    out_df = out_df.withColumnRenamed(old_colname, new_colname)

        return out_df
