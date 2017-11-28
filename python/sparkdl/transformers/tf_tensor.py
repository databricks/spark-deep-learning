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
from tensorflow.python.tools import optimize_for_inference_lib as infr_opt
import tensorframes as tfs

from pyspark.ml import Transformer

from sparkdl.graph.builder import GraphFunction
import sparkdl.graph.utils as tfx
from sparkdl.param import (keyword_only, HasInputMapping, HasOutputMapping,
                           HasTFInputGraph, HasTFHParams)

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

    def _addCastOps(self, user_graph_fn):
        """
        Given a GraphFunction object corresponding to a user-specified graph G, creates a copy G'
        of G with ops injected before each input node. The injected ops allow the input nodes of G'
        to accept tf.float64 input fed from Spark, casting float64 input into the datatype
        requested by each input node.

        :return: GraphFunction wrapping the copied, modified graph.
        """
        # Load user-specified graph into memory
        user_graph = tf.Graph()
        with user_graph.as_default():
            tf.import_graph_def(user_graph_fn.graph_def, name="")

        # Build a subgraph containing our injected ops
        # TODO: If all input tensors are of type float64, we could just do nothing here
        injected_op_subgraph = tf.Graph()
        # Arrays containing the names of input/output tensors of our injected-op subgraph
        output_names = []
        input_names = []
        with injected_op_subgraph.as_default():
            # Iterate through the input tensors of the original graph
            for _, orig_tensor_name in self.getInputMapping():
                orig_tensor = tfx.get_tensor(orig_tensor_name, user_graph)
                # Create placeholder with same shape as original input tensor, but that accepts
                # float64 input from Spark.
                spark_placeholder = tf.placeholder(tf.float64, shape=orig_tensor.shape,
                                             name=tfx.op_name(orig_tensor_name))
                # If the original tensor was of type float64, just pass through the Spark input
                if orig_tensor.dtype == tf.float64:
                    output_tensor = tf.identity(spark_placeholder)
                # Otherwise, cast the Spark input to the datatype of the original tensor
                else:
                    output_tensor = tf.cast(spark_placeholder, dtype=orig_tensor.dtype)
                input_names.append(spark_placeholder.name)
                output_names.append(output_tensor.name)

        # Concatenate our injected-op subgraph with the original graph and return the result.
        injected_op_subgraph_fn = GraphFunction(graph_def=injected_op_subgraph.as_graph_def(),
                                             input_names=input_names, output_names=output_names)
        return GraphFunction.fromList([("sparkdl_ops", injected_op_subgraph_fn),
                                      ("user_graph", user_graph_fn)])

    def _optimize_for_inference(self):
        """ Optimize the graph for inference """
        gin = self.getTFInputGraph()
        input_mapping = self.getInputMapping()
        output_mapping = self.getOutputMapping()
        input_node_names = [tfx.op_name(tnsr_name) for _, tnsr_name in input_mapping]
        output_node_names = [tfx.op_name(tnsr_name) for tnsr_name, _ in output_mapping]

        # Build GraphFunction wrapping the model graph
        graph_fn = GraphFunction(graph_def=gin.graph_def, input_names=input_node_names,
                                 output_names=output_node_names)
        # Inject cast ops to convert float64 input fed from Spark into the datatypes of the
        # Graph's input nodes.
        graph_fn_with_casts = self._addCastOps(graph_fn)

        # Optimize for inference, replacing placeholders with float64
        # NOTE(phi-dbq): Spark DataFrame assumes float64 as default floating point type
        input_names = [name for name in graph_fn_with_casts.input_names]
        opt_gdef = infr_opt.optimize_for_inference(graph_fn_with_casts.graph_def,
                                                   input_names,
                                                   graph_fn_with_casts.output_names,
                                                   tf.float64.as_datatype_enum)
        return opt_gdef

    def _transform(self, dataset):
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
            feed_dict = {tfx.op_name(tnsr_name) : col_name for col_name, tnsr_name in input_mapping}
            fetches = [tfx.get_tensor(tnsr_op_name, graph) for tnsr_op_name in out_tnsr_op_names]
            out_df = tfs.map_blocks(fetches, analyzed_df, feed_dict=feed_dict)

            # We still have to rename output columns
            for tnsr_name, new_colname in output_mapping:
                old_colname = tfx.op_name(tnsr_name, graph)
                if old_colname != new_colname:
                    out_df = out_df.withColumnRenamed(old_colname, new_colname)

        return out_df
