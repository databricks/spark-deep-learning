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

import numpy as np
import tensorflow as tf

from pyspark.sql import Row
from pyspark.sql.types import *

import tensorframes as tfs

import sparkdl.graph.utils as tfx
from sparkdl.graph.input import TFInputGraph
from sparkdl.transformers.tf_tensor import TFTransformer

from ..tests import SparkDLTestCase


class TFTransformerTests(SparkDLTestCase):
    def test_graph_array_types(self):
        test_dtypes = [(tf.int32, ArrayType(IntegerType()), np.int32),
                       (tf.int64, ArrayType(LongType()), np.int64),
                       (tf.float32, ArrayType(FloatType()), np.float32),
                       (tf.float64, ArrayType(DoubleType()), np.float64)]
        for tf_dtype, spark_dtype, np_type in test_dtypes:
            transformer = _build_transformer(lambda session:
                                             TFInputGraph.fromGraph(session.graph, session,
                                                                    [_tensor_input_name],
                                                                    [_tensor_output_name]),
                                             tf_dtype)
            gin = transformer.getTFInputGraph()
            local_features = _build_local_features(np_type)
            expected = _get_expected_result(gin, local_features)
            schema = StructType([StructField('inputCol', spark_dtype)])
            dataset = self.session.createDataFrame(local_features, schema)
            _check_transformer_output(transformer, dataset, expected)

# The name of the input tensor
_tensor_input_name = "input_tensor"
# The name of the output tensor (scalar)
_tensor_output_name = "output_tensor"
# The size of the input tensor
_tensor_size = 3
# Input mapping for the Transformer
_input_mapping = {'inputCol': tfx.tensor_name(_tensor_input_name)}
# Output mapping for the Transformer
_output_mapping = {tfx.tensor_name(_tensor_output_name): 'outputCol'}
# Numerical threshold
_all_close_tolerance = 1e-5


def _build_transformer(gin_function, tf_dtype):
    """
    Makes a session and a default graph, loads the simple graph into it, and then calls
    gin_function(session) to build the :py:obj:`TFInputGraph` object.
    Return the :py:obj:`TFTransformer` created from it.
    """
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess, graph.as_default():
        _build_graph(sess, tf_dtype)
        gin = gin_function(sess)

    return TFTransformer(tfInputGraph=gin,
                         inputMapping=_input_mapping,
                         outputMapping=_output_mapping)


def _build_graph(sess, tf_dtype):
    """
    Given a session (implicitly), adds nodes of computations

    It takes a vector input, with `_tensor_size` columns and returns an float64 scalar.
    """
    x = tf.placeholder(tf_dtype, shape=[None, _tensor_size], name=_tensor_input_name)
    _ = tf.reduce_max(x, axis=1, name=_tensor_output_name)


def _build_local_features(np_dtype):
    """
    Build numpy array (i.e. local) features.
    """
    # Build local features and DataFrame from it
    local_features = []
    np.random.seed(997)
    for idx in range(100):
        _dict = {'idx': idx}
        for colname, _ in _input_mapping.items():
            colvalue = np.random.randn(_tensor_size) * 100
            _dict[colname] = colvalue.astype(np_dtype).tolist()

        local_features.append(Row(**_dict))

    return local_features


def _get_expected_result(gin, local_features):
    """
    Running the graph in the :py:obj:`TFInputGraph` object and compute the expected results.
    :param: gin, a :py:obj:`TFInputGraph`
    :return: expected results in NumPy array
    """
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess, graph.as_default():
        # Build test graph and transformers from here
        tf.import_graph_def(gin.graph_def, name='')

        # Build the results
        _results = []
        for row in local_features:
            fetches = [tfx.get_tensor(tnsr_name, graph)
                       for tnsr_name, _ in _output_mapping.items()]
            feed_dict = {}
            for colname, tnsr_name in _input_mapping.items():
                tnsr = tfx.get_tensor(tnsr_name, graph)
                feed_dict[tnsr] = np.array(row[colname])[np.newaxis, :]

            curr_res = sess.run(fetches, feed_dict=feed_dict)
            _results.append(np.ravel(curr_res))

        expected = np.hstack(_results)

    return expected


def _check_transformer_output(transformer, dataset, expected):
    """
    Given a transformer and a spark dataset, check if the transformer
    produces the expected results.
    """
    analyzed_df = tfs.analyze(dataset)
    out_df = transformer.transform(analyzed_df)

    # Collect transformed values
    out_colnames = list(_output_mapping.values())
    _results = []
    for row in out_df.select(out_colnames).collect():
        curr_res = [row[colname] for colname in out_colnames]
        _results.append(np.ravel(curr_res))
    out_tgt = np.hstack(_results)

    _err_msg = 'not close => shape {} != {}, max_diff {} > {}'
    max_diff = np.max(np.abs(expected - out_tgt))
    err_msg = _err_msg.format(expected.shape, out_tgt.shape,
                              max_diff, _all_close_tolerance)
    assert np.allclose(expected, out_tgt, atol=_all_close_tolerance), err_msg
