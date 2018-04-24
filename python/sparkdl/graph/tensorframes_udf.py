#
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

import logging

import tensorframes as tfs  # pylint: disable=import-error

import sparkdl.graph.utils as tfx
from sparkdl.utils import jvmapi as JVMAPI

logger = logging.getLogger('sparkdl')


def makeGraphUDF(graph, udf_name, fetches, feeds_to_fields_map=None, blocked=False, register=True):
    """
    Create a Spark SQL UserDefinedFunction from a given TensorFlow Graph

    The following example creates a UDF that takes the input
    from a DataFrame column named 'image_col' and produce some random prediction.

    .. code-block:: python

        from sparkdl.graph.tensorframes_udf import makeUDF

        with IsolatedSession() as issn:
            x = tf.placeholder(tf.double, shape=[], name="input_x")
            z = tf.add(x, 3, name='z')
            makeGraphUDF(issn.graph, "my_tensorflow_udf", [z])

    Then this function can be used in a SQL query.

    .. code-block:: python

        df = spark.createDataFrame([Row(xCol=float(x)) for x in range(100)])
        df.createOrReplaceTempView("my_float_table")
        spark.sql("select my_tensorflow_udf(xCol) as zCol from my_float_table").show()

    :param graph: :py:class:`tf.Graph`, a TensorFlow Graph
    :param udf_name: str, name of the SQL UDF
    :param fetches: list, output tensors of the graph
    :param feeds_to_fields_map: a dict of str -> str,
                                The key is the name of a placeholder in the current
                                TensorFlow graph of computation.
                                The value is the name of a column in the dataframe.
                                For now, only the top-level fields in a dataframe are supported.

                                .. note:: For any placeholder that is
                                          not specified in the feed dictionary,
                                          the name of the input column is assumed to be
                                          the same as that of the placeholder.

    :param blocked: bool, if set to True, the TensorFrames will execute the function
                    over blocks/batches of rows. This should provide better performance.
                    Otherwise, the function is applied to individual rows
    :param register: bool, if set to True, the SQL UDF will be registered.
                     In this case, it will be accessible in SQL queries.
    :return: JVM function handle object
    """
    graph = tfx.validated_graph(graph)
    # pylint: disable=W0212
    # TODO: Work with TensorFlow's registered expansions
    # https://github.com/tensorflow/tensorflow/blob/v1.1.0/tensorflow/python/client/session.py#L74
    # TODO: Most part of this implementation might be better off moved to TensorFrames
    jvm_builder = JVMAPI.createTensorFramesModelBuilder()
    tfs.core._add_graph(graph, jvm_builder)

    # Obtain the fetches and their shapes
    fetch_names = [tfx.tensor_name(fetch, graph) for fetch in fetches]
    fetch_shapes = [tfx.get_shape(fetch, graph) for fetch in fetches]

    # Traverse the graph nodes and obtain all the placeholders and their shapes
    placeholder_names = []
    placeholder_shapes = []
    for node in graph.as_graph_def(add_shapes=True).node:
        # pylint: disable=len-as-condition
        # todo: refactor if not(node.input) and ...
        if len(node.input) == 0 and str(node.op) == 'Placeholder':
            tnsr_name = tfx.tensor_name(node.name, graph)
            tnsr = graph.get_tensor_by_name(tnsr_name)
            try:
                tnsr_shape = tfx.get_shape(tnsr, graph)
                placeholder_names.append(tnsr_name)
                placeholder_shapes.append(tnsr_shape)
            except ValueError:
                pass

    # Passing fetches and placeholders to TensorFrames
    jvm_builder.shape(fetch_names + placeholder_names, fetch_shapes + placeholder_shapes)
    jvm_builder.fetches(fetch_names)
    # Passing feeds to TensorFrames
    placeholder_op_names = [tfx.op_name(name, graph) for name in placeholder_names]
    # Passing the graph input to DataFrame column mapping and additional placeholder names
    tfs.core._add_inputs(jvm_builder, feeds_to_fields_map, placeholder_op_names)

    if register:
        return jvm_builder.registerUDF(udf_name, blocked)
    else:
        return jvm_builder.makeUDF(udf_name, blocked)
