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
import six

import tensorflow as tf

logger = logging.getLogger('sparkdl')

"""
When working with various pieces of TensorFlow, one is faced with
figuring out providing one of the four variants
(`tensor` OR `operation`, `name` OR `graph element`).

The various combination makes it hard to figuring out the best way.
We provide some methods to map whatever we have as input to
one of the four target variants.
"""

def validated_graph(graph):
    """
    Check if the input is a valid tf.Graph

    :param graph: tf.Graph, a TensorFlow Graph object
    """
    assert isinstance(graph, tf.Graph), 'must provide tf.Graph, but get {}'.format(type(graph))
    return graph

def get_shape(tfobj_or_name, graph):
    """
    Return the shape of the tensor as a list

    :param graph: tf.Graph, a TensorFlow Graph object
    :param tfobj_or_name: either a tf.Tensor, tf.Operation or a name to either
    """
    graph = validated_graph(graph)
    _shape = get_tensor(tfobj_or_name, graph).get_shape().as_list()
    return [-1 if x is None else x for x in _shape]

def get_op(tfobj_or_name, graph):
    """
    Get a tf.Operation object

    :param graph: tf.Graph, a TensorFlow Graph object
    :param tfobj_or_name: either a tf.Tensor, tf.Operation or a name to either
    """
    graph = validated_graph(graph)
    if isinstance(tfobj_or_name, tf.Operation):
        return tfobj_or_name
    name = tfobj_or_name
    if isinstance(tfobj_or_name, tf.Tensor):
        name = tfobj_or_name.name
    if not isinstance(name, six.string_types):
        raise TypeError('invalid op request for [type {}] {}'.format(type(name), name))
    _op_name = op_name(name, graph=None)
    op = graph.get_operation_by_name(_op_name)
    assert op is not None, \
        'cannot locate op {} in current graph'.format(_op_name)
    return op

def get_tensor(tfobj_or_name, graph):
    """
    Get a tf.Tensor object

    :param graph: tf.Graph, a TensorFlow Graph object
    :param tfobj_or_name: either a tf.Tensor, tf.Operation or a name to either
    """
    graph = validated_graph(graph)
    if isinstance(tfobj_or_name, tf.Tensor):
        return tfobj_or_name
    name = tfobj_or_name
    if isinstance(tfobj_or_name, tf.Operation):
        name = tfobj_or_name.name
    if not isinstance(name, six.string_types):
        raise TypeError('invalid tensor request for {} of {}'.format(name, type(name)))
    _tensor_name = tensor_name(name, graph=None)
    tnsr = graph.get_tensor_by_name(_tensor_name)
    assert tnsr is not None, \
        'cannot locate tensor {} in current graph'.format(_tensor_name)
    return tnsr

def tensor_name(tfobj_or_name, graph=None):
    """
    Derive tf.Tensor name from an op/tensor name.
    If the input is a name, we do not check if the tensor exist
    (as no graph parameter is passed in).

    :param tfobj_or_name: either a tf.Tensor, tf.Operation or a name to either
    """
    # If `graph` is provided, directly get the graph operation
    if graph is not None:
        return get_tensor(tfobj_or_name, graph).name
    # If `graph` is absent, check if other cases
    if isinstance(tfobj_or_name, six.string_types):
        # If input is a string, assume it is a name and infer the corresponding tensor name.
        # WARNING: this depends on TensorFlow's tensor naming convention
        name = tfobj_or_name
        name_parts = name.split(":")
        assert len(name_parts) <= 2, name_parts
        if len(name_parts) < 2:
            name += ":0"
        return name
    elif hasattr(tfobj_or_name, 'graph'):
        return get_tensor(tfobj_or_name, tfobj_or_name.graph).name
    else:
        raise TypeError('invalid tf.Tensor name query type {}'.format(type(tfobj_or_name)))

def op_name(tfobj_or_name, graph=None):
    """
    Derive tf.Operation name from an op/tensor name.
    If the input is a name, we do not check if the operation exist
    (as no graph parameter is passed in).

    :param tfobj_or_name: either a tf.Tensor, tf.Operation or a name to either
    """
    # If `graph` is provided, directly get the graph operation
    if graph is not None:
        return get_op(tfobj_or_name, graph).name
    # If `graph` is absent, check if other cases
    if isinstance(tfobj_or_name, six.string_types):
        # If input is a string, assume it is a name and infer the corresponding operation name.
        # WARNING: this depends on TensorFlow's operation naming convention
        name = tfobj_or_name
        name_parts = name.split(":")
        assert len(name_parts) <= 2, name_parts
        return name_parts[0]
    elif hasattr(tfobj_or_name, 'graph'):
        return get_op(tfobj_or_name, tfobj_or_name.graph).name
    else:
        raise TypeError('invalid tf.Operation name query type {}'.format(type(tfobj_or_name)))

def validated_output(tfobj_or_name, graph):
    """
    Validate and return the output names useable GraphFunction

    :param graph: tf.Graph, a TensorFlow Graph object
    :param tfobj_or_name: either a tf.Tensor, tf.Operation or a name to either
    """
    graph = validated_graph(graph)
    return op_name(tfobj_or_name, graph)

def validated_input(tfobj_or_name, graph):
    """
    Validate and return the input names useable GraphFunction

    :param graph: tf.Graph, a TensorFlow Graph object
    :param tfobj_or_name: either a tf.Tensor, tf.Operation or a name to either
    """
    graph = validated_graph(graph)
    name = op_name(tfobj_or_name, graph)
    op = graph.get_operation_by_name(name)
    assert 'Placeholder' == op.type, \
        ('input must be Placeholder, but get', op.type)
    return name

def strip_and_freeze_until(fetches, graph, sess=None, return_graph=False):
    """
    Create a static view of the graph by

    * Converting all variables into constants
    * Removing graph elements not reachacble to `fetches`

    :param graph: tf.Graph, the graph to be frozen
    :param fetches: list, graph elements representing the outputs of the graph
    :param return_graph: bool, if set True, return the graph function object
    :return: GraphDef, the GraphDef object with cleanup procedure applied
    """
    graph = validated_graph(graph)
    should_close_session = False
    if not sess:
        sess = tf.Session(graph=graph)
        should_close_session = True

    gdef_frozen = tf.graph_util.convert_variables_to_constants(
        sess,
        graph.as_graph_def(add_shapes=True),
        [op_name(tnsr, graph) for tnsr in fetches])

    if should_close_session:
        sess.close()

    if return_graph:
        g = tf.Graph()
        with g.as_default():
            tf.import_graph_def(gdef_frozen, name='')
        return g
    else:
        return gdef_frozen
