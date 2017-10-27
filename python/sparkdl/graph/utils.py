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
import webbrowser
from tempfile import NamedTemporaryFile

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

def get_shape(graph, tfobj_or_name):
    """
    Return the shape of the tensor as a list

    :param graph: tf.Graph, a TensorFlow Graph object
    :param tfobj_or_name: either a tf.Tensor, tf.Operation or a name to either
    """
    graph = validated_graph(graph)
    _shape = get_tensor(graph, tfobj_or_name).get_shape().as_list()
    return [-1 if x is None else x for x in _shape]

def get_op(graph, tfobj_or_name):
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
        raise TypeError('invalid op request for {} of {}'.format(name, type(name)))
    _op_name = as_op_name(name)
    op = graph.get_operation_by_name(_op_name)
    assert op is not None, \
        'cannot locate op {} in current graph'.format(_op_name)
    return op

def get_tensor(graph, tfobj_or_name):
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
    _tensor_name = as_tensor_name(name)
    tnsr = graph.get_tensor_by_name(_tensor_name)
    assert tnsr is not None, \
        'cannot locate tensor {} in current graph'.format(_tensor_name)
    return tnsr

def as_tensor_name(name):
    """
    Derive tf.Tensor name from an op/tensor name.
    We do not check if the tensor exist (as no graph parameter is passed in).

    :param name: op name or tensor name
    """
    assert isinstance(name, six.string_types)
    name_parts = name.split(":")
    assert len(name_parts) <= 2, name_parts
    if len(name_parts) < 2:
        name += ":0"
    return name

def as_op_name(name):
    """
    Derive tf.Operation name from an op/tensor name
    We do not check if the operation exist (as no graph parameter is passed in).

    :param name: op name or tensor name
    """
    assert isinstance(name, six.string_types)
    name_parts = name.split(":")
    assert len(name_parts) <= 2, name_parts
    return name_parts[0]

def op_name(graph, tfobj_or_name):
    """
    Get the name of a tf.Operation

    :param graph: tf.Graph, a TensorFlow Graph object
    :param tfobj_or_name: either a tf.Tensor, tf.Operation or a name to either
    """
    graph = validated_graph(graph)
    return get_op(graph, tfobj_or_name).name

def tensor_name(graph, tfobj_or_name):
    """
    Get the name of a tf.Tensor

    :param graph: tf.Graph, a TensorFlow Graph object
    :param tfobj_or_name: either a tf.Tensor, tf.Operation or a name to either
    """
    graph = validated_graph(graph)
    return get_tensor(graph, tfobj_or_name).name

def validated_output(graph, tfobj_or_name):
    """
    Validate and return the output names useable GraphFunction

    :param graph: tf.Graph, a TensorFlow Graph object
    :param tfobj_or_name: either a tf.Tensor, tf.Operation or a name to either
    """
    graph = validated_graph(graph)
    return op_name(graph, tfobj_or_name)

def validated_input(graph, tfobj_or_name):
    """
    Validate and return the input names useable GraphFunction

    :param graph: tf.Graph, a TensorFlow Graph object
    :param tfobj_or_name: either a tf.Tensor, tf.Operation or a name to either
    """
    graph = validated_graph(graph)
    name = op_name(graph, tfobj_or_name)
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
        [op_name(graph, tnsr) for tnsr in fetches])

    if should_close_session:
        sess.close()

    if return_graph:
        g = tf.Graph()
        with g.as_default():
            tf.import_graph_def(gdef_frozen, name='')
        return g
    else:
        return gdef_frozen
