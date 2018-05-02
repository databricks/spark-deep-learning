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
    Check if the input is a valid :py:class:`tf.Graph` and return it.
    Raise an error otherwise.

    :param graph: :py:class:`tf.Graph`, a TensorFlow Graph object
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
    Get a :py:class:`tf.Operation` object.

    :param tfobj_or_name: either a :py:class:`tf.Tensor`, :py:class:`tf.Operation` or
                          a name to either.
    :param graph: a :py:class:`tf.Graph` object containing the operation.
                  By default the graph we don't require this argument to be provided.
    """
    graph = validated_graph(graph)
    _assert_same_graph(tfobj_or_name, graph)
    if isinstance(tfobj_or_name, tf.Operation):
        return tfobj_or_name
    name = tfobj_or_name
    if isinstance(tfobj_or_name, tf.Tensor):
        name = tfobj_or_name.name
    if not isinstance(name, six.string_types):
        raise TypeError('invalid op request for [type {}] {}'.format(type(name), name))
    _op_name = op_name(name, graph=None)
    op = graph.get_operation_by_name(_op_name)  # pylint: disable=invalid-name
    err_msg = 'cannot locate op {} in the current graph, got [type {}] {}'
    assert isinstance(op, tf.Operation), err_msg.format(_op_name, type(op), op)
    return op


def get_tensor(tfobj_or_name, graph):
    """
    Get a :py:class:`tf.Tensor` object

    :param tfobj_or_name: either a :py:class:`tf.Tensor`, :py:class:`tf.Operation` or
                          a name to either.
    :param graph: a :py:class:`tf.Graph` object containing the tensor.
                  By default the graph we don't require this argument to be provided.
    """
    graph = validated_graph(graph)
    _assert_same_graph(tfobj_or_name, graph)
    if isinstance(tfobj_or_name, tf.Tensor):
        return tfobj_or_name
    name = tfobj_or_name
    if isinstance(tfobj_or_name, tf.Operation):
        name = tfobj_or_name.name
    if not isinstance(name, six.string_types):
        raise TypeError('invalid tensor request for {} of {}'.format(name, type(name)))
    _tensor_name = tensor_name(name, graph=None)
    tnsr = graph.get_tensor_by_name(_tensor_name)
    err_msg = 'cannot locate tensor {} in the current graph, got [type {}] {}'
    assert isinstance(tnsr, tf.Tensor), err_msg.format(_tensor_name, type(tnsr), tnsr)
    return tnsr


def tensor_name(tfobj_or_name, graph=None):
    """
    Derive the :py:class:`tf.Tensor` name from a :py:class:`tf.Operation` or :py:class:`tf.Tensor`
    object, or its name.
    If a name is provided and the graph is not, we will derive the tensor name based on
    TensorFlow's naming convention.
    If the input is a TensorFlow object, or the graph is given, we also check that
    the tensor exists in the associated graph.

    :param tfobj_or_name: either a :py:class:`tf.Tensor`, :py:class:`tf.Operation` or
                          a name to either.
    :param graph: a :py:class:`tf.Graph` object containing the tensor.
                  By default the graph we don't require this argument to be provided.
    """
    if graph is not None:
        return get_tensor(tfobj_or_name, graph).name
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
    Derive the :py:class:`tf.Operation` name from a :py:class:`tf.Operation` or
    :py:class:`tf.Tensor` object, or its name.
    If a name is provided and the graph is not, we will derive the operation name based on
    TensorFlow's naming convention.
    If the input is a TensorFlow object, or the graph is given, we also check that
    the operation exists in the associated graph.

    :param tfobj_or_name: either a :py:class:`tf.Tensor`, :py:class:`tf.Operation` or
                          a name to either.
    :param graph: a :py:class:`tf.Graph` object containing the operation.
                  By default the graph we don't require this argument to be provided.
    """
    if graph is not None:
        return get_op(tfobj_or_name, graph).name
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


def add_scope_to_name(scope, name):
    """ Prepends the provided scope to the passed-in op or tensor name. """
    return "%s/%s" % (scope, name)


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
    op = graph.get_operation_by_name(name)  # pylint: disable=invalid-name
    assert 'Placeholder' == op.type, ('input must be Placeholder, but get', op.type)
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
        g = tf.Graph()  # pylint: disable=invalid-name
        with g.as_default():
            tf.import_graph_def(gdef_frozen, name='')
        return g
    else:
        return gdef_frozen


def _assert_same_graph(tfobj, graph):
    if graph is not None and hasattr(tfobj, 'graph'):
        err_msg = 'the graph of TensorFlow element {} != graph {}'
        assert tfobj.graph == graph, err_msg.format(tfobj, graph)
