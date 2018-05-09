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

import tensorflow as tf
from tensorflow.core.protobuf import meta_graph_pb2  # pylint: disable=no-name-in-module

import sparkdl.graph.utils as tfx

__all__ = ["TFInputGraph"]

# pylint: disable=invalid-name,wrong-spelling-in-comment,wrong-spelling-in-docstring


class TFInputGraph(object):
    """
    An opaque object containing TensorFlow graph.
    This object can be serialized.

    .. note:: We recommend constructing this object using one of the class constructor methods.

              - :py:meth:`fromGraph`
              - :py:meth:`fromGraphDef`
              - :py:meth:`fromCheckpoint`
              - :py:meth:`fromCheckpointWithSignature`
              - :py:meth:`fromSavedModel`
              - :py:meth:`fromSavedModelWithSignature`


    When the graph contains serving signatures in which a set of well-known names are associated
    with their corresponding raw tensor names in the graph, we extract and store them here.
    For example, the TensorFlow saved model may contain the following structure,
    so that end users can retrieve the the input tensor via `well_known_input_sig` and
    the output tensor via `well_known_output_sig` without knowing the actual tensor names a priori.

    .. code-block:: python

        sigdef: {'well_known_prediction_signature':
        inputs { key: "well_known_input_sig"
          value {
            name: "tnsrIn:0"
            dtype: DT_DOUBLE
            tensor_shape { dim { size: -1 } dim { size: 17 } }
            }
          }
        outputs { key: "well_known_output_sig"
          value {
            name: "tnsrOut:0"
            dtype: DT_DOUBLE
            tensor_shape { dim { size: -1 } }
            }
        }}


    In this case, the class will internally store the mapping from signature names to tensor names.

    .. code-block:: python

        {'well_known_input_sig': 'tnsrIn:0'}
        {'well_known_output_sig': 'tnsrOut:0'}


    :param graph_def: :py:obj:`tf.GraphDef`, a serializable object containing the topology and
                       computation units of the TensorFlow graph. The graph object is prepared for
                       inference, i.e. the variables are converted to constants and operations like
                       BatchNormalization_ are converted to be independent of input batch.

   .. _BatchNormalization: https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization

    :param input_tensor_name_from_signature: dict, signature key names mapped to tensor names.
                                             Please see the example above.
    :param output_tensor_name_from_signature: dict, signature key names mapped to tensor names
                                              Please see the example above.
    """

    def __init__(self, graph_def, input_tensor_name_from_signature,
                 output_tensor_name_from_signature):
        self.graph_def = graph_def
        self.input_tensor_name_from_signature = input_tensor_name_from_signature
        self.output_tensor_name_from_signature = output_tensor_name_from_signature

    def translateInputMapping(self, input_mapping):
        """
        When the meta_graph contains signature_def, we expect users to provide
        input and output mapping with respect to the tensor reference keys
        embedded in the `signature_def`.

        This function translates the input_mapping into the canonical format,
        which maps input DataFrame column names to tensor names.

        :param input_mapping: dict, DataFrame column name to tensor reference names
                              defined in the signature_def key.
        """
        assert self.input_tensor_name_from_signature is not None
        _input_mapping = {}
        if isinstance(input_mapping, dict):
            input_mapping = list(input_mapping.items())
        assert isinstance(input_mapping, list)
        for col_name, sig_key in input_mapping:
            tnsr_name = self.input_tensor_name_from_signature[sig_key]
            _input_mapping[col_name] = tnsr_name
        return _input_mapping

    def translateOutputMapping(self, output_mapping):
        """
        When the meta_graph contains signature_def, we expect users to provide
        input and output mapping with respect to the tensor reference keys
        embedded in the `signature_def`.

        This function translates the output_mapping into the canonical format,
        which maps tensor names into input DataFrame column names.

        :param output_mapping: dict, tensor reference names defined in the signature_def keys
                               into the output DataFrame column names.
        """
        assert self.output_tensor_name_from_signature is not None
        _output_mapping = {}
        if isinstance(output_mapping, dict):
            output_mapping = list(output_mapping.items())
        assert isinstance(output_mapping, list)
        for sig_key, col_name in output_mapping:
            tnsr_name = self.output_tensor_name_from_signature[sig_key]
            _output_mapping[tnsr_name] = col_name
        return _output_mapping

    @classmethod
    def fromGraph(cls, graph, sess, feed_names, fetch_names):
        """
        Construct a TFInputGraph from a in memory `tf.Graph` object.
        The graph might contain variables that are maintained in the provided session.
        Thus we need an active session in which the graph's variables are initialized or
        restored. We do not close the session. As a result, this constructor can be used
        inside a standard TensorFlow session context.

        .. code-block:: python

             with tf.Session() as sess:
                 graph = import_my_tensorflow_graph(...)
                 input = TFInputGraph.fromGraph(graph, sess, ...)

        :param graph: a :py:class:`tf.Graph` object containing the topology and computation units of
                      the TensorFlow graph.
        :param feed_names: list, names of the input tensors.
        :param fetch_names: list, names of the output tensors.
        """
        return _build_with_feeds_fetches(sess=sess, graph=graph, feed_names=feed_names,
                                         fetch_names=fetch_names)

    @classmethod
    def fromGraphDef(cls, graph_def, feed_names, fetch_names):
        """
        Construct a TFInputGraph from a tf.GraphDef object.

        :param graph_def: :py:class:`tf.GraphDef`, a serializable object containing the topology and
                           computation units of the TensorFlow graph.
        :param feed_names: list, names of the input tensors.
        :param fetch_names: list, names of the output tensors.
        """
        assert isinstance(graph_def, tf.GraphDef), \
            ('expect tf.GraphDef type but got', type(graph_def))

        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            tf.import_graph_def(graph_def, name='')
            return _build_with_feeds_fetches(sess=sess, graph=graph, feed_names=feed_names,
                                             fetch_names=fetch_names)

    @classmethod
    def fromCheckpoint(cls, checkpoint_dir, feed_names, fetch_names):
        """
        Construct a TFInputGraph object from a checkpoint, ignore the embedded
        signature_def, if there is any.

        :param checkpoint_dir: str, name of the directory containing the TensorFlow graph
                               training checkpoint.
        :param feed_names: list, names of the input tensors.
        :param fetch_names: list, names of the output tensors.
        """
        return _from_checkpoint_impl(checkpoint_dir, signature_def_key=None, feed_names=feed_names,
                                     fetch_names=fetch_names)

    @classmethod
    def fromCheckpointWithSignature(cls, checkpoint_dir, signature_def_key):
        """
        Construct a TFInputGraph object from a checkpoint, using the embedded
        signature_def. Throw an error if we cannot find an entry with the `signature_def_key`
        inside the `signature_def`.

        :param checkpoint_dir: str, name of the directory containing the TensorFlow graph
                               training checkpoint.
        :param signature_def_key: str, key (name) of the signature_def to use. It should be in
                                  the list of `signature_def` structures saved with the checkpoint.
        """
        assert signature_def_key is not None
        return _from_checkpoint_impl(checkpoint_dir, signature_def_key, feed_names=None,
                                     fetch_names=None)

    @classmethod
    def fromSavedModel(cls, saved_model_dir, tag_set, feed_names, fetch_names):
        """
        Construct a TFInputGraph object from a saved model (`tf.SavedModel`) directory.
        Ignore the the embedded signature_def, if there is any.

        :param saved_model_dir: str, name of the directory containing the TensorFlow graph
                                training checkpoint.
        :param tag_set: str, name of the graph stored in this meta_graph of the saved model
                        that we are interested in using.
        :param feed_names: list, names of the input tensors.
        :param fetch_names: list, names of the output tensors.
        """
        return _from_saved_model_impl(saved_model_dir, tag_set, signature_def_key=None,
                                      feed_names=feed_names, fetch_names=fetch_names)

    @classmethod
    def fromSavedModelWithSignature(cls, saved_model_dir, tag_set, signature_def_key):
        """
        Construct a TFInputGraph object from a saved model (`tf.SavedModel`) directory,
        using the embedded signature_def. Throw error if we cannot find an entry with
        the `signature_def_key` inside the `signature_def`.

        :param saved_model_dir: str, name of the directory containing the TensorFlow graph
                                training checkpoint.
        :param tag_set: str, name of the graph stored in this meta_graph of the saved model
                        that we are interested in using.
        :param signature_def_key: str, key (name) of the signature_def to use. It should be in
                                  the list of `signature_def` structures saved with the
                                  TensorFlow `SavedModel`.
        """
        assert signature_def_key is not None
        return _from_saved_model_impl(saved_model_dir, tag_set, signature_def_key=signature_def_key,
                                      feed_names=None, fetch_names=None)


def _from_checkpoint_impl(checkpoint_dir, signature_def_key, feed_names, fetch_names):
    """
    Construct a TFInputGraph from a model checkpoint.
    Notice that one should either provide the `signature_def_key` or provide both
    `feed_names` and `fetch_names`. Please set the unprovided values to None.

    :param signature_def_key: str, name of the mapping contained inside the `signature_def`
                              from which we retrieve the signature key to tensor names mapping.
    :param feed_names: list, names of the input tensors.
    :param fetch_names: list, names of the output tensors.
    """
    assert (feed_names is None) == (fetch_names is None), \
        'feed_names and fetch_names, if provided must be both non-None.'
    assert (feed_names is None) != (signature_def_key is None), \
        'must either provide feed_names or singnature_def_key'

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        # Load checkpoint and import the graph
        ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)

        # NOTE(phi-dbq): we must manually load meta_graph_def to get the signature_def
        #                the current `import_graph_def` function seems to ignore
        #                any signature_def fields in a checkpoint's meta_graph_def.
        meta_graph_def = meta_graph_pb2.MetaGraphDef()
        with open("{}.meta".format(ckpt_path), 'rb') as fin:
            meta_graph_def.ParseFromString(fin.read())

        saver = tf.train.import_meta_graph(meta_graph_def, clear_devices=True)
        saver.restore(sess, ckpt_path)

        if signature_def_key is not None:
            sig_def = meta_graph_def.signature_def[signature_def_key]
            return _build_with_sig_def(sess=sess, graph=graph, sig_def=sig_def)
        else:
            return _build_with_feeds_fetches(sess=sess, graph=graph, feed_names=feed_names,
                                             fetch_names=fetch_names)


def _from_saved_model_impl(saved_model_dir, tag_set, signature_def_key, feed_names, fetch_names):
    """
    Construct a TFInputGraph from a SavedModel.
    Notice that one should either provide the `signature_def_key` or provide both
    `feed_names` and `fetch_names`. Please set the unprovided values to None.

    :param signature_def_key: str, name of the mapping contained inside the `signature_def`
                              from which we retrieve the signature key to tensor names mapping.
    :param feed_names: list, names of the input tensors.
    :param fetch_names: list, names of the output tensors.
    """
    assert (feed_names is None) == (fetch_names is None), \
        'feed_names and fetch_names, if provided must be both non-None.'
    assert (feed_names is None) != (signature_def_key is None), \
        'must either provide feed_names or singnature_def_key'

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tag_sets = tag_set.split(',')
        meta_graph_def = tf.saved_model.loader.load(sess, tag_sets, saved_model_dir)

        if signature_def_key is not None:
            sig_def = tf.contrib.saved_model.get_signature_def_by_key(meta_graph_def,
                                                                      signature_def_key)
            return _build_with_sig_def(sess=sess, graph=graph, sig_def=sig_def)
        else:
            return _build_with_feeds_fetches(sess=sess, graph=graph, feed_names=feed_names,
                                             fetch_names=fetch_names)


def _build_with_sig_def(sess, graph, sig_def):
    # pylint: disable=protected-access
    assert sig_def, 'signature_def must not be None'

    with sess.as_default(), graph.as_default():
        feed_mapping = {}
        feed_names = []
        for sigdef_key, tnsr_info in sig_def.inputs.items():
            tnsr_name = tnsr_info.name
            feed_mapping[sigdef_key] = tnsr_name
            feed_names.append(tnsr_name)

        fetch_mapping = {}
        fetch_names = []
        for sigdef_key, tnsr_info in sig_def.outputs.items():
            tnsr_name = tnsr_info.name
            fetch_mapping[sigdef_key] = tnsr_name
            fetch_names.append(tnsr_name)

        for tnsr_name in feed_names:
            assert tfx.get_op(tnsr_name, graph), \
                'requested tensor {} but found none in graph {}'.format(tnsr_name, graph)
        fetches = [tfx.get_tensor(tnsr_name, graph) for tnsr_name in fetch_names]
        graph_def = tfx.strip_and_freeze_until(fetches, graph, sess)

    return TFInputGraph(graph_def=graph_def, input_tensor_name_from_signature=feed_mapping,
                        output_tensor_name_from_signature=fetch_mapping)


def _build_with_feeds_fetches(sess, graph, feed_names, fetch_names):
    assert feed_names is not None, "must provide feed_names"
    assert fetch_names is not None, "must provide fetch names"

    with sess.as_default(), graph.as_default():
        for tnsr_name in feed_names:
            assert tfx.get_op(tnsr_name, graph), \
                'requested tensor {} but found none in graph {}'.format(tnsr_name, graph)
        fetches = [tfx.get_tensor(tnsr_name, graph) for tnsr_name in fetch_names]
        graph_def = tfx.strip_and_freeze_until(fetches, graph, sess)

    return TFInputGraph(graph_def=graph_def, input_tensor_name_from_signature=None,
                        output_tensor_name_from_signature=None)
