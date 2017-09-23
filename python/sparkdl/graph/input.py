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


class TFInputGraph(object):
    """
    An opaque object containing TensorFlow graph.
    This object can be serialized.

    [WARNING] This class should not be called by any user code.
    """

    def __init__(self):
        raise NotImplementedError(
            "Please do NOT build TFInputGraph directly. Instead, use one of the helper functions")

    @classmethod
    def _new_obj_internal(cls):
        # pylint: disable=attribute-defined-outside-init
        obj = object.__new__(cls)
        # TODO: for (de-)serialization, the class should correspond to a ProtocolBuffer definition.
        ##============================================================
        obj.graph_def = None
        obj.input_tensor_name_from_signature = None
        obj.output_tensor_name_from_signature = None
        ##============================================================
        return obj

    @classmethod
    def fromGraph(cls, graph, sess, feed_names, fetch_names):
        """
        Construct a TFInputGraph from a in memory `tf.Graph` object
        """
        return _build_impl(sess=sess, graph=graph, sig_def=None,
                           feed_names=feed_names, fetch_names=fetch_names)

    @classmethod
    def fromGraphDef(cls, graph_def, feed_names, fetch_names):
        """
        Construct a TFInputGraph from a tf.GraphDef object
        """
        assert isinstance(graph_def, tf.GraphDef), \
            ('expect tf.GraphDef type but got', type(graph_def))

        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            tf.import_graph_def(graph_def, name='')
            gin = _build_impl(sess=sess, graph=graph, sig_def=None,
                              feed_names=feed_names, fetch_names=fetch_names)
        return gin

    @classmethod
    def fromCheckpoint(cls, checkpoint_dir, feed_names, fetch_names):
        return _from_checkpoint_impl(checkpoint_dir, signature_def_key=None, feed_names=feed_names,
                                     fetch_names=fetch_names)

    @classmethod
    def fromCheckpointWithSignature(cls, checkpoint_dir, signature_def_key):
        assert signature_def_key is not None
        return _from_checkpoint_impl(checkpoint_dir, signature_def_key, feed_names=None,
                                     fetch_names=None)

    @classmethod
    def fromSavedModel(cls, saved_model_dir, tag_set, feed_names, fetch_names):
        return _from_saved_model_impl(saved_model_dir, tag_set, signature_def_key=None,
                                      feed_names=feed_names, fetch_names=fetch_names)

    @classmethod
    def fromSavedModelWithSignature(cls, saved_model_dir, tag_set, signature_def_key):
        assert signature_def_key is not None
        return _from_saved_model_impl(saved_model_dir, tag_set, signature_def_key=signature_def_key,
                                      feed_names=None, fetch_names=None)


def _from_checkpoint_impl(checkpoint_dir, signature_def_key=None, feed_names=None,
                          fetch_names=None):
    """
    Construct a TFInputGraph from a model checkpoint
    """
    assert (feed_names is None) == (fetch_names is None), \
        'feed_names and fetch_names, if provided must appear together'
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

        sig_def = None
        if signature_def_key is not None:
            sig_def = meta_graph_def.signature_def[signature_def_key]
            assert sig_def, 'singnature_def_key {} provided, '.format(signature_def_key) + \
                'but failed to find it from the meta_graph_def ' + \
                'from checkpoint {}'.format(checkpoint_dir)

        gin = _build_impl(sess=sess, graph=graph, sig_def=sig_def,
                          feed_names=feed_names, fetch_names=fetch_names)
    return gin


def _from_saved_model_impl(saved_model_dir, tag_set, signature_def_key=None, feed_names=None,
                           fetch_names=None):
    """
    Construct a TFInputGraph from a SavedModel
    """
    assert (feed_names is None) == (fetch_names is None), \
        'feed_names and fetch_names, if provided must appear together'
    assert (feed_names is None) != (signature_def_key is None), \
        'must either provide feed_names or singnature_def_key'

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tag_sets = tag_set.split(',')
        meta_graph_def = tf.saved_model.loader.load(sess, tag_sets, saved_model_dir)

        sig_def = None
        if signature_def_key is not None:
            sig_def = tf.contrib.saved_model.get_signature_def_by_key(meta_graph_def,
                                                                      signature_def_key)

        gin = _build_impl(sess=sess, graph=graph, sig_def=sig_def,
                          feed_names=feed_names, fetch_names=fetch_names)
    return gin


def _build_impl(sess, graph, sig_def, feed_names, fetch_names):
    # pylint: disable=protected-access,attribute-defined-outside-init
    assert (feed_names is None) == (fetch_names is None), \
        "if provided, feed_names {} and fetch_names {} ".format(feed_names, fetch_names) + \
        "must be provided together"
    # NOTE(phi-dbq): both have to be set to default
    with sess.as_default(), graph.as_default():
        #_ginfo = import_graph_fn(sess)
        # If `feed_names` nor `fetch_names` is not provided, must infer them from signature
        if feed_names is None and fetch_names is None:
            assert sig_def is not None, \
                "require graph info to figure out the signature mapping"

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
        else:
            feed_mapping = None
            fetch_mapping = None

        for tnsr_name in feed_names:
            assert tfx.get_op(graph, tnsr_name), \
                'requested tensor {} but found none in graph {}'.format(tnsr_name, graph)
        fetches = [tfx.get_tensor(graph, tnsr_name) for tnsr_name in fetch_names]
        graph_def = tfx.strip_and_freeze_until(fetches, graph, sess)

    gin = TFInputGraph._new_obj_internal()
    gin.input_tensor_name_from_signature = feed_mapping
    gin.output_tensor_name_from_signature = fetch_mapping
    gin.graph_def = graph_def
    return gin
