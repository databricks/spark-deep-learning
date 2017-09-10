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

import tensorflow as tf

import sparkdl.graph.utils as tfx
from sparkdl.image.imageIO import imageType

# image stuff

IMAGE_INPUT_PLACEHOLDER_NAME = "sparkdl_image_input"

def imageInputPlaceholder(nChannels=None):
    return tf.placeholder(tf.float32, [None, None, None, nChannels],
                          name=IMAGE_INPUT_PLACEHOLDER_NAME)

class ImageNetConstants:
    NUM_CLASSES = 1000

# InceptionV3 is used in a lot of tests, so we'll make this shortcut available
# For other networks, see the keras_applications module.
class InceptionV3Constants:
    INPUT_SHAPE = (299, 299)
    NUM_OUTPUT_FEATURES = 131072


class TFInputGraph(object):
    """
    An opaque serializable object containing TensorFlow graph.
    """
    # TODO: for (de-)serialization, the class should correspond to a ProtocolBuffer definition.
    def __init__(self, graph_def, input_mapping, output_mapping):
        # tf.GraphDef
        self.graph_def = graph_def

        if isinstance(input_mapping, dict):
            input_mapping = list(input_mapping.items())
        assert isinstance(input_mapping, list), \
            "output mapping must be a list of strings, found type {}".format(type(input_mapping))
        self.input_mapping = sorted(input_mapping)

        if isinstance(output_mapping, dict):
            output_mapping = list(output_mapping.items())
        assert isinstance(output_mapping, list), \
            "output mapping must be a list of strings, found type {}".format(type(output_mapping))
        self.output_mapping = sorted(output_mapping)


class TFInputGraphBuilder(object):
    """
    Create a builder function so as to be able to compile graph for inference.
    The actual compilation will be done at the time when the
    inputs (feeds) and outputs (fetches) are provided.
    :param graph_import_fn: `tf.Session` -> `tf.signature_def`, load a graph to the provided session.
                            If the meta_graph contains a `signature_def`, return it.
    """
    def __init__(self, graph_import_fn):
        # Return signature_def if the underlying graph contains one
        self.graph_import_fn = graph_import_fn

    def build(self, input_mapping, output_mapping):
        """
        Create a serializable TensorFlow graph representation
        :param input_mapping: dict, from input DataFrame column name to internal graph name.
        """
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            sig_def = self.graph_import_fn(sess)

            # Append feeds and input mapping
            _input_mapping = {}
            for input_colname, tnsr_or_sig in input_mapping.items():
                if sig_def:
                    tnsr = sig_def.inputs[tnsr_or_sig].name
                else:
                    tnsr = tnsr_or_sig
                _input_mapping[input_colname] = tfx.op_name(graph, tnsr)
            input_mapping = _input_mapping

            # Append fetches and output mapping
            fetches = []
            _output_mapping = {}
            # By default the output columns will have the name of their
            # corresponding `tf.Graph` operation names.
            # We have to convert them to the user specified output names
            for tnsr_or_sig, requested_colname in output_mapping.items():
                if sig_def:
                    tnsr = sig_def.outputs[tnsr_or_sig].name
                else:
                    tnsr = tnsr_or_sig
                fetches.append(tfx.get_tensor(graph, tnsr))
                tf_output_colname = tfx.op_name(graph, tnsr)
                _output_mapping[tf_output_colname] = requested_colname
            output_mapping = _output_mapping

            gdef = tfx.strip_and_freeze_until(fetches, graph, sess)

        return TFInputGraph(gdef, input_mapping, output_mapping)

    @classmethod
    def fromGraph(cls, graph):
        """
        Construct a TFInputGraphBuilder from a in memory tf.Graph object
        """
        assert isinstance(graph, tf.Graph), \
            ('expect tf.Graph type but got', type(graph))

        def import_graph_fn(sess):
            gdef = graph.as_graph_def(add_shapes=True)
            with sess.as_default():
                tf.import_graph_def(gdef, name='')
            return None  # no meta_graph_def

        return cls(import_graph_fn)

    @classmethod
    def fromGraphDef(cls, graph_def):
        """
        Construct a TFInputGraphBuilder from a tf.GraphDef object
        """
        assert isinstance(graph_def, tf.GraphDef), \
            ('expect tf.GraphDef type but got', type(graph_def))

        def import_graph_fn(sess):
            with sess.as_default():
                tf.import_graph_def(graph_def, name='')
            return None

        return cls(import_graph_fn)

    @classmethod
    def fromCheckpoint(cls, checkpoint_dir, signature_def_key=None):
        """
        Construct a TFInputGraphBuilder from a model checkpoint
        """
        def import_graph_fn(sess):
            # Load checkpoint and import the graph
            ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
            saver = tf.train.import_meta_graph("{}.meta".format(ckpt_path), clear_devices=True)
            saver.restore(sess, ckpt_path)
            meta_graph_def = saver.export_meta_graph(clear_devices=True)

            sig_def = None
            if signature_def_key is not None:
                sig_def = tf.contrib.saved_model.get_signature_def_by_key(
                    meta_graph_def, signature_def_key)

            return sig_def

        return cls(import_graph_fn)

    @classmethod
    def fromSavedModel(cls, saved_model_dir, tag_set, signature_def_key=None):
        """
        Construct a TFInputGraphBuilder from a SavedModel
        """
        def import_graph_fn(sess):
            tag_sets = tag_set.split(',')
            meta_graph_def = tf.saved_model.loader.load(sess, tag_sets, saved_model_dir)

            sig_def = None
            if signature_def_key is not None:
                sig_def = tf.contrib.saved_model.get_signature_def_by_key(
                    meta_graph_def, signature_def_key)

            return sig_def

        return cls(import_graph_fn)
