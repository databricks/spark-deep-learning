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

import contextlib
import shutil
import numpy as np
import os
import tensorflow as tf
import tempfile
import glob

import sparkdl.graph.utils as tfx
from sparkdl.graph.input import TFInputGraph


class TestGraphImport(object):
    def test_graph_novar(self):
        gin = _build_graph_input(lambda session:
                                 TFInputGraph.fromGraph(session.graph, session, [_tensor_input_name],
                                                        [_tensor_output_name]))
        _check_input_novar(gin)

    def test_graphdef_novar(self):
        gin = _build_graph_input(lambda session:
                                 TFInputGraph.fromGraphDef(session.graph.as_graph_def(),
                                                           [_tensor_input_name], [_tensor_output_name]))
        _check_input_novar(gin)

    def test_saved_model_novar(self):
        with _make_temp_directory() as tmp_dir:
            saved_model_dir = os.path.join(tmp_dir, 'saved_model')

            def gin_fun(session):
                _build_saved_model(session, saved_model_dir)
                # Build the transformer from exported serving model
                # We are using signatures, thus must provide the keys
                return TFInputGraph.fromSavedModelWithSignature(saved_model_dir, _serving_tag,
                                                                _serving_sigdef_key)

            gin = _build_graph_input(gin_fun)
            _check_input_novar(gin)

    def test_saved_model_iomap(self):
        with _make_temp_directory() as tmp_dir:
            saved_model_dir = os.path.join(tmp_dir, 'saved_model')
            graph = tf.Graph()
            with tf.Session(graph=graph) as sess, graph.as_default():
                _build_graph()
                _build_saved_model(sess, saved_model_dir)
                # Build the transformer from exported serving model
                # We are using signatures, thus must provide the keys
                gin = TFInputGraph.fromSavedModelWithSignature(saved_model_dir, _serving_tag,
                                                               _serving_sigdef_key)

                _input_mapping_with_sigdef = {'inputCol': _tensor_input_signature}
                # Input mapping for the Transformer
                _translated_input_mapping = gin.translateInputMapping(_input_mapping_with_sigdef)
                _expected_input_mapping = {'inputCol': tfx.tensor_name(_tensor_input_name)}
                # Output mapping for the Transformer
                _output_mapping_with_sigdef = {_tensor_output_signature: 'outputCol'}
                _translated_output_mapping = gin.translateOutputMapping(_output_mapping_with_sigdef)
                _expected_output_mapping = {tfx.tensor_name(_tensor_output_name): 'outputCol'}

                err_msg = "signature based input mapping {} and output mapping {} " + \
                          "must be translated correctly into tensor name based mappings"
                assert _translated_input_mapping == _expected_input_mapping \
                    and _translated_output_mapping == _expected_output_mapping, \
                    err_msg.format(_translated_input_mapping, _translated_output_mapping)

    def test_saved_graph_novar(self):
        with _make_temp_directory() as tmp_dir:
            saved_model_dir = os.path.join(tmp_dir, 'saved_model')

            def gin_fun(session):
                _build_saved_model(session, saved_model_dir)
                return TFInputGraph.fromGraph(session.graph, session,
                                              [_tensor_input_name], [_tensor_output_name])

            gin = _build_graph_input(gin_fun)
            _check_input_novar(gin)

    def test_checkpoint_sig_var(self):
        with _make_temp_directory() as tmp_dir:
            def gin_fun(session):
                _build_checkpointed_model(session, tmp_dir)
                return TFInputGraph.fromCheckpointWithSignature(tmp_dir, _serving_sigdef_key)

            gin = _build_graph_input_var(gin_fun)
            _check_input_novar(gin)

    def test_checkpoint_nosig_var(self):
        with _make_temp_directory() as tmp_dir:
            def gin_fun(session):
                _build_checkpointed_model(session, tmp_dir)
                return TFInputGraph.fromCheckpoint(tmp_dir,
                                                   [_tensor_input_name], [_tensor_output_name])

            gin = _build_graph_input_var(gin_fun)
            _check_input_novar(gin)

    def test_checkpoint_graph_var(self):
        with _make_temp_directory() as tmp_dir:
            def gin_fun(session):
                _build_checkpointed_model(session, tmp_dir)
                return TFInputGraph.fromGraph(session.graph, session,
                                              [_tensor_input_name], [_tensor_output_name])

            gin = _build_graph_input_var(gin_fun)
            _check_input_novar(gin)

    def test_graphdef_novar_2(self):
        gin = _build_graph_input_2(lambda session:
                                   TFInputGraph.fromGraphDef(session.graph.as_graph_def(),
                                                             [_tensor_input_name], [_tensor_output_name]))
        _check_output_2(gin, np.array([1, 2, 3]), np.array([2, 2, 2]), 1)

    def test_saved_graph_novar_2(self):
        with _make_temp_directory() as tmp_dir:
            saved_model_dir = os.path.join(tmp_dir, 'saved_model')

            def gin_fun(session):
                _build_saved_model(session, saved_model_dir)
                return TFInputGraph.fromGraph(session.graph, session,
                                              [_tensor_input_name], [_tensor_output_name])

            gin = _build_graph_input_2(gin_fun)
            _check_output_2(gin, np.array([1, 2, 3]), np.array([2, 2, 2]), 1)


_serving_tag = "serving_tag"
_serving_sigdef_key = 'prediction_signature'
# The name of the input tensor
_tensor_input_name = "input_tensor"
# For testing graphs with 2 inputs
_tensor_input_name_2 = "input_tensor_2"
# The name of the output tensor (scalar)
_tensor_output_name = "output_tensor"
# Input signature name
_tensor_input_signature = 'well_known_input_sig'
# Output signature name
_tensor_output_signature = 'well_known_output_sig'
# The name of the variable
_tensor_var_name = "variable"
# The size of the input tensor
_tensor_size = 3


def _build_checkpointed_model(session, tmp_dir):
    """
    Writes a model checkpoint in the given directory. The graph is assumed to be generated
     with _build_graph_var.
    """
    ckpt_path_prefix = os.path.join(tmp_dir, 'model_ckpt')
    input_tensor = tfx.get_tensor(_tensor_input_name, session.graph)
    output_tensor = tfx.get_tensor(_tensor_output_name, session.graph)
    w = tfx.get_tensor(_tensor_var_name, session.graph)
    saver = tf.train.Saver(var_list=[w])
    _ = saver.save(session, ckpt_path_prefix, global_step=2702)
    sig_inputs = {_tensor_input_signature: tf.saved_model.utils.build_tensor_info(input_tensor)}
    sig_outputs = {_tensor_output_signature: tf.saved_model.utils.build_tensor_info(output_tensor)}
    serving_sigdef = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=sig_inputs, outputs=sig_outputs)

    # A rather contrived way to add signature def to a meta_graph
    meta_graph_def = tf.train.export_meta_graph()

    # Find the meta_graph file (there should be only one)
    _ckpt_meta_fpaths = glob.glob('{}/*.meta'.format(tmp_dir))
    assert len(_ckpt_meta_fpaths) == 1, \
        'expected only one meta graph, but got {}'.format(','.join(_ckpt_meta_fpaths))
    ckpt_meta_fpath = _ckpt_meta_fpaths[0]

    # Add signature_def to the meta_graph and serialize it
    # This will overwrite the existing meta_graph_def file
    meta_graph_def.signature_def[_serving_sigdef_key].CopyFrom(serving_sigdef)
    with open(ckpt_meta_fpath, mode='wb') as fout:
        fout.write(meta_graph_def.SerializeToString())


def _build_saved_model(session, saved_model_dir):
    """
    Saves a model in a file. The graph is assumed to be generated with _build_graph_novar.
    """
    builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
    input_tensor = tfx.get_tensor(_tensor_input_name, session.graph)
    output_tensor = tfx.get_tensor(_tensor_output_name, session.graph)
    sig_inputs = {_tensor_input_signature: tf.saved_model.utils.build_tensor_info(input_tensor)}
    sig_outputs = {_tensor_output_signature: tf.saved_model.utils.build_tensor_info(output_tensor)}
    serving_sigdef = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=sig_inputs, outputs=sig_outputs)

    builder.add_meta_graph_and_variables(
        session, [_serving_tag], signature_def_map={_serving_sigdef_key: serving_sigdef})
    builder.save()


@contextlib.contextmanager
def _make_temp_directory():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


def _build_graph_input(gin_function):
    """
    Makes a session and a default graph, loads the simple graph into it, and then calls
    gin_function(session) to return the graph input object
    """
    graph = tf.Graph()
    with tf.Session(graph=graph) as s, graph.as_default():
        _build_graph()
        return gin_function(s)


def _build_graph_input_2(gin_function):
    """
    Makes a session and a default graph, loads the simple graph into it (graph_2), and then calls
    gin_function(session) to return the graph input object
    """
    graph = tf.Graph()
    with tf.Session(graph=graph) as s, graph.as_default():
        _build_graph_2()
        return gin_function(s)


def _build_graph_input_var(gin_function):
    """
    Makes a session and a default graph, loads the simple graph into it that contains a variable,
     and then calls gin_function(session) to return the graph input object
    """
    graph = tf.Graph()
    with tf.Session(graph=graph) as s, graph.as_default():
        _build_graph_var(s)
        return gin_function(s)


def _build_graph():
    """
    Given a session (implicitly), adds nodes of computations

    It takes a vector input, with vec_size columns and returns an int32 scalar.
    """
    x = tf.placeholder(tf.int32, shape=[_tensor_size], name=_tensor_input_name)
    _ = tf.reduce_max(x, name=_tensor_output_name)


def _build_graph_2():
    """
    Given a session (implicitly), adds nodes of computations with two inputs.

    It takes a vector input, with vec_size columns and returns an int32 scalar.
    """
    x1 = tf.placeholder(tf.int32, shape=[_tensor_size], name=_tensor_input_name)
    x2 = tf.placeholder(tf.int32, shape=[_tensor_size], name=_tensor_input_name_2)
    # Make sure that the inputs are not used in a symmetric manner.
    _ = tf.reduce_max(x1 - x2, name=_tensor_output_name)


def _build_graph_var(session):
    """
    Given a session, adds nodes that include one variable.
    """
    x = tf.placeholder(tf.int32, shape=[_tensor_size], name=_tensor_input_name)
    w = tf.Variable(tf.ones(shape=[_tensor_size], dtype=tf.int32), name=_tensor_var_name)
    _ = tf.reduce_max(x * w, name=_tensor_output_name)
    session.run(w.initializer)


def _check_input_novar(gin):
    """
    Tests that the graph from _build_graph has been serialized in the InputGraph object.
    """
    _check_output(gin, np.array([1, 2, 3]), 3)


def _check_output(gin, tf_input, expected):
    """
    Takes a TFInputGraph object (assumed to have the input and outputs of the given
    names above) and compares the outcome against some expected outcome.
    """
    graph = tf.Graph()
    graph_def = gin.graph_def
    with tf.Session(graph=graph) as sess:
        tf.import_graph_def(graph_def, name="")
        tgt_feed = tfx.get_tensor(_tensor_input_name, graph)
        tgt_fetch = tfx.get_tensor(_tensor_output_name, graph)
        # Run on the testing target
        tgt_out = sess.run(tgt_fetch, feed_dict={tgt_feed: tf_input})
        # Working on integers, the calculation should be exact
        assert np.all(tgt_out == expected), (tgt_out, expected)


# TODO: we could factorize with _check_output, but this is not worth the time doing it.
def _check_output_2(gin, tf_input1, tf_input2, expected):
    """
    Takes a TFInputGraph object (assumed to have the input and outputs of the given
    names above) and compares the outcome against some expected outcome.
    """
    graph = tf.Graph()
    graph_def = gin.graph_def
    with tf.Session(graph=graph) as sess:
        tf.import_graph_def(graph_def, name="")
        tgt_feed1 = tfx.get_tensor(_tensor_input_name, graph)
        tgt_feed2 = tfx.get_tensor(_tensor_input_name_2, graph)
        tgt_fetch = tfx.get_tensor(_tensor_output_name, graph)
        # Run on the testing target
        tgt_out = sess.run(tgt_fetch, feed_dict={tgt_feed1: tf_input1, tgt_feed2: tf_input2})
        # Working on integers, the calculation should be exact
        assert np.all(tgt_out == expected), (tgt_out, expected)
