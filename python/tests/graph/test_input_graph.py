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

from contextlib import contextmanager
from glob import glob
import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf

from sparkdl.graph.input import *
import sparkdl.graph.utils as tfx

from ..tests import PythonUnitTestCase


class TFInputGraphTest(PythonUnitTestCase):

    def setUp(self):
        self.vec_size = 23
        self.num_samples = 107

        self.input_col = 'dfInputCol'
        self.input_op_name = 'tnsrOpIn'
        self.output_col = 'dfOutputCol'
        self.output_op_name = 'tnsrOpOut'

        self.feed_names = []
        self.fetch_names = []
        self.input_mapping = {}
        self.output_mapping = {}
        self.setup_iomap(replica=1)

        self.input_graphs = []
        self.test_case_results = []
        # Build a temporary directory, which might or might not be used by the test
        self.model_output_root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.model_output_root, ignore_errors=True)

    def setup_iomap(self, replica=1):
        self.input_mapping = {}
        self.feed_names = []
        self.output_mapping = {}
        self.fetch_names = []

        if replica > 1:
            for i in range(replica):
                colname = '{}_replica{:03d}'.format(self.input_col, i)
                tnsr_op_name = '{}_replica{:03d}'.format(self.input_op_name, i)
                self.input_mapping[colname] = tnsr_op_name
                self.feed_names.append(tnsr_op_name + ':0')

                colname = '{}_replica{:03d}'.format(self.output_col, i)
                tnsr_op_name = '{}_replica{:03d}'.format(self.output_op_name, i)
                self.output_mapping[tnsr_op_name] = colname
                self.fetch_names.append(tnsr_op_name + ':0')
        else:
            self.input_mapping = {self.input_col: self.input_op_name}
            self.feed_names = [self.input_op_name + ':0']
            self.output_mapping = {self.output_op_name: self.output_col}
            self.fetch_names = [self.output_op_name + ':0']

    @contextmanager
    def _run_test_in_tf_session(self):
        """ [THIS IS NOT A TEST]: encapsulate general test workflow """

        # Build the TensorFlow graph
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess, graph.as_default():
            # Build test graph and transformers from here
            yield sess

            ref_feed = tfx.get_tensor(self.input_op_name, graph)
            ref_fetch = tfx.get_tensor(self.output_op_name, graph)

            def check_input_graph(tgt_gdef, test_idx):
                namespace = 'TEST_TGT_NS{:03d}'.format(test_idx)
                tf.import_graph_def(tgt_gdef, name=namespace)
                tgt_feed = tfx.get_tensor('{}/{}'.format(namespace, self.input_op_name), graph)
                tgt_fetch = tfx.get_tensor('{}/{}'.format(namespace, self.output_op_name), graph)

                for _ in range(10):
                    local_data = np.random.randn(31, self.vec_size)
                    ref_out = sess.run(ref_fetch, feed_dict={ref_feed: local_data})
                    tgt_out = sess.run(tgt_fetch, feed_dict={tgt_feed: local_data})
                    self.assertTrue(np.allclose(ref_out, tgt_out))

            for test_idx, input_graph in enumerate(self.input_graphs):
                check_input_graph(input_graph.graph_def, test_idx)


    def test_build_from_tf_graph(self):
        """ Build TFTransformer from tf.Graph """
        with self._run_test_in_tf_session() as sess:
            # Begin building graph
            x = tf.placeholder(tf.float64, shape=[None, self.vec_size], name=self.input_op_name)
            _ = tf.reduce_mean(x, axis=1, name=self.output_op_name)

            gin = TFInputGraph.fromGraph(sess.graph, sess, self.feed_names, self.fetch_names)
            self.input_graphs.append(gin)
            # End building graph

    def test_build_from_saved_model(self):
        """ Build TFTransformer from saved model """
        # Setup saved model export directory
        saved_model_root = self.model_output_root
        saved_model_dir = os.path.join(saved_model_root, 'saved_model')
        serving_tag = "serving_tag"
        serving_sigdef_key = 'prediction_signature'
        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)

        with self._run_test_in_tf_session() as sess:
            # Model definition: begin
            x = tf.placeholder(tf.float64, shape=[None, self.vec_size], name=self.input_op_name)
            w = tf.Variable(tf.random_normal([self.vec_size], dtype=tf.float64),
                            dtype=tf.float64, name='varW')
            z = tf.reduce_mean(x * w, axis=1, name=self.output_op_name)
            # Model definition ends

            sess.run(w.initializer)

            sig_inputs = {
                'input_sig': tf.saved_model.utils.build_tensor_info(x)}
            sig_outputs = {
                'output_sig': tf.saved_model.utils.build_tensor_info(z)}

            serving_sigdef = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=sig_inputs,
                outputs=sig_outputs)

            builder.add_meta_graph_and_variables(sess,
                                                 [serving_tag],
                                                 signature_def_map={
                                                     serving_sigdef_key: serving_sigdef})
            builder.save()

            # Build the transformer from exported serving model
            # We are using signaures, thus must provide the keys
            gin = TFInputGraph.fromSavedModelWithSignature(
                saved_model_dir, serving_tag, serving_sigdef_key)
            self.input_graphs.append(gin)

            # Build the transformer from exported serving model
            # We are not using signatures, thus must provide tensor/operation names
            gin = TFInputGraph.fromSavedModel(
                saved_model_dir, serving_tag, self.feed_names, self.fetch_names)
            self.input_graphs.append(gin)

            gin = TFInputGraph.fromGraph(
                sess.graph, sess, self.feed_names, self.fetch_names)
            self.input_graphs.append(gin)


    def test_build_from_checkpoint(self):
        """ Build TFTransformer from a model checkpoint """
        # Build the TensorFlow graph
        model_ckpt_dir = self.model_output_root
        ckpt_path_prefix = os.path.join(model_ckpt_dir, 'model_ckpt')
        serving_sigdef_key = 'prediction_signature'

        with self._run_test_in_tf_session() as sess:
            x = tf.placeholder(tf.float64, shape=[None, self.vec_size], name=self.input_op_name)
            #x = tf.placeholder(tf.float64, shape=[None, vec_size], name=input_col)
            w = tf.Variable(tf.random_normal([self.vec_size], dtype=tf.float64),
                            dtype=tf.float64, name='varW')
            z = tf.reduce_mean(x * w, axis=1, name=self.output_op_name)
            sess.run(w.initializer)
            saver = tf.train.Saver(var_list=[w])
            _ = saver.save(sess, ckpt_path_prefix, global_step=2702)

            # Prepare the signature_def
            serving_sigdef = tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    'input_sig': tf.saved_model.utils.build_tensor_info(x)
                },
                outputs={
                    'output_sig': tf.saved_model.utils.build_tensor_info(z)
                })

            # A rather contrived way to add signature def to a meta_graph
            meta_graph_def = tf.train.export_meta_graph()

            # Find the meta_graph file (there should be only one)
            _ckpt_meta_fpaths = glob('{}/*.meta'.format(model_ckpt_dir))
            self.assertEqual(len(_ckpt_meta_fpaths), 1, msg=','.join(_ckpt_meta_fpaths))
            ckpt_meta_fpath = _ckpt_meta_fpaths[0]

            # Add signature_def to the meta_graph and serialize it
            # This will overwrite the existing meta_graph_def file
            meta_graph_def.signature_def[serving_sigdef_key].CopyFrom(serving_sigdef)
            with open(ckpt_meta_fpath, mode='wb') as fout:
                fout.write(meta_graph_def.SerializeToString())

            # Build the transformer from exported serving model
            # We are using signaures, thus must provide the keys
            gin = TFInputGraph.fromCheckpointWithSignature(
                model_ckpt_dir, serving_sigdef_key)
            self.input_graphs.append(gin)

            # Transformer without using signature_def
            gin = TFInputGraph.fromCheckpoint(model_ckpt_dir, self.feed_names, self.fetch_names)
            self.input_graphs.append(gin)

            gin = TFInputGraph.fromGraph(
                sess.graph, sess, self.feed_names, self.fetch_names)
            self.input_graphs.append(gin)
