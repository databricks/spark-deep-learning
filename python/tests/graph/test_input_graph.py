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

from collections import namedtuple
from contextlib import contextmanager

from glob import glob
import os
import shutil
import tempfile

import numpy as np
# Use this to create parameterized test cases
from parameterized import parameterized
import tensorflow as tf

from sparkdl.graph.input import *
import sparkdl.graph.utils as tfx

from ..tests import PythonUnitTestCase


class TestGenBase(object):
    def __init__(self, vec_size=17, test_batch_size=231):
        # Testing data spec
        self.vec_size = vec_size
        self.test_batch_size = test_batch_size

        self.input_col = 'dfInputCol'
        self.input_op_name = 'tnsrOpIn'
        self.output_col = 'dfOutputCol'
        self.output_op_name = 'tnsrOpOut'

        self.feed_names = []
        self.fetch_names = []
        self.input_mapping = {}
        self.output_mapping = {}
        self.reset_iomap(replica=1)

        self.test_cases = []
        self.input_graphs = []
        # Build a temporary directory, which might or might not be used by the test
        self.saved_model_root = tempfile.mkdtemp()
        self.checkpoint_root = tempfile.mkdtemp()

    def tear_down_env(self):
        shutil.rmtree(self.saved_model_root, ignore_errors=True)
        shutil.rmtree(self.checkpoint_root, ignore_errors=True)

    def reset_iomap(self, replica=1):
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
    def prep_tf_session(self):
        """ Create a session to let build testing graphs

        Downstream classes could also choose to override this function to
        build custom testing behaviors
        """

        # Build the TensorFlow graph
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess, graph.as_default():
            # Build test graph and transformers from here
            # Notice that any TensorFlow graph could potentially be constructed in this session.
            # The graph might contain variables only usable in this session.
            yield sess

            # Find the input/output tensors. We expect them to use canonical names.
            ref_feed = tfx.get_tensor(self.input_op_name, graph)
            ref_fetch = tfx.get_tensor(self.output_op_name, graph)

            def create_test_result(tgt_gdef, test_idx):
                namespace = 'TEST_TGT_NS{:03d}'.format(test_idx)
                tf.import_graph_def(tgt_gdef, name=namespace)
                tgt_feed = tfx.get_tensor('{}/{}'.format(namespace, self.input_op_name), graph)
                tgt_fetch = tfx.get_tensor('{}/{}'.format(namespace, self.output_op_name), graph)

                local_data = np.random.randn(self.test_batch_size, self.vec_size)
                ref_out = sess.run(ref_fetch, feed_dict={ref_feed: local_data})
                # Run on the testing target
                tgt_out = sess.run(tgt_fetch, feed_dict={tgt_feed: local_data})

                return ref_out, tgt_out

            for test_idx, input_graph in enumerate(self.input_graphs):
                res = create_test_result(input_graph.graph_def, test_idx)
                self.test_cases.append(res)

            # Cleanup the result for next rounds
            self.input_graphs = []

    def register(self, tf_input_graph):
        self.input_graphs.append(tf_input_graph)

    def build_input_graphs(self):
        raise NotImplementedError("build your graph and test cases here")


class GenTestCases(TestGenBase):

    def build_input_graphs(self):
        self.build_from_checkpoint()
        self.build_from_graph()
        self.build_from_saved_model()

    def build_from_graph(self):
        """ Build TFTransformer from tf.Graph """
        with self.prep_tf_session() as sess:
            # Begin building graph
            x = tf.placeholder(tf.float64, shape=[None, self.vec_size], name=self.input_op_name)
            _ = tf.reduce_mean(x, axis=1, name=self.output_op_name)
            # End building graph

            self.register(
                TFInputGraph.fromGraph(sess.graph, sess, self.feed_names, self.fetch_names))
            self.register(
                TFInputGraph.fromGraphDef(sess.graph.as_graph_def(), self.feed_names,
                                          self.fetch_names))

    def build_from_saved_model(self):
        """ Build TFTransformer from saved model """
        # Setup saved model export directory
        saved_model_dir = os.path.join(self.saved_model_root, 'saved_model')
        serving_tag = "serving_tag"
        serving_sigdef_key = 'prediction_signature'
        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)

        with self.prep_tf_session() as sess:
            # Model definition: begin
            x = tf.placeholder(tf.float64, shape=[None, self.vec_size], name=self.input_op_name)
            w = tf.Variable(
                tf.random_normal([self.vec_size], dtype=tf.float64), dtype=tf.float64, name='varW')
            z = tf.reduce_mean(x * w, axis=1, name=self.output_op_name)
            # Model definition ends

            sess.run(w.initializer)

            sig_inputs = {'input_sig': tf.saved_model.utils.build_tensor_info(x)}
            sig_outputs = {'output_sig': tf.saved_model.utils.build_tensor_info(z)}

            serving_sigdef = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=sig_inputs, outputs=sig_outputs)

            builder.add_meta_graph_and_variables(
                sess,
                [serving_tag], signature_def_map={serving_sigdef_key: serving_sigdef})
            builder.save()

            # Build the transformer from exported serving model
            # We are using signatures, thus must provide the keys
            gin = TFInputGraph.fromSavedModelWithSignature(saved_model_dir, serving_tag,
                                                           serving_sigdef_key)
            self.register(gin)

            # Build the transformer from exported serving model
            # We are not using signatures, thus must provide tensor/operation names
            gin = TFInputGraph.fromSavedModel(saved_model_dir, serving_tag, self.feed_names,
                                              self.fetch_names)
            self.register(gin)

            gin = TFInputGraph.fromGraph(sess.graph, sess, self.feed_names, self.fetch_names)
            self.register(gin)


    def build_from_checkpoint(self):
        """ Build TFTransformer from a model checkpoint """
        # Build the TensorFlow graph
        model_ckpt_dir = self.checkpoint_root
        ckpt_path_prefix = os.path.join(model_ckpt_dir, 'model_ckpt')
        serving_sigdef_key = 'prediction_signature'

        with self.prep_tf_session() as sess:
            x = tf.placeholder(tf.float64, shape=[None, self.vec_size], name=self.input_op_name)
            #x = tf.placeholder(tf.float64, shape=[None, vec_size], name=input_col)
            w = tf.Variable(
                tf.random_normal([self.vec_size], dtype=tf.float64), dtype=tf.float64, name='varW')
            z = tf.reduce_mean(x * w, axis=1, name=self.output_op_name)
            sess.run(w.initializer)
            saver = tf.train.Saver(var_list=[w])
            _ = saver.save(sess, ckpt_path_prefix, global_step=2702)

            # Prepare the signature_def
            serving_sigdef = tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'input_sig': tf.saved_model.utils.build_tensor_info(x)},
                outputs={'output_sig': tf.saved_model.utils.build_tensor_info(z)})

            # A rather contrived way to add signature def to a meta_graph
            meta_graph_def = tf.train.export_meta_graph()

            # Find the meta_graph file (there should be only one)
            _ckpt_meta_fpaths = glob('{}/*.meta'.format(model_ckpt_dir))
            assert len(_ckpt_meta_fpaths) == 1, \
                'expected only one meta graph, but got {}'.format(','.join(_ckpt_meta_fpaths))
            ckpt_meta_fpath = _ckpt_meta_fpaths[0]

            # Add signature_def to the meta_graph and serialize it
            # This will overwrite the existing meta_graph_def file
            meta_graph_def.signature_def[serving_sigdef_key].CopyFrom(serving_sigdef)
            with open(ckpt_meta_fpath, mode='wb') as fout:
                fout.write(meta_graph_def.SerializeToString())

            # Build the transformer from exported serving model
            # We are using signaures, thus must provide the keys
            gin = TFInputGraph.fromCheckpointWithSignature(model_ckpt_dir, serving_sigdef_key)
            self.register(gin)

            # Transformer without using signature_def
            gin = TFInputGraph.fromCheckpoint(model_ckpt_dir, self.feed_names, self.fetch_names)
            self.register(gin)

            gin = TFInputGraph.fromGraph(sess.graph, sess, self.feed_names, self.fetch_names)
            self.register(gin)


TestCase = namedtuple('TestCase', ['ref_out', 'tgt_out', 'description'])

_TEST_CASES_GENERATORS = []


def register(obj):
    _TEST_CASES_GENERATORS.append(obj)


#========================================================================
# Register all test objects here
register(GenTestCases(vec_size=23, test_batch_size=71))
register(GenTestCases(vec_size=13, test_batch_size=71))
register(GenTestCases(vec_size=5, test_batch_size=71))
#========================================================================

_ALL_TEST_CASES = []
for obj in _TEST_CASES_GENERATORS:
    obj.build_input_graphs()
    for ref_out, tgt_out in obj.test_cases:
        test_case = TestCase(ref_out=ref_out, tgt_out=tgt_out, description=type(obj))
        _ALL_TEST_CASES.append(test_case)
    obj.tear_down_env()


class TFInputGraphTest(PythonUnitTestCase):
    @parameterized.expand(_ALL_TEST_CASES)
    def test_tf_input_graph(self, ref_out, tgt_out, description):
        """ Test build TFInputGraph from various methods """
        self.assertTrue(np.allclose(ref_out, tgt_out), msg=description)
