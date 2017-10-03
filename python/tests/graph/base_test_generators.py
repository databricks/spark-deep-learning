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
import itertools
import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf

from sparkdl.graph.input import *
import sparkdl.graph.utils as tfx

__all__ = ['TestCase', 'GenTestCases', 'TestFn']

TestCase = namedtuple('TestCase', ['bool_result', 'err_msg'])
TestFn = namedtuple('TestFn', ['test_fn', 'description', 'metadata'])
_GinInfo = namedtuple('_GinInfo', ['gin', 'description'])


class TestGenBase(object):
    def __init__(self, vec_size=17, test_batch_size=231):
        # Testing data spec
        self.vec_size = vec_size
        self.test_batch_size = test_batch_size

        # TensorFlow graph element names
        self.input_op_name = 'tnsrOpIn'
        self.feed_names = []
        self.output_op_name = 'tnsrOpOut'
        self.fetch_names = []

        # Serving signatures
        self.serving_tag = "serving_tag"
        self.serving_sigdef_key = 'prediction_signature'
        self.input_sig_name = 'wellKnownInputSig'
        self.output_sig_name = 'wellKnownOutputSig'
        self.sig_input_mapping = {}
        self.sig_output_mapping = {}

        # DataFrame column names
        self.input_col = 'dfInputCol'
        self.output_col = 'dfOutputCol'

        # Connecting data from Spark to TensorFlow
        self.input_mapping = {}
        self.output_mapping = {}

        # # When testing against multiple graph inputs,
        # # derive new names for the DataFrame columns and TensorFlow graph elements.
        # self.reset_iomap(replica=1)

        # The basic stage contains the opaque :py:obj:`TFInputGraph` objects
        # Any derived that override the :py:obj:`build_input_graphs` method will
        # populate this field.
        self.input_graphs = []
        self.input_graph_with_signature = set()

        # Construct final test cases, which will be passed to final test cases
        self.test_cases = []

        # Build a temporary directory, which might or might not be used by the test
        self._temp_dirs = []

    def make_tempdir(self):
        """ Create temp directories using this function.
        At the end of this test object's life cycle, the temporary directories
        will all be cleaned up.
        """
        tmp_dir = tempfile.mkdtemp()
        self._temp_dirs.append(tmp_dir)
        return tmp_dir

    def tear_down_env(self):
        for tmp_dir in self._temp_dirs:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def reset_iomap(self, replica=1):
        self.input_mapping = {}
        self.sig_input_mapping = {}
        self.feed_names = []
        self.output_mapping = {}
        self.sig_output_mapping = {}
        self.fetch_names = []

        if replica > 1:
            _template = '{}_replica{:03d}'
            for i in range(replica):
                colname = _template.format(self.input_col, i)
                op_name = _template.format(self.input_op_name, i)
                sig_name = _template.format(self.input_sig_name, i)
                tnsr_name = tfx.tensor_name(op_name)
                self.input_mapping[colname] = tnsr_name
                self.feed_names.append(tnsr_name)
                self.sig_input_mapping[colname] = sig_name

                colname = _template.format(self.output_col, i)
                op_name = _template.format(self.output_op_name, i)
                sig_name = _template.format(self.output_sig_name, i)
                tnsr_name = tfx.tensor_name(op_name)
                self.output_mapping[tnsr_name] = colname
                self.fetch_names.append(tnsr_name)
                self.sig_output_mapping[sig_name] = colname
        else:
            self.input_mapping = {self.input_col: tfx.tensor_name(self.input_op_name)}
            self.sig_input_mapping = {self.input_col: self.input_sig_name}
            self.feed_names = [tfx.tensor_name(self.input_op_name)]
            self.output_mapping = {tfx.tensor_name(self.output_op_name): self.output_col}
            self.sig_output_mapping = {self.output_sig_name: self.output_col}
            self.fetch_names = [tfx.tensor_name(self.output_op_name)]

    @contextmanager
    def prep_tf_session(self, io_replica=1):
        """ Create a session to let build testing graphs

        Downstream classes could also choose to override this function to
        build custom testing behaviors
        """
        # Reset states
        # In each `prep_tf_session`, the implementation is expected to define ONE graph and
        # pass all test cases derived from it. We execute the graph and compare the result
        # with each test case, and return the numerical results.
        self.reset_iomap(replica=io_replica)
        self.input_graphs = []
        self.input_graph_with_signature.clear()

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

            # Build test data and reference results
            test_data = np.random.randn(self.test_batch_size, self.vec_size)
            ref_out = sess.run(ref_fetch, feed_dict={ref_feed: test_data})

        for gin_info in self.input_graphs:
            graph_def = gin_info.gin.graph_def
            description = gin_info.description
            input_op_name = self.input_op_name
            output_op_name = self.output_op_name

            def gen_input_graph_test_case():
                graph = tf.Graph()
                with tf.Session(graph=graph) as sess:
                    namespace = 'TEST_TF_INPUT_GRAPH'
                    tf.import_graph_def(graph_def, name=namespace)
                    tgt_feed = tfx.get_tensor('{}/{}'.format(namespace, input_op_name), graph)
                    tgt_fetch = tfx.get_tensor('{}/{}'.format(namespace, output_op_name), graph)
                    # Run on the testing target
                    tgt_out = sess.run(tgt_fetch, feed_dict={tgt_feed: test_data})

                # Uncomment to check if test cases work in parallel
                # if np.random(1) < 0.3:
                #     raise RuntimeError('randomly killing tests')

                max_diff = np.max(np.abs(ref_out - tgt_out))
                err_msg = '{}: max abs diff {}'.format(description, max_diff)
                return TestCase(bool_result=np.allclose(ref_out, tgt_out), err_msg=err_msg)

            test_case = TestFn(test_fn=gen_input_graph_test_case,
                               description=description, metadata={})
            self.test_cases.append(test_case)

    def register(self, gin, description):
        self.input_graphs.append(_GinInfo(gin=gin, description=description))

    def build_input_graphs(self):
        raise NotImplementedError("build your graph and test cases here")


class GenTestCases(TestGenBase):
    """ Define various test graphs

    Please define all the graphs to be built for test cases in this class.
    This is useful for other classes to inherent and change the definition of
    the actual testing method. That is, by overriding :py:meth:`prep_tf_session`,
    the subclass can change the evaluation behavior.
    """

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

            gin = TFInputGraph.fromGraph(sess.graph, sess, self.feed_names, self.fetch_names)
            self.register(gin=gin, description='from graph with graph')
            gin = TFInputGraph.fromGraphDef(sess.graph.as_graph_def(), self.feed_names,
                                            self.fetch_names)
            self.register(gin=gin, description='from graph with graph_def')

    def build_from_saved_model(self):
        """ Build TFTransformer from saved model """
        # Setup saved model export directory
        saved_model_dir = os.path.join(self.make_tempdir(), 'saved_model')
        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)

        with self.prep_tf_session() as sess:
            # Model definition: begin
            x = tf.placeholder(tf.float64, shape=[None, self.vec_size], name=self.input_op_name)
            w = tf.Variable(
                tf.random_normal([self.vec_size], dtype=tf.float64), dtype=tf.float64, name='varW')
            z = tf.reduce_mean(x * w, axis=1, name=self.output_op_name)
            # Model definition ends

            sess.run(w.initializer)

            sig_inputs = {self.input_sig_name: tf.saved_model.utils.build_tensor_info(x)}
            sig_outputs = {self.output_sig_name: tf.saved_model.utils.build_tensor_info(z)}

            serving_sigdef = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=sig_inputs, outputs=sig_outputs)

            builder.add_meta_graph_and_variables(
                sess, [self.serving_tag], signature_def_map={self.serving_sigdef_key: serving_sigdef})
            builder.save()

            # Build the transformer from exported serving model
            # We are using signatures, thus must provide the keys
            gin = TFInputGraph.fromSavedModelWithSignature(saved_model_dir, self.serving_tag,
                                                           self.serving_sigdef_key)
            self.register(gin=gin, description='saved model with signature')

            _imap = {self.input_sig_name: tfx.tensor_name(x)}
            _omap = {self.output_sig_name: tfx.tensor_name(z)}
            self._add_signature_tensor_name_test_cases(gin, _imap, _omap)

            # Build the transformer from exported serving model
            # We are not using signatures, thus must provide tensor/operation names
            gin = TFInputGraph.fromSavedModel(saved_model_dir, self.serving_tag, self.feed_names,
                                              self.fetch_names)
            self.register(gin=gin, description='saved model no signature')

            gin = TFInputGraph.fromGraph(sess.graph, sess, self.feed_names, self.fetch_names)
            self.register(gin=gin, description='saved model with graph')

    def build_from_checkpoint(self):
        """ Build TFTransformer from a model checkpoint """
        # Build the TensorFlow graph
        model_ckpt_dir = self.make_tempdir()
        ckpt_path_prefix = os.path.join(model_ckpt_dir, 'model_ckpt')

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
                inputs={self.input_sig_name: tf.saved_model.utils.build_tensor_info(x)},
                outputs={self.output_sig_name: tf.saved_model.utils.build_tensor_info(z)})

            # A rather contrived way to add signature def to a meta_graph
            meta_graph_def = tf.train.export_meta_graph()

            # Find the meta_graph file (there should be only one)
            _ckpt_meta_fpaths = glob('{}/*.meta'.format(model_ckpt_dir))
            assert len(_ckpt_meta_fpaths) == 1, \
                'expected only one meta graph, but got {}'.format(','.join(_ckpt_meta_fpaths))
            ckpt_meta_fpath = _ckpt_meta_fpaths[0]

            # Add signature_def to the meta_graph and serialize it
            # This will overwrite the existing meta_graph_def file
            meta_graph_def.signature_def[self.serving_sigdef_key].CopyFrom(serving_sigdef)
            with open(ckpt_meta_fpath, mode='wb') as fout:
                fout.write(meta_graph_def.SerializeToString())

            # Build the transformer from exported serving model
            # We are using signaures, thus must provide the keys
            gin = TFInputGraph.fromCheckpointWithSignature(model_ckpt_dir, self.serving_sigdef_key)
            self.register(gin=gin, description='checkpoint with signature')

            _imap = {self.input_sig_name: tfx.tensor_name(x)}
            _omap = {self.output_sig_name: tfx.tensor_name(z)}
            self._add_signature_tensor_name_test_cases(gin, _imap, _omap)

            # Transformer without using signature_def
            gin = TFInputGraph.fromCheckpoint(model_ckpt_dir, self.feed_names, self.fetch_names)
            self.register(gin=gin, description='checkpoint no signature')

            gin = TFInputGraph.fromGraph(sess.graph, sess, self.feed_names, self.fetch_names)
            self.register(gin=gin, description='checkpoing, graph')

    def _add_signature_tensor_name_test_cases(self, gin, input_sig2tnsr, output_sig2tnsr):
        """ Add tests for checking signature to tensor names mapping """
        imap_tgt = gin.input_tensor_name_from_signature
        description = 'test input signature to tensor name mapping'

        def check_imap():
            err_msg = '{}: {} != {}'.format(description, imap_tgt, input_sig2tnsr)
            bool_result = input_sig2tnsr == imap_tgt
            return TestCase(bool_result=bool_result, err_msg=err_msg)

        self.test_cases.append(TestFn(test_fn=check_imap, description=description, metadata={}))

        omap_tgt = gin.output_tensor_name_from_signature
        description = 'test output signature to tensor name mapping'

        def check_omap():
            err_msg = '{}: {} != {}'.format(description, omap_tgt, output_sig2tnsr)
            bool_result = output_sig2tnsr == omap_tgt
            return TestCase(bool_result=bool_result, err_msg=err_msg)

        self.input_graph_with_signature.add(gin)
        self.test_cases.append(TestFn(test_fn=check_omap, description=description, metadata={}))
