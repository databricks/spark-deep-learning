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

from keras.layers import Conv1D, Dense, Flatten, MaxPool1D
import numpy as np
from parameterized import parameterized
import tensorflow as tf
import tensorframes as tfs

from pyspark.sql.types import Row

from sparkdl.graph.builder import IsolatedSession
from sparkdl.graph.input import *
import sparkdl.graph.utils as tfx
from sparkdl.transformers.tf_tensor import TFTransformer

from ..tests import SparkDLTestCase, TestSparkContext
from ..graph.base_test_generators import *

class GenTFTransformerTestCases(GenTestCases):

    def __init__(self, vec_size=17, test_batch_size=231):
        super(GenTFTransformerTestCases, self).__init__(vec_size, test_batch_size)
        self.sig_input_mapping = None
        self.sig_output_mapping = None
        self.all_close_tol = 1e-8

    def build_input_graphs(self):
        super(GenTFTransformerTestCases, self).build_input_graphs()
        self.build_mixed_keras_graph()
        self.build_multi_io()

    @contextmanager
    def prep_tf_session(self, io_replica=1):
        """ [THIS IS NOT A TEST]: encapsulate general test workflow """
        self.reset_iomap(replica=io_replica)
        self.input_graphs = []
        self.input_graph_with_signature.clear()

        # Build local features and DataFrame from it
        local_features = []
        for idx in range(self.test_batch_size):
            _dict = {'idx': idx}
            for colname, _ in self.input_mapping.items():
                _dict[colname] = np.random.randn(self.vec_size).tolist()

            local_features.append(Row(**_dict))

        # Build the TensorFlow graph
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess, graph.as_default():
            # Build test graph and transformers from here
            yield sess

            # Get the reference data
            _results = []
            for row in local_features:
                fetches = [tfx.get_tensor(tnsr_name, graph)
                           for tnsr_name, _ in self.output_mapping.items()]
                feed_dict = {}
                for colname, tnsr_name in self.input_mapping.items():
                    tnsr = tfx.get_tensor(tnsr_name, graph)
                    feed_dict[tnsr] = np.array(row[colname])[np.newaxis, :]

                curr_res = sess.run(fetches, feed_dict=feed_dict)
                _results.append(np.ravel(curr_res))

            out_ref = np.hstack(_results)

        # Must make sure the function does not depend on `self`
        tnsr2col_mapping = self.output_mapping
        all_close_tol = self.all_close_tol

        # We have sessions, now create transformers out of them
        def check_transformer(transformer, create_dataframe_fn):
            df = create_dataframe_fn(local_features)
            analyzed_df = tfs.analyze(df)
            out_df = transformer.transform(analyzed_df)

            # Collect transformed values
            out_colnames = list(tnsr2col_mapping.values())
            _results = []
            for row in out_df.select(out_colnames).collect():
                curr_res = [row[colname] for colname in out_colnames]
                _results.append(np.ravel(curr_res))
            out_tgt = np.hstack(_results)

            _err_msg = 'not close => shape {} != {}, max_diff {} > {}'
            max_diff = np.max(np.abs(out_ref - out_tgt))
            err_msg = _err_msg.format(out_ref.shape, out_tgt.shape,
                                      max_diff, all_close_tol)
            bool_result = np.allclose(out_ref, out_tgt, atol=all_close_tol)
            return TestCase(bool_result=bool_result, err_msg=err_msg)


        # Apply the transform
        for gin_info in self.input_graphs:
            input_graph = gin_info.gin
            transformer = TFTransformer(tfInputGraph=input_graph,
                                        inputMapping=self.input_mapping,
                                        outputMapping=self.output_mapping)

            description = '{} for TFTransformer'.format(gin_info.description)
            self.test_cases.append(TestFn(test_fn=lambda fn: check_transformer(transformer, fn),
                                          description=description,
                                          metadata=dict(need_dataframe=True)))

            if input_graph in self.input_graph_with_signature:
                _imap = input_graph.translateInputMapping(self.sig_input_mapping)
                _omap = input_graph.translateOutputMapping(self.sig_output_mapping)
                transformer = TFTransformer(tfInputGraph=input_graph,
                                            inputMapping=_imap,
                                            outputMapping=_omap)
                self.test_cases.append(TestFn(test_fn=lambda fn: check_transformer(transformer, fn),
                                              description='dunno 2',
                                              metadata=dict(need_dataframe=True)))

    def build_multi_io(self):
        """ Build TFTransformer with multiple I/O tensors """
        with self.prep_tf_session(io_replica=3) as sess:
            xs = []
            for tnsr_name in self.input_mapping.values():
                x = tf.placeholder(tf.float64, shape=[None, self.vec_size],
                                   name=tfx.op_name(tnsr_name))
                xs.append(x)

            zs = []
            for i, tnsr_name in enumerate(self.output_mapping.keys()):
                z = tf.reduce_mean(xs[i], axis=1, name=tfx.op_name(tnsr_name))
                zs.append(z)

            gin = TFInputGraph.fromGraph(sess.graph, sess, self.feed_names, self.fetch_names)
            description = 'TFInputGraph with multiple input/output tensors'
            self.register(gin=gin, description=description)


    def build_mixed_keras_graph(self):
        """ Build TFTransformer from mixed keras graph """
        with IsolatedSession(using_keras=True) as issn:
            tnsr_in = tf.placeholder(
                tf.double, shape=[None, self.vec_size], name=self.input_op_name)
            inp = tf.expand_dims(tnsr_in, axis=2)
            # Keras layers does not take tf.double
            inp = tf.cast(inp, tf.float32)
            conv = Conv1D(filters=4, kernel_size=2)(inp)
            pool = MaxPool1D(pool_size=2)(conv)
            flat = Flatten()(pool)
            dense = Dense(1)(flat)
            # We must keep the leading dimension of the output
            redsum = tf.reduce_logsumexp(dense, axis=1)
            tnsr_out = tf.cast(redsum, tf.double, name=self.output_op_name)

            # Initialize the variables
            init_op = tf.global_variables_initializer()
            issn.run(init_op)
            # We could train the model ... but skip it here
            gfn = issn.asGraphFunction([tnsr_in], [tnsr_out])

        self.all_close_tol = 1e-5
        gin = TFInputGraph.fromGraphDef(gfn.graph_def, self.feed_names, self.fetch_names)
        self.input_graphs.append(gin)

        with self.prep_tf_session() as sess:
            tf.import_graph_def(gfn.graph_def, name='')
            gin = TFInputGraph.fromGraph(sess.graph, sess, self.feed_names, self.fetch_names)
            description = 'TFInputGraph from Keras'
            self.register(gin=gin, description=description)



_TEST_CASES_GENERATORS = []


def _REGISTER_(obj):
    _TEST_CASES_GENERATORS.append(obj)


#========================================================================
# Register all test objects here
_REGISTER_(GenTFTransformerTestCases(vec_size=23, test_batch_size=71))
_REGISTER_(GenTFTransformerTestCases(vec_size=13, test_batch_size=23))
_REGISTER_(GenTFTransformerTestCases(vec_size=5, test_batch_size=17))
#========================================================================

_ALL_TEST_CASES = []
_CLEAN_UP_TASKS = []

for obj in _TEST_CASES_GENERATORS:
    obj.build_input_graphs()
    _ALL_TEST_CASES += obj.test_cases
    _CLEAN_UP_TASKS.append(obj.tear_down_env)


class TFTransformerTests(SparkDLTestCase):
    @classmethod
    def tearDownClass(cls):
        for clean_fn in _CLEAN_UP_TASKS:
            clean_fn()

    @parameterized.expand(_ALL_TEST_CASES)
    def test_tf_transformers(self, test_fn, description, metadata):  # pylint: disable=unused-argument
        """ Test build TFInputGraph from various sources """
        if metadata and metadata['need_dataframe']:
            bool_result, err_msg = test_fn(lambda xs: self.session.createDataFrame(xs))
        else:
            bool_result, err_msg = test_fn()

        self.assertTrue(bool_result, msg=err_msg)
