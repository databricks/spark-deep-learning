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
import tensorflow as tf
import tensorframes as tfs

from pyspark.sql.types import Row

from sparkdl.graph.builder import IsolatedSession
from sparkdl.graph.input import *
import sparkdl.graph.utils as tfx
from sparkdl.transformers.tf_tensor import TFTransformer

from ..tests import SparkDLTestCase
from ..graph.test_input_graph import TFInputGraphTest


class TFTransformerTest(TFInputGraphTest, SparkDLTestCase):

    def _build_default_session_tests(self, sess):
        gin = TFInputGraph.fromGraph(
            sess.graph, sess, self.feed_names, self.fetch_names)
        self.build_standard_transformers(sess, gin)

        gin = TFInputGraph.fromGraphDef(
            sess.graph.as_graph_def(), self.feed_names, self.fetch_names)
        self.build_standard_transformers(sess, gin)

    def build_standard_transformers(self, sess, tf_input_graph):
        def _add_transformer(imap, omap):
            trnsfmr = TFTransformer(
                tfInputGraph=tf_input_graph, inputMapping=imap, outputMapping=omap)
            self.transformers.append(trnsfmr)

        imap = dict((col, tfx.tensor_name(sess.graph, op_name))
                    for col, op_name in self.input_mapping.items())
        omap = dict((tfx.tensor_name(sess.graph, op_name), col)
                    for op_name, col in self.output_mapping.items())
        _add_transformer(imap, omap)

    @contextmanager
    def _run_test_in_tf_session(self):
        """ [THIS IS NOT A TEST]: encapsulate general test workflow """

        # Build local features and DataFrame from it
        print("OVERRIDING default", repr(self.__class__))
        local_features = []
        for idx in range(self.num_samples):
            _dict = {'idx': idx}
            for colname, _ in self.input_mapping.items():
                _dict[colname] = np.random.randn(self.vec_size).tolist()

            local_features.append(Row(**_dict))

        df = self.session.createDataFrame(local_features)
        analyzed_df = tfs.analyze(df)

        # Build the TensorFlow graph
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess, graph.as_default():
            # Build test graph and transformers from here
            yield sess

            # Get the reference data
            _results = []
            for row in local_features:
                fetches = [tfx.get_tensor(graph, tnsr_op_name)
                           for tnsr_op_name in self.output_mapping.keys()]
                feed_dict = {}
                for colname, tnsr_op_name in self.input_mapping.items():
                    tnsr = tfx.get_tensor(graph, tnsr_op_name)
                    feed_dict[tnsr] = np.array(row[colname])[np.newaxis, :]

                curr_res = sess.run(fetches, feed_dict=feed_dict)
                _results.append(np.ravel(curr_res))

            out_ref = np.hstack(_results)

        # We have sessions, now create transformers out of them

        # Apply the transform
        for input_graph in self.input_graphs:
            transformer = TFTransformer(tfInputGraph=input_graph,
                                        inputMapping=self.input_mapping,
                                        outputMapping=self.output_mapping)
            print('built transformer', repr(transformer))

            out_df = transformer.transform(analyzed_df)
            out_colnames = []
            for old_colname, new_colname in self.output_mapping.items():
                out_colnames.append(new_colname)
                if old_colname != new_colname:
                    out_df = out_df.withColumnRenamed(old_colname, new_colname)

            _results = []
            for row in out_df.select(out_colnames).collect():
                curr_res = [row[colname] for colname in out_colnames]
                _results.append(np.ravel(curr_res))
            out_tgt = np.hstack(_results)

            err_msg = 'not close => {} != {}, max_diff {}'
            self.assertTrue(np.allclose(out_ref, out_tgt),
                            msg=err_msg.format(out_ref.shape, out_tgt.shape,
                                               np.max(np.abs(out_ref - out_tgt))))


    # def test_multi_io(self):
    #     """ Build TFTransformer with multiple I/O tensors """
    #     self.setup_iomap(replica=3)
    #     with self._run_test_in_tf_session() as sess:
    #         xs = []
    #         for tnsr_op_name in self.input_mapping.values():
    #             x = tf.placeholder(tf.float64, shape=[None, self.vec_size], name=tnsr_op_name)
    #             xs.append(x)

    #         zs = []
    #         for i, tnsr_op_name in enumerate(self.output_mapping.keys()):
    #             z = tf.reduce_mean(xs[i], axis=1, name=tnsr_op_name)
    #             zs.append(z)

    #         self._build_default_session_tests(sess)


    # def test_mixed_keras_graph(self):
    #     """ Build mixed keras graph """
    #     with IsolatedSession(using_keras=True) as issn:
    #         tnsr_in = tf.placeholder(
    #             tf.double, shape=[None, self.vec_size], name=self.input_op_name)
    #         inp = tf.expand_dims(tnsr_in, axis=2)
    #         # Keras layers does not take tf.double
    #         inp = tf.cast(inp, tf.float32)
    #         conv = Conv1D(filters=4, kernel_size=2)(inp)
    #         pool = MaxPool1D(pool_size=2)(conv)
    #         flat = Flatten()(pool)
    #         dense = Dense(1)(flat)
    #         # We must keep the leading dimension of the output
    #         redsum = tf.reduce_logsumexp(dense, axis=1)
    #         tnsr_out = tf.cast(redsum, tf.double, name=self.output_op_name)

    #         # Initialize the variables
    #         init_op = tf.global_variables_initializer()
    #         issn.run(init_op)
    #         # We could train the model ... but skip it here
    #         gfn = issn.asGraphFunction([tnsr_in], [tnsr_out])

    #     with self._run_test_in_tf_session() as sess:
    #         tf.import_graph_def(gfn.graph_def, name='')
    #         self._build_default_session_tests(sess)
