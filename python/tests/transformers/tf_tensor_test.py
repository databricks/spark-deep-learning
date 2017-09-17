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

from sparkdl.graph.input import *
import sparkdl.graph.utils as tfx
from sparkdl.transformers.tf_tensor import TFTransformer

from ..tests import SparkDLTestCase


class TFTransformerTest(SparkDLTestCase):

    def setUp(self):
        self.vec_size = 17
        self.num_vecs = 31

        self.input_col = 'vec'
        self.input_op_name = 'tnsrOpIn'
        self.output_col = 'outputCol'
        self.output_op_name = 'tnsrOpOut'

        self.feed_names = []
        self.fetch_names = []
        self.input_mapping = {}
        self.output_mapping = {}

        self.transformers = []
        self.test_case_results = []
        # Build a temporary directory, which might or might not be used by the test
        self.model_output_root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.model_output_root, ignore_errors=True)

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

    def setup_iomap(self, replica=1):
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

        # Build local features and DataFrame from it
        local_features = []
        for idx in range(self.num_vecs):
            _dict = {'idx': idx}
            for colname, _ in self.input_mapping.items():
                _dict[colname] = np.random.randn(self.vec_size).tolist()

            local_features.append(Row(**_dict))

        df = self.session.createDataFrame(local_features)
        analyzed_df = tfs.analyze(df)

        # Build the TensorFlow graph
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
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

        # Apply the transform
        for transfomer in self.transformers:
            out_df = transfomer.transform(analyzed_df)
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

            self.assertTrue(np.allclose(out_ref, out_tgt),
                            msg=repr(transfomer))


    def test_build_from_tf_graph(self):
        self.setup_iomap(replica=1)
        with self._run_test_in_tf_session() as sess:
            # Begin building graph
            x = tf.placeholder(tf.float64, shape=[None, self.vec_size], name=self.input_op_name)
            _ = tf.reduce_mean(x, axis=1, name=self.output_op_name)
            # End building graph

            # Begin building transformers
            self.build_standard_transformers(
                sess, TFInputGraph.fromGraph(sess.graph, sess, self.feed_names, self.fetch_names))
            gdef = sess.graph.as_graph_def()
            self.build_standard_transformers(
                sess, TFInputGraph.fromGraphDef(gdef, self.feed_names, self.fetch_names))
            # End building transformers


    def test_build_from_saved_model(self):
        self.setup_iomap(replica=1)
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
            tfInputGraph = TFInputGraph.fromSavedModelWithSignature(
                saved_model_dir, serving_tag, serving_sigdef_key)

            inputMapping = tfInputGraph.translateInputMapping({
                self.input_col: 'input_sig'
            })
            outputMapping = tfInputGraph.translateOutputMapping({
                'output_sig': self.output_col
            })
            trans_with_sig = TFTransformer(tfInputGraph=tfInputGraph,
                                           inputMapping=inputMapping,
                                           outputMapping=outputMapping)
            self.transformers.append(trans_with_sig)

            # Build the transformer from exported serving model
            # We are not using signatures, thus must provide tensor/operation names
            gin_builder = TFInputGraph.fromSavedModel(
                saved_model_dir, serving_tag, self.feed_names, self.fetch_names)
            self.build_standard_transformers(sess, gin_builder)


    # def test_build_from_checkpoint(self):
    #     """
    #     Test constructing a Transformer from a TensorFlow training checkpoint
    #     """
    #     # Build the TensorFlow graph
    #     model_ckpt_dir = self.model_output_root
    #     ckpt_path_prefix = os.path.join(model_ckpt_dir, 'model_ckpt')
    #     serving_sigdef_key = 'prediction_signature'
    #     # Warning: please use a new graph for each test cases
    #     #          or the tests could affect one another
    #     with self.run_test_in_tf_session() as sess:
    #         x = tf.placeholder(tf.float64, shape=[None, self.vec_size], name=self.input_op_name)
    #         #x = tf.placeholder(tf.float64, shape=[None, vec_size], name=input_col)
    #         w = tf.Variable(tf.random_normal([self.vec_size], dtype=tf.float64),
    #                         dtype=tf.float64, name='varW')
    #         z = tf.reduce_mean(x * w, axis=1, name=self.output_op_name)
    #         sess.run(w.initializer)
    #         saver = tf.train.Saver(var_list=[w])
    #         _ = saver.save(sess, ckpt_path_prefix, global_step=2702)

    #         # Prepare the signature_def
    #         serving_sigdef = tf.saved_model.signature_def_utils.build_signature_def(
    #             inputs={
    #                 'input_sig': tf.saved_model.utils.build_tensor_info(x)
    #             },
    #             outputs={
    #                 'output_sig': tf.saved_model.utils.build_tensor_info(z)
    #             })

    #         # A rather contrived way to add signature def to a meta_graph
    #         meta_graph_def = tf.train.export_meta_graph()

    #         # Find the meta_graph file (there should be only one)
    #         _ckpt_meta_fpaths = glob('{}/*.meta'.format(model_ckpt_dir))
    #         self.assertEqual(len(_ckpt_meta_fpaths), 1, msg=','.join(_ckpt_meta_fpaths))
    #         ckpt_meta_fpath = _ckpt_meta_fpaths[0]

    #         # Add signature_def to the meta_graph and serialize it
    #         # This will overwrite the existing meta_graph_def file
    #         meta_graph_def.signature_def[serving_sigdef_key].CopyFrom(serving_sigdef)
    #         with open(ckpt_meta_fpath, mode='wb') as fout:
    #             fout.write(meta_graph_def.SerializeToString())

    #         tfInputGraph, inputMapping, outputMapping = get_params_from_checkpoint(
    #             model_ckpt_dir, serving_sigdef_key,
    #             input_mapping={
    #                 self.input_col: 'input_sig'},
    #             output_mapping={
    #                 'output_sig': self.output_col})
    #         trans_with_sig = TFTransformer(tfInputGraph=tfInputGraph,
    #                                        inputMapping=inputMapping,
    #                                        outputMapping=outputMapping)
    #         self.transformers.append(trans_with_sig)

    #         gin_builder = TFInputGraphBuilder.fromCheckpoint(model_ckpt_dir)
    #         self.build_standard_transformers(sess, gin_builder)


    # def test_multi_io(self):
    #     # Build the TensorFlow graph
    #     with self.run_test_in_tf_session(replica=2) as sess:
    #         xs = []
    #         for tnsr_op_name in self.input_mapping.values():
    #             x = tf.placeholder(tf.float64, shape=[None, self.vec_size], name=tnsr_op_name)
    #             xs.append(x)

    #         zs = []
    #         for i, tnsr_op_name in enumerate(self.output_mapping.keys()):
    #             z = tf.reduce_mean(xs[i], axis=1, name=tnsr_op_name)
    #             zs.append(z)

    #         self.build_standard_transformers(sess, sess.graph)
    #         self.build_standard_transformers(sess, TFInputGraphBuilder.fromGraph(sess.graph))

    # def test_mixed_keras_graph(self):
    #     # Build the graph: the output should have the same leading/batch dimension
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
    #         redsum = tf.reduce_sum(dense, axis=1)
    #         tnsr_out = tf.cast(redsum, tf.double, name=self.output_op_name)

    #         # Initialize the variables
    #         init_op = tf.global_variables_initializer()
    #         issn.run(init_op)
    #         # We could train the model ... but skip it here
    #         gfn = issn.asGraphFunction([tnsr_in], [tnsr_out])

    #     with self.run_test_in_tf_session() as sess:
    #         tf.import_graph_def(gfn.graph_def, name='')

    #         self.build_standard_transformers(sess, sess.graph)
    #         self.build_standard_transformers(sess, TFInputGraphBuilder.fromGraph(sess.graph))
    #         self.build_standard_transformers(sess, gfn.graph_def)
    #         self.build_standard_transformers(sess, TFInputGraphBuilder.fromGraphDef(gfn.graph_def))
