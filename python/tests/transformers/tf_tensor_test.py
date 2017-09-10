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

import os
import shutil
import tempfile

from keras.layers import Conv1D, Dense, Flatten, MaxPool1D
import numpy as np
import tensorflow as tf
import tensorframes as tfs

from pyspark.sql.types import Row

from sparkdl.graph.builder import IsolatedSession
import sparkdl.graph.utils as tfx
from sparkdl.transformers.tf_tensor import TFTransformer, TFInputGraphBuilder

from ..tests import SparkDLTestCase

def grab_df_arr(df, output_col):
    """ Stack the numpy array from a DataFrame column """
    return np.array([row.asDict()[output_col]
                     for row in df.select(output_col).collect()])

class TFTransformerTest(SparkDLTestCase):

    def _get_rand_vec_df(self, num_rows, vec_size):
        return self.session.createDataFrame(
            Row(idx=idx, vec=np.random.randn(vec_size).tolist())
            for idx in range(num_rows))

    def test_build_from_tf_graph(self):
        # Build a simple input DataFrame
        vec_size = 17
        num_vecs = 31
        df = self._get_rand_vec_df(num_vecs, vec_size)
        analyzed_df = tfs.analyze(df)

        # Build the TensorFlow graph
        with tf.Session() as sess:
            #x = tf.placeholder(tf.float64, shape=[None, vec_size])
            x = tfs.block(analyzed_df, 'vec')
            z = tf.reduce_mean(x, axis=1)
            graph = sess.graph

            # Get the reference data
            _results = []
            for row in df.collect():
                arr = np.array(row.vec)[np.newaxis, :]
                _results.append(sess.run(z, {x: arr}))
            out_ref = np.hstack(_results)

        # Apply the transform
        gin_from_graph = TFInputGraphBuilder.fromGraph(graph)
        for gin in [gin_from_graph, graph]:
            transfomer = TFTransformer(
                tfInputGraph=TFInputGraphBuilder.fromGraph(graph),
                inputMapping={
                    'vec': x
                },
                outputMapping={
                    z: 'outCol'
                })
            final_df = transfomer.transform(analyzed_df)
            out_tgt = grab_df_arr(final_df, 'outCol')
            self.assertTrue(np.allclose(out_ref, out_tgt))


    def test_build_from_saved_model(self):
        # Setup dataset
        vec_size = 17
        num_vecs = 31
        df = self._get_rand_vec_df(num_vecs, vec_size)
        analyzed_df = tfs.analyze(df)
        input_col = 'vec'
        output_col = 'outputCol'

        # Setup saved model export directory
        saved_model_root = tempfile.mkdtemp()
        saved_model_dir = os.path.join(saved_model_root, 'saved_model')
        serving_tag = "serving_tag"
        serving_sigdef_key = 'prediction_signature'

        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
        with tf.Session(graph=tf.Graph()) as sess:
            # Model definition: begin
            x = tf.placeholder(tf.float64, shape=[None, vec_size], name='tnsrIn')
            #x = tf.placeholder(tf.float64, shape=[None, vec_size], name=input_col)
            w = tf.Variable(tf.random_normal([vec_size], dtype=tf.float64),
                            dtype=tf.float64, name='varW')
            z = tf.reduce_mean(x * w, axis=1, name='tnsrOut')
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
                                                     serving_sigdef_key: serving_sigdef
                                                 })
            # Get the reference data
            _results = []
            for row in df.collect():
                arr = np.array(row.vec)[np.newaxis, :]
                _results.append(sess.run(z, {x: arr}))
            out_ref = np.hstack(_results)

        # Save the model
        builder.save()

        # Build the transformer from exported serving model
        # We are using signaures, thus must provide the keys
        trans_with_sig = TFTransformer(
            tfInputGraph=TFInputGraphBuilder.fromSavedModel(
                saved_model_dir, tag_set=serving_tag, signature_def_key=serving_sigdef_key),
            inputMapping={
                input_col: 'input_sig'
            },
            outputMapping={
                'output_sig': output_col
            })

        # Build the transformer from exported serving model
        # We are not using signatures, thus must provide tensor/operation names
        trans_no_sig = TFTransformer(
            tfInputGraph=TFInputGraphBuilder.fromSavedModel(
                saved_model_dir, tag_set=serving_tag, signature_def_key=None),
            inputMapping={
                input_col: 'tnsrIn'
            },
            outputMapping={
                'tnsrOut': output_col
            })

        df_trans_with_sig = trans_with_sig.transform(analyzed_df)
        df_trans_no_sig = trans_no_sig.transform(analyzed_df)
        out_with_sig_tgt = grab_df_arr(df_trans_with_sig, output_col)
        out_no_sig_tgt = grab_df_arr(df_trans_no_sig, output_col)
        # Cleanup the resources
        shutil.rmtree(saved_model_root, ignore_errors=True)
        self.assertTrue(np.allclose(out_ref, out_with_sig_tgt))
        self.assertTrue(np.allclose(out_ref, out_no_sig_tgt))


    def test_build_from_checkpoint(self):
        vec_size = 17
        num_vecs = 31
        df = self._get_rand_vec_df(num_vecs, vec_size)
        analyzed_df = tfs.analyze(df)
        input_col = 'vec'
        output_col = 'outputCol'

        # Build the TensorFlow graph
        model_ckpt_dir = tempfile.mkdtemp()
        ckpt_path_prefix = os.path.join(model_ckpt_dir, 'model_ckpt')
        # Warning: please use a new graph for each test cases
        #          or the tests could affect one another
        with tf.Session(graph=tf.Graph()) as sess:
            x = tf.placeholder(tf.float64, shape=[None, vec_size], name='tnsrIn')
            #x = tf.placeholder(tf.float64, shape=[None, vec_size], name=input_col)
            w = tf.Variable(tf.random_normal([vec_size], dtype=tf.float64),
                            dtype=tf.float64, name='varW')
            z = tf.reduce_mean(x * w, axis=1, name='tnsrOut')
            sess.run(w.initializer)
            saver = tf.train.Saver(var_list=[w])
            _ = saver.save(sess, ckpt_path_prefix, global_step=2702)

            # Get the reference data
            _results = []
            for row in df.collect():
                arr = np.array(row.vec)[np.newaxis, :]
                _results.append(sess.run(z, {x: arr}))
            out_ref = np.hstack(_results)

        transformer = TFTransformer(
            tfInputGraph=TFInputGraphBuilder.fromCheckpoint(model_ckpt_dir),
            inputMapping={
                input_col: 'tnsrIn'
            },
            outputMapping={
                'tnsrOut': output_col
            })
        final_df = transformer.transform(analyzed_df)
        out_tgt = grab_df_arr(final_df, output_col)

        shutil.rmtree(model_ckpt_dir, ignore_errors=True)
        self.assertTrue(np.allclose(out_ref, out_tgt))


    def test_multi_io(self):
        # Build a simple input DataFrame
        vec_size = 17
        num_vecs = 31
        _df = self._get_rand_vec_df(num_vecs, vec_size)
        df_x = _df.withColumnRenamed('vec', 'vec_x')
        _df = self._get_rand_vec_df(num_vecs, vec_size)
        df_y = _df.withColumnRenamed('vec', 'vec_y')
        df = df_x.join(df_y, on='idx', how='inner')
        analyzed_df = tfs.analyze(df)

        # Build the TensorFlow graph
        with tf.Session() as sess:
            x = tfs.block(analyzed_df, 'vec_x')
            y = tfs.block(analyzed_df, 'vec_y')
            p = tf.reduce_mean(x + y, axis=1)
            q = tf.reduce_mean(x - y, axis=1)
            graph = sess.graph

            # Get the reference data
            p_out_ref = []
            q_out_ref = []
            for row in df.collect():
                arr_x = np.array(row['vec_x'])[np.newaxis, :]
                arr_y = np.array(row['vec_y'])[np.newaxis, :]
                p_val, q_val = sess.run([p, q], {x: arr_x, y: arr_y})
                p_out_ref.append(p_val)
                q_out_ref.append(q_val)
            p_out_ref = np.hstack(p_out_ref)
            q_out_ref = np.hstack(q_out_ref)

        # Apply the transform
        transfomer = TFTransformer(
            tfInputGraph=TFInputGraphBuilder.fromGraph(graph),
            inputMapping={
                'vec_x': x,
                'vec_y': y
            },
            outputMapping={
                p: 'outcol_p',
                q: 'outcol_q'
            })
        final_df = transfomer.transform(analyzed_df)
        p_out_tgt = grab_df_arr(final_df, 'outcol_p')
        q_out_tgt = grab_df_arr(final_df, 'outcol_q')

        self.assertTrue(np.allclose(p_out_ref, p_out_tgt))
        self.assertTrue(np.allclose(q_out_ref, q_out_tgt))

    def test_mixed_keras_graph(self):

        vec_size = 17
        num_vecs = 137
        df = self._get_rand_vec_df(num_vecs, vec_size)
        analyzed_df = tfs.analyze(df)

        input_col = 'vec'
        output_col = 'outCol'

        # Build the graph: the output should have the same leading/batch dimension
        with IsolatedSession(using_keras=True) as issn:
            tnsr_in = tfs.block(analyzed_df, input_col)
            inp = tf.expand_dims(tnsr_in, axis=2)
            # Keras layers does not take tf.double
            inp = tf.cast(inp, tf.float32)
            conv = Conv1D(filters=4, kernel_size=2)(inp)
            pool = MaxPool1D(pool_size=2)(conv)
            flat = Flatten()(pool)
            dense = Dense(1)(flat)
            # We must keep the leading dimension of the output
            redsum = tf.reduce_sum(dense, axis=1)
            tnsr_out = tf.cast(redsum, tf.double, name='TnsrOut')

            # Initialize the variables
            init_op = tf.global_variables_initializer()
            issn.run(init_op)
            # We could train the model ... but skip it here
            gfn = issn.asGraphFunction([tnsr_in], [tnsr_out])

        with IsolatedSession() as issn:
            # Import the graph function object
            feeds, fetches = issn.importGraphFunction(gfn, prefix='')

            # Rename the input column name to the feed op's name
            orig_in_name = tfx.op_name(issn.graph, feeds[0])
            input_df = analyzed_df.withColumnRenamed(input_col, orig_in_name)

            # Do the actual computation
            output_df = tfs.map_blocks(fetches, input_df)

            # Rename the output column (by default, the name of the fetch op's name)
            orig_out_name = tfx.op_name(issn.graph, fetches[0])
            final_df = output_df.withColumnRenamed(orig_out_name, output_col)

        arr_ref = grab_df_arr(final_df, output_col)

        # Using the Transformer
        gin_from_gdef = TFInputGraphBuilder.fromGraphDef(gfn.graph_def)
        for gin in [gin_from_gdef, gfn.graph_def]:
            transformer = TFTransformer(
                tfInputGraph=gin,
                inputMapping={
                    input_col: gfn.input_names[0]
                },
                outputMapping={
                    gfn.output_names[0]: output_col
                })

            transformed_df = transformer.transform(analyzed_df)
            arr_tgt = grab_df_arr(transformed_df, output_col)
            self.assertTrue(np.allclose(arr_ref, arr_tgt))
