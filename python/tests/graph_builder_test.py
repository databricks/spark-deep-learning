#
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

from __future__ import print_function

from glob import glob
import os

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.applications import InceptionV3
from keras.applications import inception_v3 as iv3
from keras.preprocessing.image import load_img, img_to_array

from pyspark import SparkContext
from pyspark.sql import DataFrame, Row
from pyspark.sql.functions import udf

from sparkdl.graph_builder import GraphBuilderSession
from .tests import SparkDLTestCase
from .transformers.image_utils import _getSampleJPEGDir, getSampleImagePathsDF


class GraphBuilderTest(SparkDLTestCase):

    def test_tf_consistency(self):
        """ Should get the same graph as running pure tf """

        x_val = 2702.142857
        g = tf.Graph()
        with tf.Session(graph=g) as sess:
            x = tf.placeholder(tf.double, shape=[], name="x")
            z = tf.add(x, 3, name='z')
            gdef_ref = g.as_graph_def(add_shapes=True)
            z_ref = sess.run(z, {x: x_val})

        with GraphBuilderSession() as builder:
            x = tf.placeholder(tf.double, shape=[], name="x")
            z = tf.add(x, 3, name='z')
            z_tgt = builder.sess.run(z, {x: x_val})
            gfn = builder.asGraphFunction([x], [z])

        self.assertEqual(z_ref, z_tgt)

        # Version texts are not essential part of the graph, ignore them
        gdef_ref.ClearField("versions")
        gfn.graph_def.ClearField("versions")

        # The GraphDef contained in the GraphFunction object
        # should be the same as that in the one exported directly from TensorFlow session
        self.assertEqual(str(gfn.graph_def), str(gdef_ref))


    def test_get_graph_elemets(self):
        """ Fetching graph elemetns by names and other graph element objects """

        with GraphBuilderSession() as builder:
            x = tf.placeholder(tf.double, shape=[], name="x")
            z = tf.add(x, 3, name='z')

            g = builder.graph
            self.assertEqual(builder.get_tensor(z), z)
            self.assertEqual(builder.get_tensor(x), x)
            self.assertEqual(g.get_tensor_by_name("x:0"), builder.get_tensor(x))
            self.assertEqual("x:0", builder.tensor_name(x))
            self.assertEqual(g.get_operation_by_name("x"), builder.get_op(x))
            self.assertEqual("x", builder.op_name(x))
            self.assertEqual("x", builder.op_name(x))
            self.assertEqual("z", builder.op_name(z))
            self.assertEqual(builder.get_tensor(z), z)
            self.assertEqual(builder.get_tensor(x), x)


    def test_import_export_graph_function(self):
        """ Function import and export must be consistent """

        with GraphBuilderSession() as builder:
            x = tf.placeholder(tf.double, shape=[], name="x")
            z = tf.add(x, 3, name='z')
            gfn_ref = builder.asGraphFunction([x], [z])

        with GraphBuilderSession() as builder:
            feeds, fetches = builder.import_graph_function(gfn_ref, name="")
            gfn_tgt = builder.asGraphFunction(feeds, fetches)

        self.assertEqual(gfn_tgt.input_names, gfn_ref.input_names)
        self.assertEqual(gfn_tgt.output_names, gfn_ref.output_names)
        self.assertEqual(str(gfn_tgt.graph_def), str(gfn_ref.graph_def))


    def test_keras_consistency(self):
        """ Exported model in Keras should get same result as original """

        img_fpaths = glob(os.path.join(_getSampleJPEGDir(), '*.jpg'))

        def keras_load_and_preproc(fpath):            
            img = load_img(fpath, target_size=(299, 299))
            img_arr = img_to_array(img)
            img_iv3_input = iv3.preprocess_input(img_arr)
            return np.expand_dims(img_iv3_input, axis=0)

        imgs_iv3_input = np.vstack([keras_load_and_preproc(fp) for fp in img_fpaths])

        model_ref = InceptionV3(weights="imagenet")
        preds_ref = model_ref.predict(imgs_iv3_input)

        with GraphBuilderSession(wrap_keras=True) as builder:
            K.set_learning_phase(0)
            model = InceptionV3(weights="imagenet")
            gfn = builder.asGraphFunction(model.inputs, model.outputs)

        with GraphBuilderSession(wrap_keras=True) as builder:
            K.set_learning_phase(0)
            feeds, fetches = builder.import_graph_function(gfn, name="InceptionV3")
            preds_tgt = builder.sess.run(fetches[0], {feeds[0]: imgs_iv3_input})

        self.assertTrue(np.all(preds_tgt == preds_ref))
