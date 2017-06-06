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
import numpy.random as prng
import tensorflow as tf
import keras.backend as K
from keras.applications import InceptionV3
from keras.applications import inception_v3 as iv3
from keras.applications import Xception
from keras.applications import xception as xcpt
from keras.applications import ResNet50
from keras.applications import resnet50 as rsnt
from keras.preprocessing.image import load_img, img_to_array

from pyspark import SparkContext
from pyspark.sql import DataFrame, Row
from pyspark.sql.functions import udf

from sparkdl.image.imageIO import imageToStruct, SparkMode
from sparkdl.graph_builder import GraphBuilderSession
from sparkdl.graph_factory import GraphFunctionFactory as factory
from .tests import SparkDLTestCase
from .transformers.image_utils import _getSampleJPEGDir, getSampleImagePathsDF


class GraphFactoryTest(SparkDLTestCase):


    def test_spimage_converter_module(self):
        """ spimage converter module must preserve original image """
        img_fpaths = glob(os.path.join(_getSampleJPEGDir(), '*.jpg'))

        def exec_gfn_spimg_decode(spimg_dict, img_dtype):
            gfn = factory.build_spimage_converter(img_dtype)
            with GraphBuilderSession() as builder:
                feeds, fetches = builder.import_graph_function(gfn, name="")
                feed_dict = dict((tnsr, spimg_dict[builder.op_name(tnsr)]) for tnsr in feeds)
                img_out = builder.sess.run(fetches[0], feed_dict=feed_dict)
            return img_out

        def check_image_round_trip(img_arr):
            spimg_dict = imageToStruct(img_arr).asDict()
            spimg_dict['data'] = bytes(spimg_dict['data'])
            img_arr_out = exec_gfn_spimg_decode(spimg_dict, spimg_dict['mode'])
            self.assertTrue(np.all(img_arr_out == img_arr))

        for fp in img_fpaths:
            img = load_img(fp)

            img_arr_byte = img_to_array(img).astype(np.uint8)
            check_image_round_trip(img_arr_byte)

            img_arr_float = img_to_array(img).astype(np.float)
            check_image_round_trip(img_arr_float)

            img_arr_preproc = iv3.preprocess_input(img_to_array(img))
            check_image_round_trip(img_arr_preproc)

    def test_identity_module(self):
        """ identity module should preserve input """

        gfn = factory.build_identity()
        for _ in range(10):
            m, n = prng.randint(10, 1000, size=2)
            mat = prng.randn(m, n).astype(np.float32)
            with GraphBuilderSession() as builder:
                feeds, fetches = builder.import_graph_function(gfn)
                mat_out = builder.sess.run(fetches[0], {feeds[0]: mat})

            self.assertTrue(np.all(mat_out == mat))

    def test_flattener_module(self):
        """ flattener module should preserve input data """

        gfn = factory.build_flattener()
        for _ in range(10):
            m, n = prng.randint(10, 1000, size=2)
            mat = prng.randn(m, n).astype(np.float32)
            with GraphBuilderSession() as builder:
                feeds, fetches = builder.import_graph_function(gfn)
                vec_out = builder.sess.run(fetches[0], {feeds[0]: mat})

            self.assertTrue(np.all(vec_out == mat.flatten()))

    def test_bare_keras_module(self):
        """ Keras GraphFunctions should give the same result as standard Keras models """
        img_fpaths = glob(os.path.join(_getSampleJPEGDir(), '*.jpg'))

        for model_gen, preproc_fn in [(InceptionV3, iv3.preprocess_input),
                                      (Xception, xcpt.preprocess_input),
                                      (ResNet50, rsnt.preprocess_input)]:

            keras_model = model_gen(weights="imagenet")
            target_size = tuple(keras_model.input.shape.as_list()[1:-1])
            def keras_load_and_preproc(fpath):
                img = load_img(fpath, target_size=target_size)
                # WARNING: must apply expand dimensions first, or ResNet50 preprocessor fails
                img_arr = np.expand_dims(img_to_array(img), axis=0)
                return preproc_fn(img_arr)

            imgs_input = np.vstack([keras_load_and_preproc(fp) for fp in img_fpaths])

            preds_ref = keras_model.predict(imgs_input)

            gfn_bare_keras = factory.import_bare_keras(keras_model)

            with GraphBuilderSession(wrap_keras=True) as builder:
                K.set_learning_phase(0)
                feeds, fetches = builder.import_graph_function(gfn_bare_keras)
                preds_tgt = builder.sess.run(fetches[0], {feeds[0]: imgs_input})

            self.assertTrue(np.all(preds_tgt == preds_ref))

    def test_pipeline(self):
        """ Pipeline should provide correct function composition """
        img_fpaths = glob(os.path.join(_getSampleJPEGDir(), '*.jpg'))

        xcpt_model = Xception(weights="imagenet")
        stages = [('spimage', factory.build_spimage_converter(SparkMode.RGB_FLOAT32)),
                  ('xception', factory.import_bare_keras(xcpt_model))]
        piped_model = factory.pipeline(stages)

        for fpath in img_fpaths:
            target_size = tuple(xcpt_model.input.shape.as_list()[1:-1])
            img = load_img(fpath, target_size=target_size)
            img_arr = np.expand_dims(img_to_array(img), axis=0)
            img_input = xcpt.preprocess_input(img_arr)
            preds_ref = xcpt_model.predict(img_input)

            spimg_input_dict = imageToStruct(img_input).asDict()
            spimg_input_dict['data'] = bytes(spimg_input_dict['data'])
            with GraphBuilderSession() as builder:
                # Need blank import scope name so that spimg fields match the input names
                feeds, fetches = builder.import_graph_function(piped_model, name="")
                feed_dict = dict((tnsr, spimg_input_dict[builder.op_name(tnsr)]) for tnsr in feeds)
                preds_tgt = builder.sess.run(fetches[0], feed_dict=feed_dict)
                # If in REPL, uncomment the line below to see the graph
                # html_code = builder.show_tf_graph()

            self.assertTrue(np.all(preds_tgt == preds_ref))
