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
from tempfile import NamedTemporaryFile

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

from sparkdl.graph.builder import IsolatedSession, GraphFunction
import sparkdl.graph.pieces as gfac
import sparkdl.graph.utils as tfx
from sparkdl.image.imageIO import imageArrayToStruct
from sparkdl.image.imageIO import imageTypeByOrdinal


from ..tests import SparkDLTestCase
from ..transformers.image_utils import _getSampleJPEGDir, getSampleImagePathsDF


class GraphPiecesTest(SparkDLTestCase):
    
    featurizerCompareDigitsExact = 5

    def test_spimage_converter_module(self):
        """ spimage converter module must preserve original image """
        img_fpaths = glob(os.path.join(_getSampleJPEGDir(), '*.jpg'))

        def exec_gfn_spimg_decode(spimg_dict, img_dtype):
            gfn = gfac.buildSpImageConverter('BGR', img_dtype)
            with IsolatedSession() as issn:
                feeds, fetches = issn.importGraphFunction(gfn, prefix="")
                feed_dict = dict(
                    (tnsr, spimg_dict[tfx.op_name(tnsr, issn.graph)]) for tnsr in feeds)
                img_out = issn.run(fetches[0], feed_dict=feed_dict)
            return img_out

        def check_image_round_trip(img_arr):
            spimg_dict = imageArrayToStruct(img_arr).asDict()
            spimg_dict['data'] = bytes(spimg_dict['data'])
            img_arr_out = exec_gfn_spimg_decode(
                spimg_dict, imageTypeByOrdinal(spimg_dict['mode']).dtype)
            self.assertTrue(np.all(img_arr_out == img_arr))

        for fp in img_fpaths:
            img = load_img(fp)

            img_arr_byte = img_to_array(img).astype(np.uint8)
            check_image_round_trip(img_arr_byte)

            img_arr_float = img_to_array(img).astype(np.float32)
            check_image_round_trip(img_arr_float)

            img_arr_preproc = iv3.preprocess_input(img_to_array(img))
            check_image_round_trip(img_arr_preproc)

    def test_identity_module(self):
        """ identity module should preserve input """

        with IsolatedSession() as issn:
            pred_input = tf.placeholder(tf.float32, [None, None])
            final_output = tf.identity(pred_input, name='output')
            gfn = issn.asGraphFunction([pred_input], [final_output])

        for _ in range(10):
            m, n = prng.randint(10, 1000, size=2)
            mat = prng.randn(m, n).astype(np.float32)
            with IsolatedSession() as issn:
                feeds, fetches = issn.importGraphFunction(gfn)
                mat_out = issn.run(fetches[0], {feeds[0]: mat})

            self.assertTrue(np.all(mat_out == mat))

    def test_flattener_module(self):
        """ flattener module should preserve input data """

        gfn = gfac.buildFlattener()
        for _ in range(10):
            m, n = prng.randint(10, 1000, size=2)
            mat = prng.randn(m, n).astype(np.float32)
            with IsolatedSession() as issn:
                feeds, fetches = issn.importGraphFunction(gfn)
                vec_out = issn.run(fetches[0], {feeds[0]: mat})

            self.assertTrue(np.all(vec_out == mat.flatten()))

    def test_bare_keras_module(self):
        """ Keras GraphFunctions should give the same result as standard Keras models """
        img_fpaths = glob(os.path.join(_getSampleJPEGDir(), '*.jpg'))

        for model_gen, preproc_fn, target_size in [(InceptionV3, iv3.preprocess_input, model_sizes['InceptionV3']),
                                      (Xception, xcpt.preprocess_input, model_sizes['Xception']),
                                      (ResNet50, rsnt.preprocess_input, model_sizes['ResNet50'])]:

            keras_model = model_gen(weights="imagenet")
            _preproc_img_list = []
            for fpath in img_fpaths:
                img = load_img(fpath, target_size=target_size)
                # WARNING: must apply expand dimensions first, or ResNet50 preprocessor fails
                img_arr = np.expand_dims(img_to_array(img), axis=0)
                _preproc_img_list.append(preproc_fn(img_arr))

            imgs_input = np.vstack(_preproc_img_list)

            preds_ref = keras_model.predict(imgs_input)

            gfn_bare_keras = GraphFunction.fromKeras(keras_model)

            with IsolatedSession(using_keras=True) as issn:
                K.set_learning_phase(0)
                feeds, fetches = issn.importGraphFunction(gfn_bare_keras)
                preds_tgt = issn.run(fetches[0], {feeds[0]: imgs_input})

            np.testing.assert_array_almost_equal(preds_tgt,
                                                 preds_ref,
                                                 decimal=self.featurizerCompareDigitsExact)

    def test_pipeline(self):
        """ Pipeline should provide correct function composition """
        img_fpaths = glob(os.path.join(_getSampleJPEGDir(), '*.jpg'))

        xcpt_model = Xception(weights="imagenet")
        stages = [('spimage', gfac.buildSpImageConverter('BGR', 'float32')),
                  ('xception', GraphFunction.fromKeras(xcpt_model))]
        piped_model = GraphFunction.fromList(stages)

        for fpath in img_fpaths:
            target_size = model_sizes['Xception']
            img = load_img(fpath, target_size=target_size)
            img_arr = np.expand_dims(img_to_array(img), axis=0)
            img_input = xcpt.preprocess_input(img_arr)
            preds_ref = xcpt_model.predict(img_input)

            spimg_input_dict = imageArrayToStruct(img_input).asDict()
            spimg_input_dict['data'] = bytes(spimg_input_dict['data'])
            with IsolatedSession() as issn:
                # Need blank import scope name so that spimg fields match the input names
                feeds, fetches = issn.importGraphFunction(piped_model, prefix="")
                feed_dict = dict(
                    (tnsr, spimg_input_dict[tfx.op_name(tnsr, issn.graph)]) for tnsr in feeds)
                preds_tgt = issn.run(fetches[0], feed_dict=feed_dict)
                # Uncomment the line below to see the graph
                # tfx.write_visualization_html(issn.graph,
                # NamedTemporaryFile(prefix="gdef", suffix=".html").name)

            np.testing.assert_array_almost_equal(preds_tgt,
                                                 preds_ref,
                                                 decimal=self.featurizerCompareDigitsExact)

model_sizes = {'InceptionV3': (299, 299), 'Xception': (299, 299), 'ResNet50': (224, 224)}
