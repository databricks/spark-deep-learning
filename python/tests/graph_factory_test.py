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

from sparkdl.image.imageIO import imageToStruct, SparkMode
from sparkdl.graph_builder import GraphBuilderSession
from sparkdl.graph_factory import GraphFunctionFactory as factory
from .tests import SparkDLTestCase
from .transformers.image_utils import _getSampleJPEGDir, getSampleImagePathsDF


class GraphFactoryTest(SparkDLTestCase):

    
    def test_spimage_converter_module(self):
        """ spimage converter module must preserve original image """
        img_fpaths = glob(os.path.join(_getSampleJPEGDir(), '*.jpg'))
        
        def convert_image(spimg_dict, img_dtype):
            gfn = factory.build_spimage_converter(img_dtype)
            with GraphBuilderSession() as builder:
                feeds, fetches = builder.import_graph_function(gfn, name="")            
                feed_dict = dict((tnsr, spimg_dict[builder.op_name(tnsr)]) for tnsr in feeds)
                img_out = builder.sess.run(fetches[0], feed_dict=feed_dict)
            return img_out
            
        def check_image_round_trip(img_arr):
            spimg_dict = imageToStruct(img_arr).asDict()
            spimg_dict['data'] = bytes(spimg_dict['data'])
            img_arr_out = convert_image(spimg_dict, spimg_dict['mode'])
            self.assertTrue(np.all(img_arr_out == img_arr))

        for fp in img_fpaths:            
            img = load_img(fp)

            img_arr_byte = img_to_array(img).astype(np.uint8)
            check_image_round_trip(img_arr_byte)

            img_arr_float = img_to_array(img).astype(np.float)
            check_image_round_trip(img_arr_float)

            img_arr_preproc = iv3.preprocess_input(img_to_array(img))
            check_image_round_trip(img_arr_preproc)
