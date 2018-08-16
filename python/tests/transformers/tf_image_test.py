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

from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import keras.backend as K
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import tensorflow as tf

import sparkdl.graph.utils as tfx
from sparkdl.image.imageIO import imageStructToArray
from sparkdl.image import imageIO
from sparkdl.transformers.keras_utils import KSessionWrap
from sparkdl.transformers.tf_image import TFImageTransformer
import sparkdl.transformers.utils as utils
from sparkdl.transformers.utils import ImageNetConstants, InceptionV3Constants
from ..tests import SparkDLTestCase
from .image_utils import ImageNetOutputComparisonTestCase
from . import image_utils


class TFImageTransformerExamplesTest(SparkDLTestCase, ImageNetOutputComparisonTestCase):

    # Test loading & pre-processing as an example of a simple graph
    # NOTE: resizing here/tensorflow and in keras workflow are different, so the
    # test would fail with resizing added in.
    def _loadImageViaKeras(self, raw_uri):
        uri = raw_uri[5:] if raw_uri.startswith("file:/") else raw_uri
        image = img_to_array(load_img(uri))
        image = np.expand_dims(image, axis=0)
        return preprocess_input(image)

    def test_load_image_vs_keras(self):
        g = tf.Graph()
        with g.as_default():
            image_arr = utils.imageInputPlaceholder()
            # keras expects array in RGB order, we get it from image schema in BGR => need to flip
            preprocessed = preprocess_input(imageIO._reverseChannels(image_arr))

        output_col = "transformed_image"
        transformer = TFImageTransformer(channelOrder='BGR', inputCol="image", outputCol=output_col, graph=g,
                                         inputTensor=image_arr, outputTensor=preprocessed.name,
                                         outputMode="vector")

        image_df = image_utils.getSampleImageDF()
        df = transformer.transform(image_df.limit(5))

        for row in df.collect():
            processed = np.array(row[output_col]).astype(np.float32)
            # compare to keras loading
            images = self._loadImageViaKeras(row["image"]['origin'])
            image = images[0]
            image.shape = (1, image.shape[0] * image.shape[1] * image.shape[2])
            keras_processed = image[0]
            np.testing.assert_array_almost_equal(keras_processed, processed, decimal=6)

    def test_load_image_vs_keras_RGB(self):
        g = tf.Graph()
        with g.as_default():
            image_arr = utils.imageInputPlaceholder()
            # keras expects array in RGB order, we get it from image schema in BGR => need to flip
            preprocessed = preprocess_input(image_arr)

        output_col = "transformed_image"
        transformer = TFImageTransformer(channelOrder='RGB', inputCol="image", outputCol=output_col, graph=g,
                                         inputTensor=image_arr, outputTensor=preprocessed.name,
                                         outputMode="vector")

        image_df = image_utils.getSampleImageDF()
        df = transformer.transform(image_df.limit(5))

        for row in df.collect():
            processed = np.array(row[output_col], dtype = np.float32)
            # compare to keras loading
            images = self._loadImageViaKeras(row["image"]['origin'])
            image = images[0]
            image.shape = (1, image.shape[0] * image.shape[1] * image.shape[2])
            keras_processed = image[0]
            np.testing.assert_array_almost_equal(keras_processed, processed, decimal = 6)

    # Test full pre-processing for InceptionV3 as an example of a simple computation graph

    def _preprocessingInceptionV3Transformed(self, outputMode, outputCol):
        g = tf.Graph()
        with g.as_default():
            image_arr = utils.imageInputPlaceholder()
            resized_images = tf.image.resize_images(image_arr, InceptionV3Constants.INPUT_SHAPE)
            # keras expects array in RGB order, we get it from image schema in BGR => need to flip
            processed_images = preprocess_input(imageIO._reverseChannels(resized_images))
        self.assertEqual(processed_images.shape[1], InceptionV3Constants.INPUT_SHAPE[0])
        self.assertEqual(processed_images.shape[2], InceptionV3Constants.INPUT_SHAPE[1])

        transformer = TFImageTransformer(channelOrder='BGR', inputCol="image", outputCol=outputCol, graph=g,
                                         inputTensor=image_arr.name, outputTensor=processed_images,
                                         outputMode=outputMode)
        image_df = image_utils.getSampleImageDF()
        return transformer.transform(image_df.limit(5))

    def test_image_output(self):
        output_col = "resized_image"
        preprocessed_df = self._preprocessingInceptionV3Transformed("image", output_col)
        self.assertDfHasCols(preprocessed_df, [output_col])
        for row in preprocessed_df.collect():
            original = row["image"]
            processed = row[output_col]
            errMsg = "nChannels must match: original {} v.s. processed {}"
            errMsg = errMsg.format(original.nChannels, processed.nChannels)
            self.assertEqual(original.nChannels, processed.nChannels, errMsg)
            self.assertEqual(processed.height, InceptionV3Constants.INPUT_SHAPE[0])
            self.assertEqual(processed.width, InceptionV3Constants.INPUT_SHAPE[1])

    # TODO: add tests for non-RGB8 images, at least RGB-float32.

    # Test InceptionV3 prediction as an example of applying a trained model.

    def _executeTensorflow(self, graph, input_tensor_name, output_tensor_name,
                           df, input_col="image"):
        with tf.Session(graph=graph) as sess:
            output_tensor = graph.get_tensor_by_name(output_tensor_name)
            image_collected = df.collect()
            values = {}
            topK = {}
            for img_row in image_collected:
                image = np.expand_dims(imageStructToArray(img_row[input_col]), axis=0)
                uri = img_row['image']['origin']
                output = sess.run([output_tensor],
                                  feed_dict={
                                      graph.get_tensor_by_name(input_tensor_name): image
                })
                values[uri] = np.array(output[0])
                topK[uri] = decode_predictions(values[uri], top=5)[0]
        return values, topK

    def test_prediction_vs_tensorflow_inceptionV3(self):
        output_col = "prediction"
        image_df = image_utils.getSampleImageDF()

        # An example of how a pre-trained keras model can be used with TFImageTransformer
        with KSessionWrap() as (sess, g):
            with g.as_default():
                K.set_learning_phase(0)    # this is important but it's on the user to call it.
                # nChannels needed for input_tensor in the InceptionV3 call below
                image_string = utils.imageInputPlaceholder(nChannels=3)
                resized_images = tf.image.resize_images(image_string,
                                                        InceptionV3Constants.INPUT_SHAPE)
                # keras expects array in RGB order, we get it from image schema in BGR =>
                # need to flip
                preprocessed = preprocess_input(imageIO._reverseChannels(resized_images))
                model = InceptionV3(input_tensor=preprocessed, weights="imagenet")
                graph = tfx.strip_and_freeze_until([model.output], g, sess, return_graph=True)

        transformer = TFImageTransformer(channelOrder='BGR', inputCol="image", outputCol=output_col, graph=graph,
                                         inputTensor=image_string, outputTensor=model.output,
                                         outputMode="vector")
        transformed_df = transformer.transform(image_df.limit(10))
        self.assertDfHasCols(transformed_df, [output_col])
        collected = transformed_df.collect()
        transformer_values, transformer_topK = self.transformOutputToComparables(collected,
                                                                                 output_col, lambda row: row['image']['origin'])

        tf_values, tf_topK = self._executeTensorflow(graph, image_string.name, model.output.name,
                                                     image_df)
        self.compareClassSets(tf_topK, transformer_topK)
        self.compareClassOrderings(tf_topK, transformer_topK)
        self.compareArrays(tf_values, transformer_values, decimal=5)
