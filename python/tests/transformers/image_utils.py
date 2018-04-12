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

import os
import tempfile
import unittest
from glob import glob
from warnings import warn


from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array, load_img
import keras.backend as K
import numpy as np
import PIL.Image


from pyspark.sql.types import StringType

from sparkdl.image import imageIO
from sparkdl.transformers.utils import ImageNetConstants, InceptionV3Constants


# Methods for getting some test data to work with.

def _getSampleJPEGDir():
    cur_dir = os.path.dirname(__file__)
    return os.path.join(cur_dir, "../resources/images")

def getImageFiles():
    return [path for path in glob(os.path.join(_getSampleJPEGDir(), "*"))
            if not os.path.isdir(path)]

def getSampleImageDF():
    return imageIO.readImagesWithCustomFn(path=_getSampleJPEGDir(), decode_f=imageIO.PIL_decode)

def getSampleImagePaths():
    dirpath = _getSampleJPEGDir()
    files = [os.path.abspath(os.path.join(dirpath, f)) for f in os.listdir(dirpath)
             if f.endswith('.jpg')]
    return files

def getSampleImagePathsDF(sqlContext, colName):
    files = getSampleImagePaths()
    return sqlContext.createDataFrame(files, StringType()).toDF(colName)

# Methods for making comparisons between outputs of using different frameworks.
# For ImageNet.


class ImageNetOutputComparisonTestCase(unittest.TestCase):

    def transformOutputToComparables(self, collected, output_col, get_uri):
        values = {}
        topK = {}
        for row in collected:
            uri = get_uri(row)
            predictions = row[output_col]
            self.assertEqual(len(predictions), ImageNetConstants.NUM_CLASSES)
            values[uri] = np.expand_dims(predictions, axis=0)
            topK[uri] = decode_predictions(values[uri], top=5)[0]
        return values, topK

    def compareArrays(self, values1, values2, decimal=None):
        """
        values1 & values2 are {key => numpy array}.
        """
        for k, v1 in values1.items():
            v1f = v1.astype(np.float32)
            v2f = values2[k].astype(np.float32)
            if decimal:
                np.testing.assert_array_almost_equal(v1f, v2f, decimal)
            else:
                np.testing.assert_array_equal(v1f, v2f)

    def compareClassOrderings(self, preds1, preds2):
        """
        preds1 & preds2 are {key => (class, description, probability)}.
        """
        for k, v1 in preds1.items():
            self.assertEqual([v[1] for v in v1], [v[1] for v in preds2[k]])

    def compareClassSets(self, preds1, preds2):
        """
        values1 & values2 are {key => numpy array}.
        """
        for k, v1 in preds1.items():
            self.assertEqual(set([v[1] for v in v1]), set([v[1] for v in preds2[k]]))


def executeKerasInceptionV3(image_df, uri_col="filePath"):
    """
    Apply Keras InceptionV3 Model on input DataFrame.
    :param image_df: Dataset. contains a column (uri_col) for where the image file lives.
    :param uri_col: str. name of the column indicating where each row's image file lives.
    :return: ({str => np.array[float]}, {str => (str, str, float)}).
      image file uri to prediction probability array,
      image file uri to top K predictions (class id, class description, probability).
    """
    K.set_learning_phase(0)
    model = InceptionV3(weights="imagenet")

    values = {}
    topK = {}
    for row in image_df.select(uri_col).collect():
        raw_uri = row[uri_col]
        image = loadAndPreprocessKerasInceptionV3(raw_uri)
        values[raw_uri] = model.predict(image)
        topK[raw_uri] = decode_predictions(values[raw_uri], top=5)[0]
    return values, topK


def loadAndPreprocessKerasInceptionV3(raw_uri):
    # this is the canonical way to load and prep images in keras
    uri = raw_uri[5:] if raw_uri.startswith("file:/") else raw_uri
    image = img_to_array(load_img(uri, target_size=InceptionV3Constants.INPUT_SHAPE))
    image = np.expand_dims(image, axis=0)
    return preprocess_input(image)


def prepInceptionV3KerasModelFile(fileName):
    model_dir_tmp = tempfile.mkdtemp("sparkdl_keras_tests", dir="/tmp")
    path = model_dir_tmp + "/" + fileName

    height, width = InceptionV3Constants.INPUT_SHAPE
    input_shape = (height, width, 3)
    model = InceptionV3(weights="imagenet", include_top=True, input_shape=input_shape)
    model.save(path)
    return path
