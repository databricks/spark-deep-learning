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

import os
import shutil
import tempfile
import uuid

import PIL.Image
import numpy as np
from keras.applications import Xception
from keras.applications.imagenet_utils import preprocess_input

from sparkdl.estimators.keras_image_file_estimator import KerasImageFileEstimator
from sparkdl.transformers.keras_image import KerasImageFileTransformer
import sparkdl.utils.keras_model as kmutil

from ..tests import SparkDLTestCase
from ..transformers.image_utils import getSampleImagePaths

def _load_image_from_uri(local_uri):
    img = (PIL.Image
           .open(local_uri)
           .convert('RGB')
           .resize((299, 299), PIL.Image.ANTIALIAS))
    img_arr = np.array(img).astype(np.float32)
    img_tnsr = preprocess_input(img_arr[np.newaxis, :])
    return img_tnsr

class KerasEstimatorsTest(SparkDLTestCase):

    def _create_test_image_uri(self, repeat_factor=1, cardinality=100):
        image_uris = getSampleImagePaths() * repeat_factor
        image_labels = [np.random.randint(low=0, high=cardinality), len(image_uris)]
        image_uri_df = self.sc.parallelize(
            zip(image_uris, image_labels)).toDF([self.input_col, self.label_col])
        return image_uri_df

    def _get_estimator(self, model, label_cardinality,
                       optimizer='adam', loss='categorical_crossentropy',
                       keras_fit_params={'verbose': 1}):                
        model_filename = os.path.join(self.temp_dir, 'model-{}.h5'.format(str(uuid.uuid4())))
        model.save(model_filename)
        estm = KerasImageFileEstimator(inputCol=self.input_col, outputCol=self.output_col,
                                       labelCol=self.label_col, labelCardinality=label_cardinality,
                                       isOneHotLabel=False, imageLoader=_load_image_from_uri,
                                       optimizer=optimizer, loss=loss,
                                       kerasFitParams=keras_fit_params, modelFile=model_filename)  
        return estm

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_col = 'kerasTestImageUri'
        self.label_col = 'kerasTestlabel'
        self.output_col = 'kerasTestPreds'

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_valid_workflow(self):
        # Create image URI dataframe
        label_cardinality = 1000
        image_uri_df = self._create_test_image_uri(repeat_factor=10, cardinality=label_cardinality)
        estimator = self._get_estimator(Xception(weights=None), label_cardinality)
        self.assertTrue(estimator._validateParams())
        transformers = estimator.fit(image_uri_df)
        self.assertEqual(1, len(transformers))
        self.assertIsInstance(transformers[0]['transformer'], KerasImageFileTransformer)

    def test_keras_training_utils(self):
        self.assertTrue(kmutil.is_valid_optimizer('adam'))
        self.assertFalse(kmutil.is_valid_optimizer('noSuchOptimizer'))
        self.assertTrue(kmutil.is_valid_loss_function('mse'))
        self.assertFalse(kmutil.is_valid_loss_function('noSuchLossFunction'))
