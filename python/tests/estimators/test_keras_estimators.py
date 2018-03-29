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
from keras.layers import Activation, Dense, Flatten
from keras.models import Sequential
from keras.applications.imagenet_utils import preprocess_input

import pyspark.ml.linalg as spla
import pyspark.sql.types as sptyp
import pyspark.sql.functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

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

    def _create_train_image_uris_and_labels(self, repeat_factor=1, cardinality=100):
        image_uris = getSampleImagePaths() * repeat_factor
        # Create image categorical labels (integer IDs)
        local_rows = []
        for uri in image_uris:
            label = np.random.randint(low=0, high=cardinality, size=1)[0]
            label_inds = np.zeros(cardinality)
            label_inds[label] = 1.0
            label_inds = label_inds.ravel()
            assert label_inds.shape[0] == cardinality, label_inds.shape
            one_hot_vec = spla.Vectors.dense(label_inds.tolist())
            _row_struct = {self.input_col: uri, self.one_hot_col: one_hot_vec, self.label_col: float(label)}
            row = sptyp.Row(**_row_struct)
            local_rows.append(row)

        image_uri_df = self.session.createDataFrame(local_rows)
        image_uri_df.printSchema()
        return image_uri_df

    def _get_model(self, label_cardinality):
        # We need a small model so that machines with limited resources can run it
        model = Sequential()
        model.add(Flatten(input_shape=(299, 299, 3)))
        model.add(Dense(label_cardinality))
        model.add(Activation("softmax"))
        return model

    def _get_estimator(self, model):
        """
        Create a :py:obj:`KerasImageFileEstimator` from an existing Keras model
        """
        _random_filename_suffix = str(uuid.uuid4())
        model_filename = os.path.join(self.temp_dir, 'model-{}.h5'.format(_random_filename_suffix))
        model.save(model_filename)
        estm = KerasImageFileEstimator(inputCol=self.input_col,
                                       outputCol=self.output_col,
                                       labelCol=self.one_hot_col,
                                       imageLoader=_load_image_from_uri,
                                       kerasOptimizer='adam',
                                       kerasLoss='categorical_crossentropy',
                                       modelFile=model_filename)
        return estm

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_col = 'kerasTestImageUri'
        self.label_col = 'kerasTestLabel'
        self.one_hot_col = 'kerasTestOneHot'
        self.output_col = 'kerasTestPreds'

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_single_training(self):
        # Create image URI dataframe
        label_cardinality = 10
        image_uri_df = self._create_train_image_uris_and_labels(
            repeat_factor=3, cardinality=label_cardinality)

        model = self._get_model(label_cardinality)
        estimator = self._get_estimator(model)
        estimator.setKerasFitParams({'verbose': 1})
        self.assertTrue(estimator._validateParams())

        transformer = estimator.fit(image_uri_df)
        self.assertIsInstance(transformer, KerasImageFileTransformer, "output should be KIFT")
        output_df = transformer.transform(image_uri_df)
        self.assertEqual(len(output_df.select(F.col(self.output_col)).collect()), len(image_uri_df.collect()),
                         "output column should exist and be equal sized")

    def test_tuning(self):
        # Create image URI dataframe
        label_cardinality = 2
        image_uri_df = self._create_train_image_uris_and_labels(
            repeat_factor=3, cardinality=label_cardinality)

        model = self._get_model(label_cardinality)
        estimator = self._get_estimator(model)

        paramGrid = (
            ParamGridBuilder()
            .addGrid(estimator.kerasFitParams, [{"batch_size": 32}, {"batch_size": 64}])
            .build()
        )

        bc = BinaryClassificationEvaluator(rawPredictionCol=self.output_col, labelCol=self.label_col)
        cv = CrossValidator(estimator=estimator, estimatorParamMaps=paramGrid, evaluator=bc, numFolds=2)

        transformer = cv.fit(image_uri_df)
        self.assertIsInstance(transformer.bestModel, KerasImageFileTransformer, "best model should be KIFT")
        output_df = transformer.transform(image_uri_df)
        self.assertEqual(len(output_df.select(F.col(self.output_col)).collect()), len(image_uri_df.collect()),
                         "output column should exist and be equal sized")

    def test_keras_training_utils(self):
        self.assertTrue(kmutil.is_valid_optimizer('adam'))
        self.assertFalse(kmutil.is_valid_optimizer('noSuchOptimizer'))
        self.assertTrue(kmutil.is_valid_loss_function('mse'))
        self.assertFalse(kmutil.is_valid_loss_function('noSuchLossFunction'))
