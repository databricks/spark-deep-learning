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
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.initializers import glorot_uniform

from sparkdl.transformers.keras_transformer import KerasTransformer
from ..tests import SparkDLTempDirTestCase


class KerasTransformerTest(SparkDLTempDirTestCase):

    NUM_FEATURES = 10
    RANDOM_SEED = 997

    def test_keras_transformer_single_dim(self):
        """
        Compare KerasTransformer output to that of a Keras model on a one-dimensional input dataset.
        """
        input_col = "features"
        output_col = "preds"
        id_col = "id"
        # Create Keras model, persist it to disk, and create KerasTransformer
        model = self._getKerasModel()
        model_path = self._prepareKerasModelFile(model, "keras_transformer_test_model.h5")
        transformer = KerasTransformer(inputCol=input_col, outputCol=output_col,
                                       modelFile=model_path)

        # Load dataset, transform it with KerasTransformer
        df = self._getInputDF(self.sql, inputCol=input_col, idCol=id_col)
        final_df = transformer.transform(df)
        sparkdl_predictions = self._convertOutputToComparables(final_df, id_col, output_col)

        self.assertDfHasCols(final_df, [input_col, output_col, id_col])
        self.assertEqual(len(final_df.columns), 3)

        # Compute Keras model local execution output
        keras_predictions_raw = self.executeKerasModel(df=df, model=model,
                                                       input_col=input_col, id_col=id_col)
        keras_predictions = keras_predictions_raw.reshape((len(keras_predictions_raw),))

        # Compare KerasTransformer & Keras model output
        max_pred_diff = np.max(np.abs(sparkdl_predictions - keras_predictions))
        diff_tolerance = 1e-5
        assert np.allclose(sparkdl_predictions, keras_predictions, atol=diff_tolerance), "" \
            "KerasTransformer output differed (absolute difference) from Keras model output by " \
            "as much as %s, maximum allowed deviation = %s"%(max_pred_diff, diff_tolerance)


    def _loadNumpyData(self, num_examples, num_features):
        """
        Construct and return a 2D numpy array of shape (num_examples, num_features) corresponding
        to a dataset of one-dimensional examples.
        """
        np.random.seed(self.RANDOM_SEED)
        return np.random.randn(num_examples, num_features)

    def _getInputDF(self, sqlContext, inputCol, idCol):
        """ Return a DataFrame containing a long ID column and an input column of arrays. """
        x_train = self._loadNumpyData(num_examples=20, num_features=self.NUM_FEATURES)
        train_rows = [{idCol : i, inputCol : x_train[i].tolist()} for i in range(len(x_train))]
        return sqlContext.createDataFrame(train_rows)

    def _getKerasModelWeightInitializer(self):
        """
        Get initializer for a set of weights (e.g. the kernel/bias of a single Dense layer)
        within a Keras model.
        """
        return glorot_uniform(seed=self.RANDOM_SEED)

    def _getKerasModel(self):
        """
        Build and return keras model for (binary) classification on one-dimensional data.

        The model has a single hidden layer that feeds into a single-unit, sigmoid-activated output
        unit. Weights of the hidden & output layer are drawn from a zero-mean uniform distribution.
        """
        model = Sequential()

        # We add a vanilla hidden layer, specifying the initializer (RNG) for the kernel & bias
        # weights
        model.add(Dense(units=50, input_shape=(self.NUM_FEATURES, ),
                        bias_initializer=self._getKerasModelWeightInitializer(),
                        kernel_initializer=self._getKerasModelWeightInitializer()))
        model.add(Activation('relu'))

        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(units=1, bias_initializer=self._getKerasModelWeightInitializer(),
                        kernel_initializer=self._getKerasModelWeightInitializer()))
        model.add(Activation('sigmoid'))
        return model

    def _prepareKerasModelFile(self, model, filename):
        """
        Saves the passed-in keras model to a temporary directory with the specified filename.
        """
        path = os.path.join(self.tempdir, filename)
        model.save(path)
        return path

    def _convertOutputToComparables(self, final_df, id_col, output_col):
        """
        Given the output of KerasTransformer.transform(), collects transformer output and
        returns a list of model predictions ordered by row id.

        params:
        :param final_df: DataFrame, output of KerasTransformer.transform()
        :param id_col: String, Column assumed to contain a row ID (Long)
        :param output_col: String, Column containing transform() output. We assume that for
                           each row, transform() has produced a single-element array containing a
                           Keras model's prediction on that row)
        """
        collected = final_df.collect()
        collected.sort(key=lambda row: row[id_col])
        return [row[output_col][0] for row in collected]

    def _executeKerasModelLocally(self, df, model, input_col, id_col):
        """
        Given an input DataFrame, locally collects the data in the specified input column and
        applies the passed-in Keras model, returning a numpy array of the model output sorted by
        increasing row ID (where row ID is contained in id_col).
        """
        rows = df.select(input_col, id_col).collect()
        rows.sort(key=lambda row: row[id_col])
        # Get numpy array (num_examples, num_features) containing input data
        x_predict = np.array([row[input_col] for row in rows])
        return model.predict(x_predict)
