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
import tempfile

from keras.models import Sequential
from keras.layers import Dense, Activation

from sparkdl.transformers.keras_transformer import KerasTransformer
from ..tests import SparkDLTestCase

class KerasTransformerTest(SparkDLTestCase):

    def _loadNumpyData(self, num_examples, num_features):
        """
        Construct and return a 2D numpy array of shape (num_examples, num_features) corresponding
        to one-dimensional input data.
        """
        local_features = []
        np.random.seed(997)
        return np.random.randn(num_examples, num_features)

    def getInputDF(self, sqlContext, inputCol, idCol):
        """
        Return a DataFrame containing an integer ID column and an input column of arrays.
        :param inputCol: Input column name
        :param idCol: ID column name
        """
        x_train = self._loadNumpyData(num_examples=2, num_features=10)
        train_rows = [{idCol : i, inputCol : x_train[i].tolist()} for i in range(len(x_train))]
        return sqlContext.createDataFrame(train_rows)

    def getKerasModel(self):
        """ Build and return keras model for (binary) classification on one-dimensional data. """
        model = Sequential()

        # We add a vanilla hidden layer:
        model.add(Dense(units=50, input_shape=(10, )))
        model.add(Activation('relu'))

        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        return model

    def prepareKerasModelFile(self, model, filename):
        model_dir_tmp = tempfile.mkdtemp("sparkdl_keras_tests", dir="/tmp")
        path = os.path.join(model_dir_tmp, filename)
        model.save(path)
        return path

    def executeKerasModel(self, df, model, input_col, id_col):
        # Collect dataframe, sort rows by ID column
        rows = df.select(input_col, id_col).collect()
        rows.sort(key=lambda row: row[id_col])
        # Get numpy array (num_examples, num_features) containing input data
        x_predict = np.array([row[input_col] for row in rows])
        return model.predict(x_predict)

    def transformOutputToComparables(self, collected, id_col, output_col):
        """
        Returns a list of model predictions ordered by row ID
        params:
        :param collected: Output (DataFrame) of KerasTransformer.transform()
        :param id_col: Column containing row ID
        :param output_col: Column containing transform() output
        """
        collected.sort(key=lambda row: row[id_col])
        return [row[output_col][0] for row in collected]

    def test_imdb_model_vs_keras(self):
        input_col = "features"
        output_col = "preds"
        id_col = "id"
        model = self.getKerasModel()
        model_path = self.prepareKerasModelFile(model, "keras_transformer_test_model.h5")
        transformer = KerasTransformer(inputCol=input_col, outputCol=output_col,
                                       modelFile=model_path)

        # Load dataset, transform it with transformer
        df = self.getInputDF(self.sql, inputCol=input_col, idCol=id_col)
        final_df = transformer.transform(df)

        # Verify that result DF has the specified input & output columns
        self.assertDfHasCols(final_df, [input_col, output_col, id_col])
        self.assertEqual(len(final_df.columns), 3)

        # Compare transformer output to keras model output
        collected = final_df.collect()
        sparkdl_predictions = self.transformOutputToComparables(collected, id_col, output_col)
        keras_predictions_raw = self.executeKerasModel(df=df, model=model,
                                                   input_col=input_col, id_col=id_col)
        keras_predictions = keras_predictions_raw.reshape((len(keras_predictions_raw),))

        max_pred_diff = np.max(np.abs(sparkdl_predictions - keras_predictions))
        # Maximum acceptable (absolute) difference in KerasTransformer & Keras model output
        diff_tolerance = 1e-5
        assert np.allclose(sparkdl_predictions, keras_predictions, atol=diff_tolerance), "" \
            "KerasTransformer output differed (absolute difference) from Keras model output by "\
            "as much as %s, maximum allowed deviation = %s"%(max_pred_diff, diff_tolerance)
