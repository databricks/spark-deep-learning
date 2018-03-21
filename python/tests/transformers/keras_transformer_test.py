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
from keras.layers import Dense, Activation, Reshape
from keras.initializers import glorot_uniform

from pyspark.sql.types import *

from sparkdl.transformers.keras_tensor import KerasTransformer
from ..tests import SparkDLTempDirTestCase


class KerasTransformerTest(SparkDLTempDirTestCase):
    # Seed used for random-number generation in tests
    RANDOM_SEED = 997

    def test_keras_transformer_single_dim(self):
        """
        Test that KerasTransformer correctly handles single-dimensional input data.
        """
        # Construct a model for simple binary classification (with a single hidden layer)
        model = Sequential()
        input_shape = [10]
        model.add(Dense(units=10, input_shape=input_shape,
                        bias_initializer=self._getKerasModelWeightInitializer(),
                        kernel_initializer=self._getKerasModelWeightInitializer()))
        model.add(Activation('relu'))
        model.add(Dense(units=1, bias_initializer=self._getKerasModelWeightInitializer(),
                        kernel_initializer=self._getKerasModelWeightInitializer()))
        model.add(Activation('sigmoid'))
        # Compare KerasTransformer output to raw Keras model output
        self._test_keras_transformer_helper(model, model_filename="keras_transformer_single_dim")

    def test_keras_transformer_2dim(self):
        """
        Test that KerasTransformer correctly handles two-dimensional input/output tensors.
        """
        model = Sequential()
        input_shape = [2, 3]
        model.add(Reshape(input_shape=input_shape, target_shape=input_shape))
        self._test_keras_transformer_helper(model, model_filename="keras_transformer_multi_dim")

    def _test_keras_transformer_helper(self, model, model_filename):
        """
        Compares KerasTransformer output to raw Keras model output for the passed-in model.
        Saves the model to model_filename so that it can be loaded-in by KerasTransformer.
        """
        input_col = "inputCol"
        output_col = "outputCol"
        id_col = "id"

        # Create Keras model, persist it to disk, and create KerasTransformer
        save_filename = "%s.h5" % (model_filename)
        model_path = self._writeKerasModelFile(model, save_filename)
        transformer = KerasTransformer(inputCol=input_col, outputCol=output_col,
                                       modelFile=model_path)

        # Load dataset, transform it with KerasTransformer
        input_shape = list(model.input_shape[1:])  # Get shape of a single example
        df = self._getInputDF(self.sql, inputShape=input_shape, inputCol=input_col, idCol=id_col)
        final_df = transformer.transform(df)
        sparkdl_predictions = self._convertOutputToComparables(final_df, id_col, output_col)

        self.assertDfHasCols(final_df, [input_col, output_col, id_col])
        self.assertEqual(len(final_df.columns), 3)

        # Compute Keras model local execution output
        keras_predictions = self._executeKerasModelLocally(df=df, model=model,
                                                           input_col=input_col, id_col=id_col)
        # Compare KerasTransformer & Keras model output
        max_pred_diff = np.max(np.abs(sparkdl_predictions - keras_predictions))
        diff_tolerance = 1e-5
        assert np.allclose(sparkdl_predictions, keras_predictions, atol=diff_tolerance), "" \
            "KerasTransformer output differed (absolute difference) from Keras model output by " \
            "as much as %s, maximum allowed deviation = %s" % (max_pred_diff, diff_tolerance)

    def _getKerasModelWeightInitializer(self):
        """
        Get initializer for a set of weights (e.g. the kernel/bias of a single Dense layer)
        within a Keras model.
        """
        return glorot_uniform(seed=self.RANDOM_SEED)

    def _createNumpyData(self, num_examples, example_shape):
        """
        Return np array of num_examples data points where each data point has shape example_shape.
        """
        np.random.seed(self.RANDOM_SEED)
        data_shape = [num_examples] + example_shape
        return np.random.randn(*data_shape).astype(np.float32)

    def _getInputDF(self, sqlContext, inputShape, inputCol, idCol):
        """ Return a DataFrame containing a long ID column and an input column of arrays. """
        x_train = self._createNumpyData(num_examples=20, example_shape=inputShape)
        train_rows = [{idCol: i, inputCol: x_train[i].tolist()} for i in range(len(x_train))]
        input_col_type = FloatType()
        for _ in range(len(inputShape)):
            input_col_type = ArrayType(input_col_type)
        schema = StructType([StructField(idCol, IntegerType()), StructField(inputCol, input_col_type)])
        return sqlContext.createDataFrame(train_rows, schema)

    def _writeKerasModelFile(self, model, filename):
        """
        Saves the passed-in keras model to a temporary directory with the specified filename.
        """
        path = os.path.join(self.tempdir, filename)
        model.save(path)
        return path

    def _convertOutputToComparables(self, final_df, id_col, output_col):
        """
        Given the output of KerasTransformer.transform(), collects transformer output and
        returns a numpy array of model output ordered by row id.

        params:
        :param final_df: DataFrame, output of KerasTransformer.transform()
        :param id_col: String, Column assumed to contain a row ID (Long)
        :param output_col: String, Column containing transform() output.
        """
        collected = final_df.collect()
        collected.sort(key=lambda row: row[id_col])
        return np.array([row[output_col] for row in collected])

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
