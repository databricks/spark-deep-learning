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

from sparkdl.transformers.keras_transformer import KerasTransformer
from ..tests import SparkDLTestCase
from .one_dim_utils import ImdbDatasetOutputComparisonTestCase
from . import one_dim_utils


class KerasTransformerTest(SparkDLTestCase, ImdbDatasetOutputComparisonTestCase):


    def test_imdb_model_vs_keras(self):
        input_col = "features"
        output_col = "preds"

        # TODO: Figure out which model to test against
        model_path = one_dim_utils.prepImdbKerasModelFile("imdb_model.h5")
        transformer = KerasTransformer(inputCol=input_col, outputCol=output_col,
                                       modelFile=model_path)

        # Load dataset, transform it with transformer
        df = one_dim_utils.getSampleImagePathsDF(self.sql, input_col)
        final_df = transformer.transform(df)

        # Verify that result DF has the specified input & output columns
        self.assertDfHasCols(final_df, [input_col, output_col])
        self.assertEqual(len(final_df.columns), 2)

        # Compare transformer output to keras model output
        collected = final_df.collect()
        sparkdl_predictions = self.transformOutputToComparables(collected, input_col, output_col)
        keras_predictions = one_dim_utils.executeKerasImdb(seq_df=df, seq_col=input_col)

        max_pred_diff = np.max(np.abs(sparkdl_predictions - keras_predictions))
        # Maximum acceptable (absolute) difference in KerasTransformer & Keras model output
        diff_tolerance = 1e-5
        assert(np.allclose(sparkdl_predictions, keras_predictions, atol=diff_tolerance),
               "KerasTransformer output differed (absolute difference) from Keras model output by "
               "as much as %s, maximum allowed deviation = %s"%(max_pred_diff, diff_tolerance))
