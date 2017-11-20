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

from sparkdl.image.imageIO import imageStructToArray
from sparkdl.transformers.keras_image import KerasImageFileTransformer
from sparkdl.transformers.utils import InceptionV3Constants
from ..tests import SparkDLTestCase
from .image_utils import ImageNetOutputComparisonTestCase
from . import image_utils


class KerasVectorTransformerTest(SparkDLTestCase):


    def test_inceptionV3_vs_keras(self):
        input_col = "features"
        output_col = "preds"

        # TODO: Figure out which model to test against
        model_path = image_utils.prepSomeModelFile("inceptionV3.h5")
        transformer = KerasVectorTransformer(inputCol=input_col, outputCol=output_col,
                                             modelFile=model_path)

        # Load dataset, transform it with transformer
        df = image_utils.getSampleImagePathsDF(self.sql, input_col)
        final_df = transformer.transform(df)

        # Verify that result DF has the specified input & output columns
        self.assertDfHasCols(final_df, [input_col, output_col])
        self.assertEqual(len(final_df.columns), 2)

        # Compare transformer output to keras model output
        collected = final_df.collect()
        tvals, ttopK = self.transformOutputToComparables(collected, input_col, output_col)
        kvals, ktopK = image_utils.executeKerasInceptionV3(df, uri_col=input_col)

        # TODO: Figure out how to compare model output, the methods below are in
        # image_utils.py: ImageNetOutputComparisonTestCase
        self.compareClassSets(ktopK, ttopK)
        self.compareClassOrderings(ktopK, ttopK)
        self.compareArrays(kvals, tvals)
