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


class KerasImageFileTransformerTest(SparkDLTestCase):

    def test_loadImages(self):
        input_col = "uri"
        output_col = "preds"

        model_path = image_utils.prepInceptionV3KerasModelFile("inceptionV3.h5")
        transformer = KerasImageFileTransformer(
            inputCol=input_col, outputCol=output_col, modelFile=model_path,
            imageLoader=image_utils.loadAndPreprocessKerasInceptionV3, outputMode="vector")

        uri_df = image_utils.getSampleImagePathsDF(self.sql, input_col)
        image_df = transformer.loadImagesInternal(uri_df, input_col)
        self.assertEqual(len(image_df.columns), 2)

        img_col = transformer._loadedImageCol()
        expected_shape = InceptionV3Constants.INPUT_SHAPE + (3,)
        for row in image_df.collect():
            arr = imageStructToArray(row[img_col])
            self.assertEqual(arr.shape, expected_shape)


class KerasImageFileTransformerExamplesTest(SparkDLTestCase, ImageNetOutputComparisonTestCase):

    def test_inceptionV3_vs_keras(self):
        input_col = "uri"
        output_col = "preds"

        model_path = image_utils.prepInceptionV3KerasModelFile("inceptionV3.h5")
        transformer = KerasImageFileTransformer(
            inputCol=input_col, outputCol=output_col, modelFile=model_path,
            imageLoader=image_utils.loadAndPreprocessKerasInceptionV3, outputMode="vector")

        uri_df = image_utils.getSampleImagePathsDF(self.sql, input_col)
        final_df = transformer.transform(uri_df)
        self.assertDfHasCols(final_df, [input_col, output_col])
        self.assertEqual(len(final_df.columns), 2)

        collected = final_df.collect()
        tvals, ttopK = self.transformOutputToComparables(
            collected, output_col, lambda row: row["uri"])
        kvals, ktopK = image_utils.executeKerasInceptionV3(uri_df, uri_col=input_col)

        self.compareClassSets(ktopK, ttopK)
        self.compareClassOrderings(ktopK, ttopK)
        self.compareArrays(kvals, tvals, decimal=5)

    # TODO: test a workflow with ImageDataGenerator and see if it fits. (It might not.)
