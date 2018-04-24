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

"""
Some parts are copied from pyspark.ml.param.shared and some are complementary
to pyspark.ml.param. The copy is due to some useful pyspark fns/classes being
private APIs.
"""

from pyspark.ml.image import ImageSchema
from pyspark.ml.param import Param, Params
from pyspark.sql.functions import udf
from sparkdl.image.imageIO import _reverseChannels, imageArrayToStruct
from sparkdl.param.converters import SparkDLTypeConverters

OUTPUT_MODES = ["vector", "image"]


class CanLoadImage(Params):
    """
    In standard Keras workflow, we use provides an image loading function
    that takes a file path URI and convert it to an image tensor ready
    to be fed to the desired Keras model.

    This parameter allows users to specify such an image loading function.
    When using inside a pipeline stage, calling this function on an input DataFrame
    will load each image from the image URI column, encode the image in
    our :py:obj:`~sparkdl.imageIO.imageSchema` format and store it in the
    :py:meth:`~_loadedImageCol` column.

    Below is an example ``image_loader`` function to load Xception https://arxiv.org/abs/1610.02357
    compatible images.


    .. code-block:: python

        from keras.applications.xception import preprocess_input
        import numpy as np
        import PIL.Image

        def image_loader(uri):
            img = PIL.Image.open(uri).convert('RGB')
            img_resized = img.resize((299, 299), PIL.Image.ANTIALIAS))
            img_arr = np.array(img_resized).astype(np.float32)
            img_tnsr = preprocess_input(img_arr[np.newaxis, :])
            return img_tnsr
    """

    imageLoader = Param(
        Params._dummy(),
        "imageLoader",
        """Function containing the logic for loading and pre-processing images. The function
        should take in a URI string and return a 4-d numpy.array with shape (batch_size (1),
        height, width, num_channels). Expected to return result with color channels in RGB
        order.""")

    def setImageLoader(self, value):
        return self._set(imageLoader=value)

    def getImageLoader(self):
        return self.getOrDefault(self.imageLoader)

    def _loadedImageCol(self):  # pylint: disable=no-self-use
        return "__sdl_img"

    def loadImagesInternal(self, dataframe, inputCol):
        """
        Load image files specified in dataset as image format specified in `sparkdl.image.imageIO`.
        """
        # plan 1: udf(loader() + convert from np.array to imageSchema) -> call TFImageTransformer
        # plan 2: udf(loader()) ... we don't support np.array as a dataframe column type...
        loader = self.getImageLoader()
        # Load from external resources can fail, so we should allow None to be returned

        def load_image_uri_impl(uri):
            try:
                return imageArrayToStruct(_reverseChannels(loader(uri)))
            except BaseException:  # pylint: disable=bare-except
                return None
        load_udf = udf(load_image_uri_impl, ImageSchema.imageSchema['image'].dataType)
        return dataframe.withColumn(self._loadedImageCol(), load_udf(dataframe[inputCol]))


class HasOutputMode(Params):
    # TODO: docs
    outputMode = Param(
        Params._dummy(),
        "outputMode",
        """How the output column should be formatted. 'vector' for a 1-d MLlib Vector of floats.
        'image' to format the output to work with the image tools in this package.""",
        typeConverter=SparkDLTypeConverters.buildSupportedItemConverter(OUTPUT_MODES))

    def __init__(self):
        super(HasOutputMode, self).__init__()
        self._setDefault(outputMode="vector")

    def setOutputMode(self, value):
        return self._set(outputMode=value)

    def getOutputMode(self):
        return self.getOrDefault(self.outputMode)
