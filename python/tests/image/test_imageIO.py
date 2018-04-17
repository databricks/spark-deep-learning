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

from io import BytesIO

# 3rd party
import numpy as np
import PIL.Image

# pyspark
from pyspark.sql.functions import col, udf
from pyspark.sql.types import BinaryType, StringType, StructField, StructType

from sparkdl.image import imageIO
from pyspark.ml.image import ImageSchema
from ..tests import SparkDLTestCase

# Create some fake image data to work with


def create_image_data():
    # Random image-like data
    array = np.random.randint(0, 256, (10, 11, 3), 'uint8')

    # Compress as png
    imgFile = BytesIO()
    PIL.Image.fromarray(array).save(imgFile, 'png')
    imgFile.seek(0)

    # Get Png data as stream
    pngData = imgFile.read()
    # PIL is RGB but image schema is BGR => flip the channels
    return imageIO._reverseChannels(array), pngData


array, pngData = create_image_data()


class BinaryFilesMock(object):

    defaultParallelism = 4

    def __init__(self, sc):
        self.sc = sc

    def binaryFiles(self, path, minPartitions=None):
        imagesData = [["file/path", pngData],
                      ["another/file/path", pngData],
                      ["bad/image", b"badImageData"]
                      ]
        rdd = self.sc.parallelize(imagesData)
        if minPartitions is not None:
            rdd = rdd.repartition(minPartitions)
        return rdd


class TestReadImages(SparkDLTestCase):
    @classmethod
    def setUpClass(cls):
        super(TestReadImages, cls).setUpClass()
        cls.binaryFilesMock = BinaryFilesMock(cls.sc)

    @classmethod
    def tearDownClass(cls):
        super(TestReadImages, cls).tearDownClass()
        cls.binaryFilesMock = None

    def test_resize(self):
        self.assertRaises(ValueError, imageIO.createResizeImageUDF, [1, 2, 3])

        make_smaller = imageIO.createResizeImageUDF([4, 5]).func
        imgAsRow = imageIO.imageArrayToStruct(array)
        smallerImg = make_smaller(imgAsRow)
        self.assertEqual(smallerImg.height, 4)
        self.assertEqual(smallerImg.width, 5)

        # Compare to PIL resizing
        imgAsPIL = PIL.Image.fromarray(obj=imageIO._reverseChannels(array)).resize((5, 4))
        smallerAry = imageIO._reverseChannels(np.asarray(imgAsPIL))
        np.testing.assert_array_equal(smallerAry, imageIO.imageStructToArray(smallerImg))
        # Test that resize with the same size is a no-op
        sameImage = imageIO.createResizeImageUDF((imgAsRow.height, imgAsRow.width)).func(imgAsRow)
        self.assertEqual(imgAsRow, sameImage)
        # Test that we have a valid image schema (all fields are in)
        for n in ImageSchema.imageSchema['image'].dataType.names:
            smallerImg[n]

    def test_imageConversions(self):
        """"
        Test conversion image array <-> image struct
        """
        def _test(array):
            height, width, chan = array.shape
            imgAsStruct = imageIO.imageArrayToStruct(array)
            self.assertEqual(imgAsStruct.height, height)
            self.assertEqual(imgAsStruct.width, width)
            self.assertEqual(imgAsStruct.data, array.tobytes())
            imgReconstructed = imageIO.imageStructToArray(imgAsStruct)
            np.testing.assert_array_equal(array, imgReconstructed)
        for nChannels in (1, 3, 4):
            # unsigned bytes
            _test(np.random.randint(0, 256, (10, 11, nChannels), 'uint8'))
            _test(np.random.random_sample((10, 11, nChannels)).astype('float32'))

    # read images now part of spark, no need to test it here
    def test_readImages(self):
        # Test that reading
        imageDF = imageIO._readImagesWithCustomFn(
            "file/path", decode_f=imageIO.PIL_decode, numPartition=2, sc=self.binaryFilesMock)
        self.assertTrue("image" in imageDF.schema.names)

        # The DF should have 2 images and 1 null.
        self.assertEqual(imageDF.count(), 3)
        validImages = imageDF.filter(col("image").isNotNull())
        self.assertEqual(validImages.count(), 2)

        img = validImages.first().image
        self.assertEqual(img.height, array.shape[0])
        self.assertEqual(img.width, array.shape[1])
        self.assertEqual(imageIO.imageTypeByOrdinal(img.mode).nChannels, array.shape[2])
        # array comes out of PIL and is in RGB order
        self.assertEqual(img.data, array.tobytes())

    def test_udf_schema(self):
        # Test that utility functions can be used to create a udf that accepts and return
        # imageSchema
        def do_nothing(imgRow):
            array = imageIO.imageStructToArray(imgRow)
            return imageIO.imageArrayToStruct(array)
        do_nothing_udf = udf(do_nothing, ImageSchema.imageSchema['image'].dataType)

        df = imageIO._readImagesWithCustomFn(
            "file/path", decode_f=imageIO.PIL_decode, numPartition=2, sc=self.binaryFilesMock)
        df = df.filter(col('image').isNotNull()).withColumn("test", do_nothing_udf('image'))
        self.assertEqual(df.first().test.data, array.tobytes())
        df.printSchema()

    def test_filesTODF(self):
        df = imageIO.filesToDF(self.binaryFilesMock, "path", 217)
        self.assertEqual(df.rdd.getNumPartitions(), 217)
        df.schema.fields[0].dataType == StringType()
        df.schema.fields[0].dataType == BinaryType()
        first = df.first()
        self.assertTrue(hasattr(first, "filePath"))
        self.assertEqual(type(first.fileData), bytearray)


# TODO: make unit tests for arrayToImageRow on arrays of varying shapes, channels, dtypes.
