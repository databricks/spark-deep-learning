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
from ..tests import SparkDLTestCase

# Create dome fake image data to work with
def create_image_data():
    # Random image-like data
    array = np.random.randint(0, 256, (10, 11, 3), 'uint8')

    # Compress as png
    imgFile = BytesIO()
    PIL.Image.fromarray(array).save(imgFile, 'png')
    imgFile.seek(0)

    # Get Png data as stream
    pngData = imgFile.read()
    return array, pngData

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

    def test_decodeImage(self):
        badImg = imageIO._decodeImage(b"xxx")
        self.assertIsNone(badImg)
        imgRow = imageIO._decodeImage(pngData)
        self.assertIsNotNone(imgRow)
        self.assertEqual(len(imgRow), len(imageIO.imageSchema.names))
        for n in imageIO.imageSchema.names:
            imgRow[n]

    def test_resize(self):
        imgAsRow = imageIO.imageArrayToStruct(array)
        smaller = imageIO._resizeFunction([4, 5])
        smallerImg = smaller(imgAsRow)
        for n in imageIO.imageSchema.names:
            smallerImg[n]
        self.assertEqual(smallerImg.height, 4)
        self.assertEqual(smallerImg.width, 5)

        sameImage = imageIO._resizeFunction([imgAsRow.height, imgAsRow.width])(imgAsRow)
        self.assertEqual(sameImage, sameImage)

        self.assertRaises(ValueError, imageIO._resizeFunction, [1, 2, 3])

    def test_imageArrayToStruct(self):
        SparkMode = imageIO.SparkMode
        # Check converting with matching types
        height, width, chan = array.shape
        imgAsStruct = imageIO.imageArrayToStruct(array)
        self.assertEqual(imgAsStruct.height, height)
        self.assertEqual(imgAsStruct.width, width)
        self.assertEqual(imgAsStruct.data, array.tobytes())

        # Check casting
        imgAsStruct = imageIO.imageArrayToStruct(array, SparkMode.RGB_FLOAT32)
        self.assertEqual(imgAsStruct.height, height)
        self.assertEqual(imgAsStruct.width, width)
        self.assertEqual(len(imgAsStruct.data), array.size * 4)

        # Check channel mismatch
        self.assertRaises(ValueError, imageIO.imageArrayToStruct, array, SparkMode.FLOAT32)

        # Check that unsafe cast raises error
        floatArray = np.zeros((3, 4, 3), dtype='float32')
        self.assertRaises(ValueError, imageIO.imageArrayToStruct, floatArray, SparkMode.RGB)

    def test_image_round_trip(self):
        # Test round trip: array -> png -> sparkImg -> array
        binarySchema = StructType([StructField("data", BinaryType(), False)])
        df = self.session.createDataFrame([[bytearray(pngData)]], binarySchema)

        # Convert to images
        decImg = udf(imageIO._decodeImage, imageIO.imageSchema)
        imageDF = df.select(decImg("data").alias("image"))
        row = imageDF.first()

        testArray = imageIO.imageStructToArray(row.image)
        self.assertEqual(testArray.shape, array.shape)
        self.assertEqual(testArray.dtype, array.dtype)
        self.assertTrue(np.all(array == testArray))

    def test_readImages(self):
        # Test that reading
        imageDF = imageIO._readImages("some/path", 2, self.binaryFilesMock)
        self.assertTrue("image" in imageDF.schema.names)
        self.assertTrue("filePath" in imageDF.schema.names)

        # The DF should have 2 images and 1 null.
        self.assertEqual(imageDF.count(), 3)
        validImages = imageDF.filter(col("image").isNotNull())
        self.assertEqual(validImages.count(), 2)

        img = validImages.first().image
        self.assertEqual(img.height, array.shape[0])
        self.assertEqual(img.width, array.shape[1])
        self.assertEqual(imageIO.imageType(img).nChannels, array.shape[2])
        self.assertEqual(img.data, array.tobytes())

    def test_udf_schema(self):
        # Test that utility functions can be used to create a udf that accepts and return
        # imageSchema
        def do_nothing(imgRow):
            imType = imageIO.imageType(imgRow)
            array = imageIO.imageStructToArray(imgRow)
            return imageIO.imageArrayToStruct(array, imType.sparkMode)
        do_nothing_udf = udf(do_nothing, imageIO.imageSchema)

        df = imageIO._readImages("path", 2, self.binaryFilesMock)
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
