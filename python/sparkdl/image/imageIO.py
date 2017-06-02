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
from collections import namedtuple
from warnings import warn

# 3rd party
import numpy as np
from PIL import Image

# pyspark
from pyspark import Row
from pyspark.sql.types import (BinaryType, IntegerType, StringType, StructField, StructType)
from pyspark.sql.functions import udf


imgSchema = StructType([StructField("mode", StringType(), False),
                        StructField("height", IntegerType(), False),
                        StructField("width", IntegerType(), False),
                        StructField("nChannels", IntegerType(), False),
                        StructField("data", BinaryType(), False)])


# ImageType class for holding metadata about images stored in DataFrames.
# fields:
#   nChannels - number of channels in the image
#   dtype - data type of the image's "data" Column, sorted as a numpy compatible string.
#   channelContent - info about the contents of each channel currently only "I" (intensity) and
#     "RGB" are supported for 1 and 3 channel data respectively.
#   pilMode - The mode that should be used to convert to a PIL image.
#   sparkMode - Unique identifier string used in spark image representation.
ImageType = namedtuple("ImageType", ["nChannels",
                                     "dtype",
                                     "channelContent",
                                     "pilMode",
                                     "sparkMode",
                                     ])
class SparkMode(object):
    RGB = "RGB"
    FLOAT32 = "float32"
    RGB_FLOAT32 = "RGB-float32"

supportedImageTypes = [
    ImageType(3, "uint8", "RGB", "RGB", SparkMode.RGB),
    ImageType(1, "float32", "I", "F", SparkMode.FLOAT32),
    ImageType(3, "float32", "RGB", None, SparkMode.RGB_FLOAT32),
]
pilModeLookup = {t.pilMode: t for t in supportedImageTypes
                 if t.pilMode is not None}
sparkModeLookup = {t.sparkMode: t for t in supportedImageTypes}


def imageToStruct(imgArray, imageType=None):
    """
    Create spark image row from numpy array and (optional) imageType.

    to_image_udf = udf(arrayToImageRow, imgSchema)
    df.withColumn("output_img", to_image_udf(df["np_arr_col"])

    :param imgArray: ndarray, image data.
    :param imageType: ImageType, type of the image. If unspecified, it is inferred from imgArray.
    :return: list, image as a DataFrame Row compatible list.
    """
    arr = imgArray
    if imageType is None:
        # Sometimes tensors have a leading "batch-size" dimension. Assume to be 1 if it exists.
        if len(imgArray.shape) == 4:
            if imgArray.shape[0] != 1:
                raise ValueError("The first dimension of a 4-d image array is expected to be 1.")
            arr = imgArray.reshape(imgArray.shape[1:])

        sparkMode = _arrayToSparkMode(arr)
        if sparkMode in [SparkMode.FLOAT32, SparkMode.RGB_FLOAT32]:
            arr = arr.astype(np.float32)
    else:
        sparkMode = imageType.sparkMode

    height, width, nChannels = arr.shape
    data = bytearray(arr.tobytes())
    return Row(mode=sparkMode, height=height, width=width, nChannels=nChannels, data=data)


def imageType(imageRow):
    """
    Get type information about the image.

    :param imageRow: spark image row.
    :return: ImageType
    """
    return sparkModeLookup[imageRow.mode]


def imageToArray(imageRow):
    """
    Convert an image to a numpy array.

    :param imageRow: Row, must use imgSchema.
    :return: ndarray, image data.
    """
    imType = imageType(imageRow)
    shape = (imageRow.height, imageRow.width, imageRow.nChannels)
    return np.ndarray(shape, imType.dtype, imageRow.data)


def _arrayToSparkMode(arr):
    assert len(arr.shape) == 3, "Array should have 3 dimensions but has shape {}".format(arr.shape)
    num_channels = arr.shape[2]
    if num_channels == 1:
        if arr.dtype not in [np.float16, np.float32, np.float64]:
            raise ValueError("incompatible dtype (%s) for numpy array for float32 mode" %
                             arr.dtype.string)
        return SparkMode.FLOAT32
    elif num_channels != 3:
        raise ValueError("number of channels of the input array (%d) is not supported" %
                         num_channels)
    elif arr.dtype == np.uint8:
        return SparkMode.RGB
    elif arr.dtype in [np.float16, np.float32, np.float64]:
        return SparkMode.RGB_FLOAT32
    else:
        raise ValueError("did not find a sparkMode for the given array with num_channels = %d " +
                         "and dtype %s" % (num_channels, arr.dtype.string))


def _decodeImage(imageData):
    """
    Decode compressed image data into a DataFrame image row.

    :param imageData: (bytes, bytearray) compressed image data in PIL compatible format.
    :return: Row, decoded image.
    """
    try:
        img = Image.open(BytesIO(imageData))
    except IOError:
        return None

    if img.mode in pilModeLookup:
        mode = pilModeLookup[img.mode]
    else:
        msg = "We don't currently support images with mode: {mode}"
        warn(msg.format(mode=img.mode))
        return None
    imgArray = np.asarray(img)
    image = imageToStruct(imgArray, mode)
    return image

# Creating a UDF on import can cause SparkContext issues sometimes.
# decodeImage = udf(_decodeImage, imgSchema)

def filesToDF(sc, path, numPartitions=None):
    """
    Read files from a directory to a DataFrame.

    :param sc: SparkContext.
    :param path: str, path to files.
    :param numPartition: int, number or partitions to use for reading files.
    :return: DataFrame, with columns: (filePath: str, fileData: BinaryType)
    """
    numPartitions = numPartitions or sc.defaultParallelism
    schema = StructType([StructField("filePath", StringType(), False),
                         StructField("fileData", BinaryType(), False)])
    rdd = sc.binaryFiles(path, minPartitions=numPartitions).repartition(numPartitions)
    rdd = rdd.map(lambda x: (x[0], bytearray(x[1])))
    return rdd.toDF(schema)


def readImages(sc, imageDirectory, numPartition=None):
    """
    Read a directory of images (or a single image) into a DataFrame.

    :param sc: spark context
    :param imageDirectory: str, file path.
    :param numPartition: int, number or partitions to use for reading files.
    :return: DataFrame, with columns: (filepath: str, image: imgSchema).
    """
    decodeImage = udf(_decodeImage, imgSchema)
    imageData = filesToDF(sc, imageDirectory, numPartitions=numPartition)
    return imageData.select("filePath", decodeImage("fileData").alias("image"))
