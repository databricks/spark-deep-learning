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

# 3rd party
import numpy as np
from PIL import Image

# pyspark
from pyspark import SparkContext
from pyspark.sql.functions import udf
from pyspark.sql.types import (
    BinaryType, IntegerType, StringType, StructField, StructType)

from sparkdl.image.image import ImageSchema


def imageStructToPIL(imageRow):
    """
    Convert the immage from image schema struct to PIL image

    :param imageRow: Row, must have ImageSchema
    :return PIL image
    """
    ary = ImageSchema.toNDArray(imageRow)
    if ary.dtype != np.uint8:
        raise ValueError("Can not convert image of type " +
                         ary.dtype + " to PIL, can only deal with 8U format")

    # PIL expects RGB order, image schema is BGR
    # => we need to flip the order unless there is only one channel
    if imageRow.nChannels != 1:
        ary = _reverseChannels(ary)
    if imageRow.nChannels == 1:
        return Image.fromarray(obj=ary, mode='L')
    elif imageRow.nChannels == 3:
        return Image.fromarray(obj=ary, mode='RGB')
    elif imageRow.nChannels == 4:
        return Image.fromarray(obj=ary, mode='RGBA')
    else:
        raise ValueError("don't know how to convert " +
                         imgType.name + " to PIL")


def PIL_to_imageStruct(img):
    # PIL is RGB based, image schema expects BGR ordering => need to flip the channels
    return _reverseChannels(np.asarray(img))


def fixColorChannelOrdering(currentOrder, imgAry):
    if currentOrder == 'RGB':
        return _reverseChannels(imgAry)
    elif currentOrder == 'BGR':
        return imgAry
    elif currentOrder == 'L':
        if len(img.shape) != 1:
            raise ValueError(
                "channel order suggests only one color channel but got shape " + str(img.shape))
        return imgAry
    else:
        raise ValueError(
            "Unexpected channel order, expected one of L,RGB,BGR but got " + currentChannelOrder)


def _stripBatchSize(imgArray):
    """
    Strip batch size (if it's there) from a multi dimensional array.
    Assumes batch size is the first coordinate and is equal to 1.
    Batch size != 1 will cause an error.

    :param imgArray: ndarray, image data.
    :return: imgArray without the leading batch size
    """
    # Sometimes tensors have a leading "batch-size" dimension. Assume to be 1 if it exists.
    if len(imgArray.shape) == 4:
        if imgArray.shape[0] != 1:
            raise ValueError(
                "The first dimension of a 4-d image array is expected to be 1.")
        imgArray = imgArray.reshape(imgArray.shape[1:])
    return imgArray


def _reverseChannels(ary):
    return ary[..., ::-1]


def createResizeImageUDF(size):
    """ Create a udf for resizing image.

    Example usage:
    dataFrame.select(resizeImage((height, width))('imageColumn'))

    :param size: tuple, target size of new image in the form (height, width).
    :return: udf, a udf for resizing an image column to `size`.
    """
    if len(size) != 2:
        raise ValueError(
            "New image size should have format [height, width] but got {}".format(size))
    sz = (size[1], size[0])

    def _resizeImageAsRow(imgAsRow):
        if (imgAsRow.height, imgAsRow.width) == sz:
            return imgAsRow
        imgAsPil = imageStructToPIL(imgAsRow).resize(sz)
        # PIL is RGB based while image schema is BGR based => we need to flip the channels
        imgAsArray = PIL_to_imageStruct(imgAsPil)
        return ImageSchema.toImage(imgAsArray, origin=imgAsRow.origin)
    return udf(_resizeImageAsRow, ImageSchema.imageSchema['image'].dataType)


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
    rdd = sc.binaryFiles(
        path, minPartitions=numPartitions).repartition(numPartitions)
    rdd = rdd.map(lambda x: (x[0], bytearray(x[1])))
    return rdd.toDF(schema)


def PIL_decode(raw_bytes):
    """
    Decode a raw image bytes using PIL.
    :param raw_bytes:
    :return: image data as an array in CV_8UC3 format
    """
    return PIL_to_imageStruct(Image.open(BytesIO(raw_bytes)))


def PIL_decode_and_resize(size):
    """
    Decode a raw image bytes using PIL and resize it to target dimension, both using PIL.
    :param raw_bytes:
    :return: image data as an array in CV_8UC3 format
    """
    def _decode(raw_bytes):
        return PIL_to_imageStruct(Image.open(BytesIO(raw_bytes)).resize(size))
    return _decode


def readImagesWithCustomFn(path, decode_f, numPartition=None):
    """
    Read a directory of images (or a single image) into a DataFrame using a custom library to decode the images.

    :param path: str, file path.
    :param decode_f: function to decode the raw bytes into an array compatible with one of the supported OpenCv modes.
                 see @imageIO.PIL_decode for an example.
    :param numPartition: [optional] int, number or partitions to use for reading files.
    :return: DataFrame with schema == ImageSchema.imageSchema.
    """
    return _readImagesWithCustomFn(path, decode_f, numPartition, sc=SparkContext.getOrCreate())


def _readImagesWithCustomFn(path, decode_f, numPartition, sc):
    def _decode(path, raw_bytes):
        try:
            return ImageSchema.toImage(decode_f(raw_bytes), origin=path)
        except BaseException:
            return None
    decodeImage = udf(_decode, ImageSchema.imageSchema['image'].dataType)
    imageData = filesToDF(sc, path, numPartitions=numPartition)
    return imageData.select(decodeImage("filePath", "fileData").alias("image"))
