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
from pyspark import Row
from pyspark import SparkContext
from pyspark.sql.types import (BinaryType, IntegerType, StringType, StructField, StructType)
from pyspark.sql.functions import udf
from pyspark.ml.image import ImageSchema


# ImageType represents supported OpenCV types
# fields:
#   name - OpenCvMode
#   ord - Ordinal of the corresponding OpenCV mode (stored in mode field of ImageSchema).
#   nChannels - number of channels in the image
#   dtype - data type of the image's array, sorted as a numpy compatible string.
OcvType = namedtuple("OcvType",["name","ord","nChannels","dtype"])

supportedOcvTypes = (
    OcvType(name="CV_8UC1",  ord=0,  nChannels=1, dtype="uint8"  ),
    OcvType(name="CV_32FC1", ord=5,  nChannels=1, dtype="float32"),
    OcvType(name="CV_8UC3",  ord=16, nChannels=3, dtype="uint8"  ),
    OcvType(name="CV_32FC3", ord=21, nChannels=3, dtype="float32"),
    OcvType(name="CV_8UC4",  ord=24, nChannels=4, dtype="uint8"  ),
    OcvType(name="CV_32FC4", ord=29, nChannels=4, dtype="float32"),
)

__ocvTypesByName = {m.name:m for m in supportedOcvTypes}
__ocvTypesByOrdinal = {m.ord:m for m in supportedOcvTypes}

def imageTypeByOrdinal(ord):
    if not ord in __ocvTypesByOrdinal:
        raise KeyError("unsupported image type with ordinal %d, supported OpenCV types = %s"  % (ord,str(supportedOcvTypes)))
    return __ocvTypesByOrdinal[ord]

def imageTypeByName(name):
    if not name in __ocvTypesByName:
        raise KeyError("unsupported image type with name '%s', supported supported OpenCV types = %s" % (name,str(supportedOcvTypes)))
    return __ocvTypesByName[name]

def imageArrayToStruct(imgArray,origin=""):
    """
    Create a row representation of an image from an image array.

    :param imgArray: ndarray, image data.
    :return: Row, image as a DataFrame Row with schema==ImageSchema.
    """
    # Sometimes tensors have a leading "batch-size" dimension. Assume to be 1 if it exists.
    if len(imgArray.shape) == 4:
        if imgArray.shape[0] != 1:
            raise ValueError("The first dimension of a 4-d image array is expected to be 1.")
        imgArray = imgArray.reshape(imgArray.shape[1:])
    imageType = _arrayToOcvMode(imgArray)
    height, width, nChannels = imgArray.shape
    data = bytearray(imgArray.tobytes())
    return Row(origin=origin,mode=imageType.ord, height=height, width=width, nChannels=nChannels, data=data)

def imageStructToArray(imageRow):
    """
    Convert an image to a numpy array.

    :param imageRow: Row, must use imageSchema.
    :return: ndarray, image data.
    """
    imType = imageTypeByOrdinal(imageRow.mode)
    shape = (imageRow.height, imageRow.width, imageRow.nChannels)
    return np.ndarray(shape, imType.dtype, imageRow.data)

def imageStructToPIL(imageRow):
    imgType = imageTypeByOrdinal(imageRow.mode)
    if imgType.dtype != 'uint8':
        raise ValueError("Can not convert image of type " + imgType.dtype + " to PIL, can only deal with 8U format")
    ary = imageStructToArray(imageRow)
    if imgType.nChannels != 1:
        ary = _rgb2bgr(ary)
    if imgType.nChannels == 1:
        return Image.fromarray(obj=ary,mode='L')
    elif imgType.nChannels == 3:
        return Image.fromarray(obj=ary,mode='RGB')
    elif imgType.nChannels == 4:
        return Image.fromarray(obj=ary,mode='RGBA')
    else:
        raise ValueError("don't know how to convert " + imgType.name + " to PIL")


def _arrayToOcvMode(arr):
    assert len(arr.shape) == 3, "Array should have 3 dimensions but has shape {}".format(arr.shape)
    num_channels = arr.shape[2]
    if arr.dtype == "uint8":
        name = "CV_8UC%d" % num_channels
    elif arr.dtype == "float32":
        name = "CV_32FC%d" % num_channels
    else:
        raise ValueError("Unsupported type '%s'" % arr.dtype)
    return imageTypeByName(name)


def _rgb2bgr(ary):
    return ary[...,::-1]

def resizeImage_python(size):
    """ Create a udf for resizing image.

    Example usage:
    dataFrame.select(resizeImage((height, width))('imageColumn'))

    :param size: tuple, target size of new image in the form (height, width).
    :return: udf, a udf for resizing an image column to `size`.
    """
    if len(size) != 2:
        raise ValueError("New image size should have format [height, width] but got {}".format(size))
    sz = (size[1],size[0])

    def _resizeImageAsRow(imgAsRow):
        if (imgAsRow.height,imgAsRow.width) == sz:
            return imgAsRow
        imgAsPil = imageStructToPIL(imgAsRow).resize(sz)
        imgAsArray = _rgb2bgr(np.asarray(imgAsPil))
        return imageArrayToStruct(imgAsArray,origin=imgAsRow.origin)
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
    rdd = sc.binaryFiles(path, minPartitions=numPartitions).repartition(numPartitions)
    rdd = rdd.map(lambda x: (x[0], bytearray(x[1])))
    return rdd.toDF(schema)

def PIL_decode(raw_bytes):
    """
    Decode a raw image bytes using PIL.
    :param raw_bytes:
    :return: image data as an array in CV_8UC3 format
    """
    return _rgb2bgr(np.asarray(Image.open(BytesIO(raw_bytes))))


def PIL_decode_and_resize(size):
    """
    Decode a raw image bytes using PIL and resize it to target dimension, both using PIL.
    :param raw_bytes:
    :return: image data as an array in CV_8UC3 format
    """
    def _decode(raw_bytes):
        return _rgb2bgr(np.asarray(Image.open(BytesIO(raw_bytes)).resize(size)))
    return _decode

def readImagesWithCustomLib(path, decode_f, numPartition = None):
    """
    Read a directory of images (or a single image) into a DataFrame using a custom library to decode the images.

    :param path: str, file path.
    :param decode_f: function to decode the raw bytes into an array compatible with one of the supported OpenCv modes.
                 see @imageIO.PIL_decode for an example.
    :param numPartition: [optional] int, number or partitions to use for reading files.
    :return: DataFrame with schema == ImageSchema.imageSchema.
    """
    return _readImagesWithCustomLib(path, decode_f, numPartition, sc = SparkContext.getOrCreate())
def _readImagesWithCustomLib(path, decode_f, numPartition, sc):
    def _decode(path,raw_bytes):
        try:
            return imageArrayToStruct(decode_f(raw_bytes),origin=path)
        except:
            return None
    decodeImage = udf(_decode, ImageSchema.imageSchema['image'].dataType)
    imageData = filesToDF(sc, path, numPartitions=numPartition)
    return imageData.select(decodeImage("filePath","fileData").alias("image"))


