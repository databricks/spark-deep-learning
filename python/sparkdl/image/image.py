#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTE: This file is copied from Spark2.3 in order to be able to use this in already released spark versions.
# TODO: remove this when Spark 2.3 is out!

"""
.. attribute:: ImageSchema

    An attribute of this module that contains the instance of :class:`_ImageSchema`.

.. autoclass:: _ImageSchema
   :members:
"""

import numpy as np
from collections import namedtuple

from pyspark import SparkContext
from pyspark.sql.types import Row, _create_row, _parse_datatype_json_string
from pyspark.sql import DataFrame, SparkSession


class _ImageSchema(object):
    """
    Internal class for `pyspark.ml.image.ImageSchema` attribute. Meant to be private and
    not to be instantized. Use `pyspark.ml.image.ImageSchema` attribute to access the
    APIs of this class.
    """

    def __init__(self):
        self._imageSchema = None
        self._ocvTypes = None
        self._ocvTypesByName = None
        self._ocvTypesByMode = None
        self._imageFields = None
        self._undefinedImageType = None

    _OcvType = namedtuple("OcvType", ["name", "mode", "nChannels", "dataType", "nptype"])

    _ocvToNumpyMap = {
        "N/A": "N/A",
        "8U": np.dtype("uint8"),
        "8S": np.dtype("int8"),
        "16U": np.dtype('uint16'),
        "16S": np.dtype('int16'),
        "32S": np.dtype('int32'),
        "32F": np.dtype('float32'),
        "64F": np.dtype('float64')}
    _numpyToOcvMap = {x[1]: x[0] for x in _ocvToNumpyMap.items()}

    @property
    def imageSchema(self):
        """
        Returns the image schema.

        :return: a :class:`StructType` with a single column of images
               named "image" (nullable).

        .. versionadded:: 2.3.0
        """

        if self._imageSchema is None:
            ctx = SparkContext.getOrCreate()
            jschema = ctx._jvm.org.apache.spark.ml.image.ImageSchema.imageSchema()
            self._imageSchema = _parse_datatype_json_string(jschema.json())
        return self._imageSchema

    @property
    def ocvTypes(self):
        """
        Returns the OpenCV type mapping supported.

        :return: a dictionary containing the OpenCV type mapping supported.

        .. versionadded:: 2.3.0
        """
        if self._ocvTypes is None:
            ctx = SparkContext.getOrCreate()
            self._ocvTypes = [self._OcvType(name=x.name(),
                                            mode=x.mode(),
                                            nChannels=x.nChannels(),
                                            dataType=x.dataType(),
                                            nptype=self._ocvToNumpyMap[x.dataType()])
                              for x in ctx._jvm.org.apache.spark.ml.image.ImageSchema.javaOcvTypes()]
        return self._ocvTypes[:]

    def ocvTypeByName(self, name):
        if self._ocvTypesByName is None:
            self._ocvTypesByName = {x.name: x for x in self.ocvTypes}
        if not name in self._ocvTypesByName:
            raise ValueError(
                "Unsupported image format, can not find matching OpenCvFormat for type = '%s'; currently supported formats = %s" %
                (name, str(
                    self._ocvTypesByName.keys())))
        return self._ocvTypesByName[name]

    def ocvTypeByMode(self, mode):
        if self._ocvTypesByMode is None:
            self._ocvTypesByMode = {x.mode: x for x in self.ocvTypes}
        return self._ocvTypesByMode[mode]

    @property
    def imageFields(self):
        """
        Returns field names of image columns.

        :return: a list of field names.

        .. versionadded:: 2.3.0
        """

        if self._imageFields is None:
            ctx = SparkContext.getOrCreate()
            self._imageFields = list(ctx._jvm.org.apache.spark.ml.image.ImageSchema.imageFields())
        return self._imageFields

    @property
    def undefinedImageType(self):
        """
        Returns the name of undefined image type for the invalid image.

        .. versionadded:: 2.3.0
        """

        if self._undefinedImageType is None:
            ctx = SparkContext.getOrCreate()
            self._undefinedImageType = \
                ctx._jvm.org.apache.spark.ml.image.ImageSchema.undefinedImageType()
        return self._undefinedImageType

    def toNDArray(self, image):
        """
        Converts an image to an array with metadata.

        :param `Row` image: A row that contains the image to be converted. It should
            have the attributes specified in `ImageSchema.imageSchema`.
        :return: a `numpy.ndarray` that is an image.

        .. versionadded:: 2.3.0
        """

        if not isinstance(image, Row):
            raise TypeError(
                "image argument should be pyspark.sql.types.Row; however, "
                "it got [%s]." % type(image))

        if any(not hasattr(image, f) for f in self.imageFields):
            raise ValueError(
                "image argument should have attributes specified in "
                "ImageSchema.imageSchema [%s]." % ", ".join(self.imageFields))
        height = image.height
        width = image.width
        nChannels = image.nChannels
        ocvType = self.ocvTypeByMode(image.mode)
        if nChannels != ocvType.nChannels:
            raise ValueError(
                "Unexpected number of channels, image has %d channels but OcvType '%s' expects %d channels." %
                (nChannels, ocvType.name, ocvType.nChannels))
        itemSz = ocvType.nptype.itemsize
        return np.ndarray(
            shape=(height, width, nChannels),
            dtype=ocvType.nptype,
            buffer=image.data,
            strides=(width * nChannels * itemSz, nChannels * itemSz, itemSz))

    def toImage(self, array, origin=""):
        """
        Converts an array with metadata to a two-dimensional image.

        :param `numpy.ndarray` array: The array to convert to image.
        :param str origin: Path to the image, optional.
        :return: a :class:`Row` that is a two dimensional image.

        .. versionadded:: 2.3.0
        """

        if not isinstance(array, np.ndarray):
            raise TypeError(
                "array argument should be numpy.ndarray; however, it got [%s]." % type(array))

        if array.ndim != 3:
            raise ValueError("Invalid array shape %s" % str(array.shape))

        height, width, nChannels = array.shape
        dtype = array.dtype
        if not dtype in self._numpyToOcvMap:
            raise ValueError(
                "Unexpected/unsupported array data type '%s', currently only supported formats are %s" %
                (str(array.dtype), str(self._numpyToOcvMap.keys())))
        ocvName = "CV_%sC%d" % (self._numpyToOcvMap[dtype], nChannels)
        ocvType = self.ocvTypeByName(ocvName)

        # Running `bytearray(numpy.array([1]))` fails in specific Python versions
        # with a specific Numpy version, for example in Python 3.6.0 and NumPy 1.13.3.
        # Here, it avoids it by converting it to bytes.
        data = bytearray(array.tobytes())

        # Creating new Row with _create_row(), because Row(name = value, ... )
        # orders fields by name, which conflicts with expected schema order
        # when the new DataFrame is created by UDF
        return _create_row(self.imageFields,
                           [origin, height, width, nChannels, ocvType.mode, data])

    def readImages(self, path, recursive=False, numPartitions=-1,
                   dropImageFailures=False, sampleRatio=1.0, seed=0):
        """
        Reads the directory of images from the local or remote source.

        .. note:: If multiple jobs are run in parallel with different sampleRatio or recursive flag,
            there may be a race condition where one job overwrites the hadoop configs of another.

        .. note:: If sample ratio is less than 1, sampling uses a PathFilter that is efficient but
            potentially non-deterministic.

        :param str path: Path to the image directory.
        :param bool recursive: Recursive search flag.
        :param int numPartitions: Number of DataFrame partitions.
        :param bool dropImageFailures: Drop the files that are not valid images.
        :param float sampleRatio: Fraction of the images loaded.
        :param int seed: Random number seed.
        :return: a :class:`DataFrame` with a single column of "images",
               see ImageSchema for details.

        >>> df = ImageSchema.readImages('python/test_support/image/kittens', recursive=True)
        >>> df.count()
        4

        .. versionadded:: 2.3.0
        """

        ctx = SparkContext.getOrCreate()
        spark = SparkSession(ctx)
        image_schema = ctx._jvm.org.apache.spark.ml.image.ImageSchema
        jsession = spark._jsparkSession
        jresult = image_schema.readImages(path, jsession, recursive, numPartitions,
                                          dropImageFailures, float(sampleRatio), seed)
        return DataFrame(jresult, spark._wrapped)


ImageSchema = _ImageSchema()


# Monkey patch to disallow instantization of this class.
def _disallow_instance(_):
    raise RuntimeError("Creating instance of _ImageSchema class is disallowed.")


_ImageSchema.__init__ = _disallow_instance
