#
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
import logging

import tensorflow as tf

from sparkdl.graph.builder import IsolatedSession
from sparkdl.image import imageIO

logger = logging.getLogger('sparkdl')

"""
Build various pieces of the function

TODO: We might want to cache some of the big models in their GraphFunction format
      Deserializing ProtocolBuffer bytes is in general faster than directly loading Keras models.
"""


def buildSpImageConverter(channelOrder, img_dtype):
    """
    Convert a imageIO byte encoded image into a image tensor suitable as input to ConvNets
    The name of the input must be a subset of those specified in `image.imageIO.imageSchema`.

    :param img_dtype: the type of data the underlying image bytes represent
    """
    with IsolatedSession() as issn:
        # Flat image data -> image dimensions
        # This has to conform to `imageIO.imageSchema`
        height = tf.placeholder(tf.int32, [], name="height")
        width = tf.placeholder(tf.int32, [], name="width")
        num_channels = tf.placeholder(tf.int32, [], name="nChannels")
        image_buffer = tf.placeholder(tf.string, [], name="data")

        # The image is packed into bytes with height as leading dimension
        # This is the default behavior of Python Image Library
        shape = tf.reshape(tf.stack([height, width, num_channels], axis=0),
                           shape=(3,), name='shape')
        if img_dtype == 'uint8':
            image_uint8 = tf.decode_raw(image_buffer, tf.uint8, name="decode_raw")
            image_float = tf.to_float(image_uint8)
        elif img_dtype == 'float32':
            image_float = tf.decode_raw(image_buffer, tf.float32, name="decode_raw")
        else:
            raise ValueError('''unsupported image data type "%s", currently only know how to
            handle uint8 and float32''' % img_dtype)
        image_reshaped = tf.reshape(image_float, shape, name="reshaped")
        image_reshaped = imageIO.fixColorChannelOrdering(channelOrder, image_reshaped)
        image_input = tf.expand_dims(image_reshaped, 0, name="image_input")
        gfn = issn.asGraphFunction([height, width, image_buffer, num_channels], [image_input])

    return gfn


def buildFlattener():
    """
    Build a flattening layer to remove the extra leading tensor dimension.
    e.g. a tensor of shape [1, W, H, C] will have a shape [W, H, C] after applying this.
    """
    with IsolatedSession() as issn:
        mat_input = tf.placeholder(tf.float32, [None, None])
        mat_output = tf.identity(tf.reshape(mat_input, shape=[-1]), name='output')
        gfn = issn.asGraphFunction([mat_input], [mat_output])

    return gfn
