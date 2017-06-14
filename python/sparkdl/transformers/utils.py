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

import tensorflow as tf

from pyspark.ml.param import TypeConverters

from sparkdl.image.imageIO import imageType
import sparkdl.graph.utils as tfx


# image stuff

IMAGE_INPUT_PLACEHOLDER_NAME = "sparkdl_image_input"

def imageInputPlaceholder(nChannels=None):
    return tf.placeholder(tf.float32, [None, None, None, nChannels],
                          name=IMAGE_INPUT_PLACEHOLDER_NAME)

class ImageNetConstants:
    NUM_CLASSES = 1000

# probably use a separate module for each network once we have featurizers.
class InceptionV3Constants:
    INPUT_SHAPE = (299, 299)
    NUM_OUTPUT_FEATURES = 131072


# general transformer stuff

# 1. Sometimes the tf graph contains a bunch of stuff that doesn't lead to the
# output. TensorFrames does not like that, so we strip out the parts that
# are not necessary for the computation at hand.
# 2. We need to freeze the variables whose values are stored in the current
# session into constants to pass the graph around.
def stripAndFreezeGraph(tf_graph_def, sess, output_tensors):
    """
    A typical usage would look like:
      tf_graph_def = tf_graph.as_graph_def(add_shapes=True)
      sess = tf.Session(graph=tf_graph)
    """
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                    tf_graph_def,
                                                                    [op_name(t) for t in output_tensors],
                                                                    variable_names_blacklist=[])
    g2 = tf.Graph()
    with g2.as_default():
        tf.import_graph_def(output_graph_def, name='')
    return g2

# def op_name(tensor):
#     """
#     :param tensor: tensorflow.Tensor or a string
#     """
#     return _tensor_name(tensor).split(":")[0]

# def _tensor_name(tensor):
#     if isinstance(tensor, tf.Tensor):
#         return _tensor_name(tensor.name)
#     return TypeConverters.toString(tensor)
