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
