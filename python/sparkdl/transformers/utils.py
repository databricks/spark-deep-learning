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

# image stuff

IMAGE_INPUT_PLACEHOLDER_NAME = "sparkdl_image_input"


def imageInputPlaceholder(nChannels=None):
    return tf.placeholder(tf.float32, [None, None, None, nChannels],
                          name=IMAGE_INPUT_PLACEHOLDER_NAME)


class ImageNetConstants:    # pylint: disable=too-few-public-methods
    NUM_CLASSES = 1000

# InceptionV3 is used in a lot of tests, so we'll make this shortcut available
# For other networks, see the keras_applications module.


class InceptionV3Constants:     # pylint: disable=too-few-public-methods
    INPUT_SHAPE = (299, 299)
    NUM_OUTPUT_FEATURES = 131072
