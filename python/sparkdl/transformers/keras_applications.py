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

from abc import ABCMeta, abstractmethod

import keras.backend as K
from keras.applications import inception_v3, resnet50, vgg16, vgg19, xception
from keras.applications.imagenet_utils import decode_predictions
import tensorflow as tf

from sparkdl.transformers.utils import (
    imageInputPlaceholder, InceptionV3Constants)


KERAS_APPLICATION_MODELS = set(["InceptionV3"]) #, "ResNet50", "Xception", "VGG16", "VGG19"])


"""
Essentially a factory function for getting the correct KerasApplicationModelSession class
for the network name.
"""
def getKerasApplicationModel(name):
    if name not in KERAS_APPLICATION_MODELS:
        raise ValueError("%s is not a supported model. Supported models: %s" % name,
                         str(KERAS_APPLICATION_MODELS))

    if name == "InceptionV3":
        return InceptionV3Model()
    # elif name == "ResNet50":
    #     return ResNet50Model()
    # elif name == "Xception":
    #     return XceptionModel()
    # elif name == "VGG16":
    #     return VGG16Model()
    # elif name == "VGG19":
    #     return VGG19Model()
    else:
        raise ValueError("%s is not implemented but is in the supported models list: %s" %
                         name, str(KERAS_APPLICATION_MODELS))


class KerasApplicationModel:
    __metaclass__ = ABCMeta

    def getModelData(self, featurize):
        sess = tf.Session()
        with sess.as_default():
            K.set_learning_phase(0)
            inputImage = imageInputPlaceholder(nChannels=3)
            preprocessed = self.preprocess(inputImage)
            model = self.model(preprocessed, featurize)
        return dict(inputTensorName=inputImage.name,
                    outputTensorName=model.output.name,
                    session=sess,
                    inputTensorSize=self.inputShape(),
                    outputMode="vector")

    @abstractmethod
    def preprocess(self, inputImage):
        pass

    @abstractmethod
    def model(self, preprocessed, featurize):
        pass

    @abstractmethod
    def inputShape(self):
        pass

    @abstractmethod
    def numOutputFeatures(self):
        """
        For testing
        """
        pass


class InceptionV3Model(KerasApplicationModel):
    def preprocess(self, inputImage):
        return inception_v3.preprocess_input(inputImage)

    def model(self, preprocessed, featurize):
        return inception_v3.InceptionV3(input_tensor=preprocessed, weights="imagenet",
                                        include_top=(not featurize))

    def inputShape(self):
        return InceptionV3Constants.INPUT_SHAPE

    def numOutputFeatures(self):
        return InceptionV3Constants.NUM_OUTPUT_FEATURES


