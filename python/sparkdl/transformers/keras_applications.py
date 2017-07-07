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

from sparkdl.transformers.utils import (imageInputPlaceholder, InceptionV3Constants)


KERAS_APPLICATION_MODELS = set(["InceptionV3", "ResNet50", "VGG16", "VGG19", "Xception"])

"""
Essentially a factory function for getting the correct KerasApplicationModel class
for the network name.
"""
def getKerasApplicationModel(name):
    if name not in KERAS_APPLICATION_MODELS:
        raise ValueError("%s is not a supported model. Supported models: %s" %
                         (name, str(KERAS_APPLICATION_MODELS)))

    if name == "InceptionV3":
        return InceptionV3Model()
    elif name == "ResNet50":
        return ResNet50Model()
    # elif name == "VGG16":
    #     return VGG16Model()
    # elif name == "VGG19":
    #     return VGG19Model()
    elif name == "Xception":
        return XceptionModel()
    else:
        raise ValueError("%s is not implemented but is in the supported models list: %s" %
                         (name, str(KERAS_APPLICATION_MODELS)))


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

    def testPreprocess(self, inputImage):
        """
        For testing, the keras preprocess function for the model.
        """
        return self.preprocess(inputImage)

    @abstractmethod
    def testKerasModel(self, include_top):
        """
        For testing, the keras model object to compare to.
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

    def testKerasModel(self, include_top):
        return inception_v3.InceptionV3(weights="imagenet", include_top=include_top)

class ResNet50Model(KerasApplicationModel):
    def preprocess(self, inputImage):
        return _preprocess_input(inputImage)  ############## DEBUG
        #return resnet50.preprocess_input(inputImage)

    def model(self, preprocessed, featurize):
        return resnet50.ResNet50(input_tensor=preprocessed, weights="imagenet",
                                 include_top=(not featurize), input_shape=self._inputShapeFull())

    def _inputShapeFull(self):
        return (224, 224, 3)

    def inputShape(self):
        return self._inputShapeFull()[0:2]

    def testPreprocess(self, inputImage):
        return resnet50.preprocess_input(inputImage)

    def testKerasModel(self, include_top):
        return resnet50.ResNet50(weights="imagenet", include_top=include_top)

class XceptionModel(KerasApplicationModel):
    def preprocess(self, inputImage):
        return xception.preprocess_input(inputImage)

    def model(self, preprocessed, featurize):
        return xception.Xception(input_tensor=preprocessed, weights="imagenet",
                                 include_top=(not featurize))

    def inputShape(self):
        return (299, 299)

    def testKerasModel(self, include_top):
        return xception.Xception(weights="imagenet", include_top=include_top)


# Translated to tensorflow operations from numpy operations in
# https://github.com/tensorflow/tensorflow/blob/f49f801276154d0f693c5d57db6977a7eb32f017/tensorflow/contrib/keras/python/keras/applications/imagenet_utils.py#L53
# Could break, but it doesn't seem that Keras models can be easily updated, so unlikely.
def _preprocess_input(x):
    # 'RGB'->'BGR'
    y = tf.reverse(x, [-1])
    # Zero-center by mean pixel
    ones = tf.ones_like(y[:, :, :, 1])
    return y - tf.stack([ones * 103.939, ones * 116.779, ones * 123.68], axis=-1)
