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
from keras.applications import inception_v3, xception, resnet50
import numpy as np
import tensorflow as tf

from sparkdl.transformers.utils import (imageInputPlaceholder, InceptionV3Constants)


"""
Essentially a factory function for getting the correct KerasApplicationModel class
for the network name.
"""
def getKerasApplicationModel(name):
    try:
        return KERAS_APPLICATION_MODELS[name]()
    except KeyError:
        raise ValueError("%s is not a supported model. Supported models: %s" %
                         (name, ', '.join(KERAS_APPLICATION_MODELS.keys())))


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

    def _testPreprocess(self, inputImage):
        """
        For testing only. The preprocess function to be called before kerasModel.predict().
        """
        return self.preprocess(inputImage)

    @abstractmethod
    def _testKerasModel(self, include_top):
        """
        For testing only. The keras model object to compare to.
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

    def _testKerasModel(self, include_top):
        return inception_v3.InceptionV3(weights="imagenet", include_top=include_top)

class XceptionModel(KerasApplicationModel):
    def preprocess(self, inputImage):
        return xception.preprocess_input(inputImage)

    def model(self, preprocessed, featurize):
        return xception.Xception(input_tensor=preprocessed, weights="imagenet",
                                 include_top=(not featurize))

    def inputShape(self):
        return (299, 299)

    def _testKerasModel(self, include_top):
        return xception.Xception(weights="imagenet", include_top=include_top)

class ResNet50Model(KerasApplicationModel):
    def preprocess(self, inputImage):
        return _imagenet_preprocess_input(inputImage, self.inputShape())

    def model(self, preprocessed, featurize):
        return resnet50.ResNet50(input_tensor=preprocessed, weights="imagenet",
                                 include_top=(not featurize))

    def inputShape(self):
        return (224, 224)

    def _testKerasModel(self, include_top):
        return resnet50.ResNet50(weights="imagenet", include_top=include_top)

def _imagenet_preprocess_input(x, input_shape):
    """
    For ResNet50, VGG models. For InceptionV3 and Xception it's okay to use the
    keras version (e.g. InceptionV3.preprocess_input) as the code path they hit
    works okay with tf.Tensor inputs. The following was translated to tf ops from
    https://github.com/fchollet/keras/blob/fb4a0849cf4dc2965af86510f02ec46abab1a6a4/keras/applications/imagenet_utils.py#L52
    It's a possibility to change the implementation in keras to look like the
    following, but not doing it for now.
    """
    # 'RGB'->'BGR'
    x = x[..., ::-1]
    # Zero-center by mean pixel
    mean = np.ones(input_shape + (3,), dtype=np.float32)
    mean[..., 0] = 103.939
    mean[..., 1] = 116.779
    mean[..., 2] = 123.68
    return x - mean

KERAS_APPLICATION_MODELS = {
    "InceptionV3": InceptionV3Model,
    "Xception": XceptionModel,
    "ResNet50": ResNet50Model,
}

