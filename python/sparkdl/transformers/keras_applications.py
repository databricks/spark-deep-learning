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
from keras.applications import inception_v3, xception
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


KERAS_APPLICATION_MODELS = {
    "InceptionV3": InceptionV3Model,
    "Xception": XceptionModel
}

