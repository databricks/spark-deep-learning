# coding=utf-8
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

#
# Models marked below as provided by Keras are provided subject to the
# below copyright and licenses (and any additional copyrights and
# licenses specified).
#
# COPYRIGHT
#
# All contributions by François Chollet:
# Copyright (c) 2015, François Chollet.
# All rights reserved.
#
# All contributions by Google:
# Copyright (c) 2015, Google, Inc.
# All rights reserved.
#
# All contributions by Microsoft:
# Copyright (c) 2017, Microsoft, Inc.
# All rights reserved.
#
# All other contributions:
# Copyright (c) 2015 - 2017, the respective contributors.
# All rights reserved.
#
# Each contributor holds copyright over their respective contributions.
# The project versioning (Git) records all such contribution source information.
#
# LICENSE
#
# The MIT License (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


from abc import ABCMeta, abstractmethod

import keras.backend as K
from keras.applications import inception_v3, xception, resnet50, vgg16, vgg19
import numpy as np
import tensorflow as tf

from sparkdl.transformers.utils import (imageInputPlaceholder, InceptionV3Constants)
from sparkdl.image.imageIO import _reverseChannels




def getKerasApplicationModel(name):
    """
    Essentially a factory function for getting the correct KerasApplicationModel class for the
    network name.
    """
    try:
        return KERAS_APPLICATION_MODELS[name]()
    except KeyError:
        raise ValueError("%s is not a supported model. Supported models: %s" %
                         (name, ', '.join(KERAS_APPLICATION_MODELS.keys())))


class KerasApplicationModel:
    __metaclass__ = ABCMeta

    def getModelData(self, featurize):
        sess = tf.Session()
        with sess.as_default():     # pylint: disable=not-context-manager
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
        """
        Models marked as *provided by Keras* are provided subject to the MIT
        license located at https://github.com/fchollet/keras/blob/master/LICENSE
        and subject to any additional copyrights and licenses specified in the
        code or documentation.
        """
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
        # Keras expects RGB order
        return inception_v3.preprocess_input(_reverseChannels(inputImage))

    def model(self, preprocessed, featurize):
        # Model provided by Keras. All cotributions by Keras are provided subject to the
        # MIT license located at https://github.com/fchollet/keras/blob/master/LICENSE
        # and subject to the below additional copyrights and licenses.
        #
        # Copyright 2016 The TensorFlow Authors.  All rights reserved.
        #
        # Licensed under the Apache License, Version 2.0 (the "License");
        # you may not use this file except in compliance with the License.
        # You may obtain a copy of the License at
        #
        # http://www.apache.org/licenses/LICENSE-2.0
        #
        # Unless required by applicable law or agreed to in writing, software
        # distributed under the License is distributed on an "AS IS" BASIS,
        # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        # See the License for the specific language governing permissions and
        # limitations under the License.
        """
        From Keras: These weights are released under the Apache License 2.0.
        """
        return inception_v3.InceptionV3(input_tensor=preprocessed, weights="imagenet",
                                        include_top=(not featurize))

    def inputShape(self):
        return InceptionV3Constants.INPUT_SHAPE

    def _testKerasModel(self, include_top):
        return inception_v3.InceptionV3(weights="imagenet", include_top=include_top)


class XceptionModel(KerasApplicationModel):
    def preprocess(self, inputImage):
        # Keras expects RGB order
        return xception.preprocess_input(_reverseChannels(inputImage))

    def model(self, preprocessed, featurize):
        # Model provided by Keras. All cotributions by Keras are provided subject to the
        # MIT license located at https://github.com/fchollet/keras/blob/master/LICENSE.
        return xception.Xception(input_tensor=preprocessed, weights="imagenet",
                                 include_top=(not featurize))

    def inputShape(self):
        return (299, 299)

    def _testKerasModel(self, include_top):
        return xception.Xception(weights="imagenet",
                                 include_top=include_top)


class ResNet50Model(KerasApplicationModel):
    def preprocess(self, inputImage):
        return _imagenet_preprocess_input(inputImage, self.inputShape())

    def model(self, preprocessed, featurize):
        # Model provided by Keras. All cotributions by Keras are provided subject to the
        # MIT license located at https://github.com/fchollet/keras/blob/master/LICENSE
        # and subject to the below additional copyrights and licenses.
        #
        # The MIT License (MIT)
        #
        # Copyright (c) 2016 Shaoqing Ren
        #
        # Permission is hereby granted, free of charge, to any person obtaining a copy
        # of this software and associated documentation files (the "Software"), to deal
        # in the Software without restriction, including without limitation the rights
        # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        # copies of the Software, and to permit persons to whom the Software is
        # furnished to do so, subject to the following conditions:
        #
        # The above copyright notice and this permission notice shall be included in all
        # copies or substantial portions of the Software.
        #
        # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        # SOFTWARE.
        return resnet50.ResNet50(input_tensor=preprocessed, weights="imagenet",
                                 include_top=(not featurize))

    def inputShape(self):
        return (224, 224)

    def _testKerasModel(self, include_top):
        # New Keras model changed the sturecture of ResNet50, we need to add avg for to compare
        # the result. We need to change the DeepImageFeaturizer for the new Model definition in
        # Keras
        return resnet50.ResNet50(weights="imagenet", include_top=include_top, pooling='avg')


class VGG16Model(KerasApplicationModel):
    def preprocess(self, inputImage):
        return _imagenet_preprocess_input(inputImage, self.inputShape())

    def model(self, preprocessed, featurize):
        # Model provided by Keras. All cotributions by Keras are provided subject to the
        # MIT license located at https://github.com/fchollet/keras/blob/master/LICENSE
        # and subject to the below additional copyrights and licenses.
        #
        # Copyright 2014 Oxford University
        #
        # Licensed under the Creative Commons Attribution License CC BY 4.0 ("License").
        # You may obtain a copy of the License at
        #
        #     https://creativecommons.org/licenses/by/4.0/
        #
        return vgg16.VGG16(input_tensor=preprocessed, weights="imagenet",
                           include_top=(not featurize))

    def inputShape(self):
        return (224, 224)

    def _testKerasModel(self, include_top):
        return vgg16.VGG16(weights="imagenet", include_top=include_top)


class VGG19Model(KerasApplicationModel):
    def preprocess(self, inputImage):
        return _imagenet_preprocess_input(inputImage, self.inputShape())

    def model(self, preprocessed, featurize):
        # Model provided by Keras. All cotributions by Keras are provided subject to the
        # MIT license located at https://github.com/fchollet/keras/blob/master/LICENSE
        # and subject to the below additional copyrights and licenses.
        #
        # Copyright 2014 Oxford University
        #
        # Licensed under the Creative Commons Attribution License CC BY 4.0 ("License").
        # You may obtain a copy of the License at
        #
        #     https://creativecommons.org/licenses/by/4.0/
        #
        return vgg19.VGG19(input_tensor=preprocessed, weights="imagenet",
                           include_top=(not featurize))

    def inputShape(self):
        return (224, 224)

    def _testKerasModel(self, include_top):
        return vgg19.VGG19(weights="imagenet", include_top=include_top)


def _imagenet_preprocess_input(x, input_shape):
    """
    For ResNet50, VGG models. For InceptionV3 and Xception it's okay to use the
    keras version (e.g. InceptionV3.preprocess_input) as the code path they hit
    works okay with tf.Tensor inputs. The following was translated to tf ops from
    https://github.com/fchollet/keras/blob/fb4a0849cf4dc2965af86510f02ec46abab1a6a4/keras/applications/imagenet_utils.py#L52
    It's a possibility to change the implementation in keras to look like the
    following and modified to work with BGR images (standard in Spark), but not doing it for now.
    """
    # assuming 'BGR'
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
    "VGG16": VGG16Model,
    "VGG19": VGG19Model,
}
