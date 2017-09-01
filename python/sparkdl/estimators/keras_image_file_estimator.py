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

# pylint: disable=protected-access
from __future__ import absolute_import, division, print_function

import logging
import numpy as np

from pyspark.ml import Estimator
import pyspark.ml.linalg as spla
from pyspark.ml.param import Param, Params, TypeConverters

from sparkdl.image.imageIO import imageStructToArray
from sparkdl.param import (
    keyword_only, CanLoadImage, HasKerasModel,
    HasInputCol, HasInputImageNodeName, HasLabelCol,
    HasOutputNodeName, HasOutputCol, HasOutputMode)
from sparkdl.transformers.keras_image import KerasImageFileTransformer
import sparkdl.utils.jvmapi as JVMAPI
import sparkdl.utils.keras_model as kmutil

__all__ = ['KerasImageFileEstimator']

logger = logging.getLogger('sparkdl')

class KerasImageFileEstimator(Estimator, HasInputCol, HasInputImageNodeName,
                              HasOutputCol, HasOutputNodeName, HasLabelCol,
                              CanLoadImage, HasKerasModel, HasOutputMode):
    """
    Build a Estimator from a Keras model.
    """

    # TODO(phi-dbq): allow users to pass a Keras optimizer object
    #                so that they can configure learning rate, momentum, etc.
    optimizer = Param(Params._dummy(), "optimizer",
                      "named Keras optimizer (str), please refer to https://keras.io/optimizers",
                      typeConverter=TypeConverters.toString)

    loss = Param(Params._dummy(), "loss",
                 "named Keras loss (str), please refer to https://keras.io/losses",
                 typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self, inputCol=None, inputImageNodeName=None, outputCol=None,
                 outputNodeName=None, outputMode="vector", labelCol=None,
                 modelFile=None, imageLoader=None, optimizer=None, loss=None, kerasFitParams=None):
        """
        __init__(self, inputCol=None, inputImageNodeName=None, outputCol=None,
                 outputNodeName=None, outputMode="vector", labelCol=None,
                 modelFile=None, imageLoader=None, optimizer=None, loss=None, kerasFitParams=None)
        """
        # TODO: currently, we ignore output mode
        super(KerasImageFileEstimator, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, inputImageNodeName=None, outputCol=None,
                  outputNodeName=None, outputMode="vector", labelCol=None,
                  modelFile=None, imageLoader=None, optimizer=None, loss=None, kerasFitParams=None):
        """
        setParams(self, inputCol=None, inputImageNodeName=None, outputCol=None,
                  outputNodeName=None, outputMode="vector", labelCol=None,
                  modelFile=None, imageLoader=None, optimizer=None, loss=None, kerasFitParams=None)
        """
        kwargs = self._input_kwargs
        self._set(**kwargs)
        return self

    def setLabelCol(self, value):
        return self._set(labelCol=value)

    def getLabelCol(self):
        return self.getOrDefault(self.labelCol)

    def setOptimizer(self, value):
        return self._set(optimizer=value)

    def getOptimizer(self):
        return self.getOrDefault(self.optimizer)

    def setLoss(self, value):
        return self._set(loss=value)

    def getLoss(self):
        return self.getOrDefault(self.loss)

    def _validateParams(self):
        """
        Check Param values so we can throw errors on the driver, rather than workers.
        :return: True if parameters are valid
        """
        if not self.isDefined(self.optimizer):
            raise ValueError("Required Param optimizer must be specified but was not.")
        if not kmutil.is_valid_optimizer(self.getOptimizer()):
            raise ValueError("Named optimizer not supported in Keras: %s" % self.getOptimizer())
        if not self.isDefined(self.loss):
            raise ValueError("Required Param loss must be specified but was not.")
        if not kmutil.is_valid_loss_function(self.getLoss()):
            raise ValueError("Named loss not supported in Keras: %s" % self.getLoss())
        if not self.isDefined(self.inputCol):
            raise ValueError("Input column must be defined")
        if not self.isDefined(self.outputCol):
            raise ValueError("Output column must be defined")
        return True

    def _localFit(self, featuresBc, labelsBc, modelBytesBc, paramMap):
        # Copy the estimator to add to the Params without modifying this instance.
        estimator = self.copy(paramMap)
        features = featuresBc.value
        labels = None if labelsBc is None else labelsBc.value
        model = kmutil.bytes_to_model(modelBytesBc.value)

        model.compile(optimizer=estimator.getOptimizer(), loss=estimator.getLoss())
        _fit_params = estimator.getKerasFitParams()

        model.fit(x=features, y=labels, **_fit_params)
        return kmutil.model_to_bytes(model)

    def _getNumpyFeaturesAndLabels(self, dataset):
        """
        We assume the training data fits in memory on a single server.
        The input dataframe is converted to numerical image features and
        broadcast to all the worker nodes.
        """
        image_uri_col = self.getInputCol()
        label_col = None
        if self.isDefined(self.labelCol) and self.getLabelCol() != "":
            label_col = self.getLabelCol()
        tmp_image_col = self._loadedImageCol()
        image_df = self.loadImagesInternal(dataset, image_uri_col).dropna(subset=[tmp_image_col])

        localFeatures = []
        localLabels = []
        is_with_labels = label_col is not None

        if is_with_labels:
            label_schema = image_df.schema[label_col]
            assert isinstance(label_schema.dataType, spla.VectorUDT), \
                "must encode labels in one-hot vector format"

        for row in image_df.collect():
            spimg = row[tmp_image_col]
            features = imageStructToArray(spimg)
            localFeatures.append(features)
            if is_with_labels:
                try:
                    _keras_label = row[label_col].array
                except ValueError:
                    raise ValueError("Cannot extract encoded label array")
                localLabels.append(_keras_label)

        if len(localFeatures) == 0:
            raise ValueError("Given empty dataset!")

        # We must reshape input data to form to the same size so as to be stacked
        X = np.stack(localFeatures, axis=0)
        y = None
        if is_with_labels:
            if not localLabels:
                raise ValueError("Failed to load any labels from dataset, but labels are required")
            y = np.stack(localLabels, axis=0)
            assert y.shape[0] == X.shape[0], "must have same amount of features and labels"
        return X, y

    def _collectModels(self, kerasModelsBytesRDD):
        """
        Collect Keras models on workers to MLlib Models on the driver.
        :param kerasModelBytesRDD: RDD of (param_map, model_bytes) tuples
        :param paramMaps: list of ParamMaps matching the maps in `kerasModelsRDD`
        :return: list of MLlib models
        """
        transformers = []
        for (param_map, model_bytes) in kerasModelsBytesRDD.collect():
            model_filename = kmutil.bytes_to_h5file(model_bytes)
            transformers.append({
                'paramMap': param_map,
                'transformer': KerasImageFileTransformer(modelFile=model_filename)})

        return transformers

    def _fitInParallel(self, dataset, paramMaps):
        """
        Fits len(paramMaps) models in parallel, one in each Spark task.
        :param paramMaps: non-empty list or tuple of ParamMaps (dict values)
        :return: list of fitted models, matching the order of paramMaps
        """
        sc = JVMAPI._curr_sc()
        paramMapsRDD = sc.parallelize(paramMaps, numSlices=len(paramMaps))

        # Extract image URI from provided dataset and create features as numpy arrays
        localFeatures, localLabels = self._getNumpyFeaturesAndLabels(dataset)
        localFeaturesBc = sc.broadcast(localFeatures)
        localLabelsBc = None if localLabels is None else sc.broadcast(localLabels)

        # Broadcast Keras model (HDF5) file content as bytes
        modelBytes = self._loadModelAsBytes()
        modelBytesBc = sc.broadcast(modelBytes)

        kerasModelBytesRDD = paramMapsRDD.map(
            lambda paramMap: (paramMap,
                              self._localFit(localFeaturesBc, localLabelsBc, modelBytesBc, paramMap)))
        return self._collectModels(kerasModelBytesRDD)

    def _loadModelAsBytes(self):
        """
        (usable on driver only)
        Load the Keras model file as a byte string.
        :return: str containing the model data
        """
        with open(self.getModelFile(), mode='rb') as fin:
            fileContent = fin.read()
        return fileContent

    def _fit(self, dataset): # pylint: disable=unused-argument
        err_msgs = ["This function should not have been called",
                    "Please contact library mantainers to file a bug"]
        raise NotImplementedError('\n'.join(err_msgs))

    def _validateFitParams(self, params):
        """ Check if an input parameter set is valid """
        if isinstance(params, (list, tuple, dict)):
            assert self.getInputCol() not in params, \
                "params {} cannot contain input column {}".format(params, self.getInputCol())

    def fit(self, dataset, params=None):
        """
        Fits a model to the input dataset with optional parameters.
        :param dataset: input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`.
                        The column `inputCol` should be of type
                        `sparkdl.image.imageIO.imgSchema`.
        :param params: An optional param map that overrides embedded params. If a list/tuple of
                       param maps is given, this calls fit on each param map and returns a list of
                       models.
        :returns: fitted model(s).  If params includes a list of param maps, the order of these
                  models matches the order of the param maps.
        """
        self._validateParams()
        if params is None:
            paramMaps = [dict()]
        elif isinstance(params, (list, tuple)):
            if len(params) == 0:
                paramMaps = [dict()]
            else:
                self._validateFitParams(params)
                paramMaps = params
        elif isinstance(params, dict):
            self._validateFitParams(params)
            paramMaps = [params]
        else:
            raise ValueError("Params must be either a param map or a list/tuple of param maps, "
                             "but got %s." % type(params))
        return self._fitInParallel(dataset, paramMaps)
