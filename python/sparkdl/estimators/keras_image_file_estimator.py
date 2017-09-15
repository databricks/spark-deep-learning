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
    keyword_only, CanLoadImage, HasKerasModel, HasKerasOptimizer, HasKerasLoss, HasOutputMode,
    HasInputCol, HasInputImageNodeName, HasLabelCol, HasOutputNodeName, HasOutputCol)
from sparkdl.transformers.keras_image import KerasImageFileTransformer
import sparkdl.utils.jvmapi as JVMAPI
import sparkdl.utils.keras_model as kmutil

__all__ = ['KerasImageFileEstimator']

logger = logging.getLogger('sparkdl')

class KerasImageFileEstimator(Estimator, HasInputCol, HasInputImageNodeName,
                              HasOutputCol, HasOutputNodeName, HasLabelCol,
                              HasKerasModel, HasKerasOptimizer, HasKerasLoss,
                              CanLoadImage, HasOutputMode):
    """
    Build a Estimator from a Keras model.

    First, create a model and save it to file system

    .. code-block:: python

        from keras.applications.resnet50 import ResNet50
        model = ResNet50(weights=None)
        model.save("path_to_my_model.h5")

    Then, create a image loading function that reads image data from URI,
    preprocess them, and returns the numerical tensor.

    .. code-block:: python

        def load_image_and_process(uri):
            import PIL.Image
            from keras.applications.imagenet_utils import preprocess_input

            original_image = PIL.Image.open(uri).convert('RGB')
            resized_image = original_image.resize((224, 224), PIL.Image.ANTIALIAS)
            image_array = np.array(resized_image).astype(np.float32)
            image_tensor = preprocess_input(image_array[np.newaxis, :])
            return image_tensor


    Assume the image URIs live in the following DataFrame.

    .. code-block:: python

        original_dataset = spark.createDataFrame([
            Row(imageUri="image1_uri", imageLabel="image1_label"),
            Row(imageUri="image2_uri", imageLabel="image2_label"),
            # and more rows ...
        ])
        stringIndexer = StringIndexer(inputCol="imageLabel", outputCol="categoryIndex")
        indexed_dateset = stringIndexer.fit(original_dataset).transform(original_dataset)
        encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
        image_dataset = encoder.transform(indexed_dateset)

    We can then create a Keras estimator that takes our saved model file and
    train it using Spark.

    .. code-block:: python

        estimator = KerasImageFileEstimator(inputCol="imageUri",
                                            outputCol="name_of_result_column",
                                            labelCol="categoryVec",
                                            imageLoader=load_image_and_process,
                                            kerasOptimizer="adam",
                                            kerasLoss="categorical_crossentropy",
                                            kerasFitParams={"epochs": 5, "batch_size": 64},
                                            modelFile="path_to_my_model.h5")

        transformers = estimator.fit(image_dataset)

    """

    @keyword_only
    def __init__(self, inputCol=None, inputImageNodeName=None, outputCol=None,
                 outputNodeName=None, outputMode="vector", labelCol=None,
                 modelFile=None, imageLoader=None, kerasOptimizer=None, kerasLoss=None,
                 kerasFitParams=None):
        """
        __init__(self, inputCol=None, inputImageNodeName=None, outputCol=None,
                 outputNodeName=None, outputMode="vector", labelCol=None,
                 modelFile=None, imageLoader=None, kerasOptimizer=None, kerasLoss=None,
                 kerasFitParams=None)
        """
        # NOTE(phi-dbq): currently we ignore output mode, as the actual output are the
        #                trained models and the Transformers built from them.
        super(KerasImageFileEstimator, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, inputImageNodeName=None, outputCol=None,
                  outputNodeName=None, outputMode="vector", labelCol=None,
                  modelFile=None, imageLoader=None, kerasOptimizer=None, kerasLoss=None,
                  kerasFitParams=None):
        """
        setParams(self, inputCol=None, inputImageNodeName=None, outputCol=None,
                  outputNodeName=None, outputMode="vector", labelCol=None,
                  modelFile=None, imageLoader=None, kerasOptimizer=None, kerasLoss=None,
                  kerasFitParams=None)
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def fit(self, dataset, params=None):
        """
        Fits a model to the input dataset with optional parameters.

        .. warning:: This returns the byte serialized HDF5 file for each model to the driver.
                     If the model file is large, the driver might go out-of-memory.
                     As we cannot assume the existence of a sufficiently large (and writable)
                     file system, users are advised to not train too many models in a single
                     Spark job.

        :param dataset: input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`.
                        The column `inputCol` should be of type `sparkdl.image.imageIO.imgSchema`.
        :param params: An optional param map that overrides embedded params. If a list/tuple of
                       param maps is given, this calls fit on each param map and returns a list of
                       models.
        :return: fitted model(s). If params includes a list of param maps, the order of these
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

    def _validateParams(self):
        """
        Check Param values so we can throw errors on the driver, rather than workers.
        :return: True if parameters are valid
        """
        if not self.isDefined(self.inputCol):
            raise ValueError("Input column must be defined")
        if not self.isDefined(self.outputCol):
            raise ValueError("Output column must be defined")
        return True

    def _validateFitParams(self, params):
        """ Check if an input parameter set is valid """
        if isinstance(params, (list, tuple, dict)):
            assert self.getInputCol() not in params, \
                "params {} cannot contain input column name {}".format(params, self.getInputCol())
        return True

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

        # Extract features
        localFeatures = []
        rows = image_df.collect()
        for row in rows:
            spimg = row[tmp_image_col]
            features = imageStructToArray(spimg)
            localFeatures.append(features)

        if not localFeatures:  # NOTE(phi-dbq): pep-8 recommended against testing 0 == len(array)
            raise ValueError("Cannot extract any feature from dataset!")
        X = np.stack(localFeatures, axis=0)

        # Extract labels
        y = None
        if label_col is not None:
            label_schema = image_df.schema[label_col]
            label_dtype = label_schema.dataType
            assert isinstance(label_dtype, spla.VectorUDT), \
                "must encode labels in one-hot vector format, but got {}".format(label_dtype)

            localLabels = []
            for row in rows:
                try:
                    _keras_label = row[label_col].array
                except ValueError:
                    raise ValueError("Cannot extract encoded label array")
                localLabels.append(_keras_label)

            if not localLabels:
                raise ValueError("Failed to load any labels from dataset, but labels are required")

            y = np.stack(localLabels, axis=0)
            assert y.shape[0] == X.shape[0], \
                "number of features {} != number of labels {}".format(X.shape[0], y.shape[0])

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

        # Obtain params for this estimator instance
        baseParamMap = self.extractParamMap()
        baseParamDict = dict([(param.name, val) for param, val in baseParamMap.items()])
        baseParamDictBc = sc.broadcast(baseParamDict)

        def _local_fit(override_param_map):
            """
            Fit locally a model with a combination of this estimator's param,
            with overriding parameters provided by the input.
            :param override_param_map: dict, key type is MLllib Param
                                       They are meant to override the base estimator's params.
            :return: serialized Keras HDF5 file bytes
            """
            # Update params
            params = baseParamDictBc.value
            override_param_dict = dict([
                (param.name, val) for param, val in override_param_map.items()])
            params.update(override_param_dict)

            # Create Keras model
            model = kmutil.bytes_to_model(modelBytesBc.value)
            model.compile(optimizer=params['kerasOptimizer'], loss=params['kerasLoss'])

            # Retrieve features and labels and fit Keras model
            features = localFeaturesBc.value
            labels = None if localLabelsBc is None else localLabelsBc.value
            _fit_params = params['kerasFitParams']
            model.fit(x=features, y=labels, **_fit_params)

            return kmutil.model_to_bytes(model)

        kerasModelBytesRDD = paramMapsRDD.map(lambda paramMap: (paramMap, _local_fit(paramMap)))
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

    def _fit(self, dataset):  # pylint: disable=unused-argument
        err_msgs = ["This function should not have been called",
                    "Please contact library maintainers to file a bug"]
        raise NotImplementedError('\n'.join(err_msgs))
