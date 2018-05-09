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

import threading
import numpy as np

from pyspark.ml import Estimator
import pyspark.ml.linalg as spla

from sparkdl.image.imageIO import imageStructToArray
from sparkdl.param import (
    keyword_only, CanLoadImage, HasKerasModel, HasKerasOptimizer, HasKerasLoss, HasOutputMode,
    HasInputCol, HasLabelCol, HasOutputCol)
from sparkdl.transformers.keras_image import KerasImageFileTransformer
import sparkdl.utils.jvmapi as JVMAPI
import sparkdl.utils.keras_model as kmutil

__all__ = ['KerasImageFileEstimator']


# pylint: disable=too-few-public-methods
class _ThreadSafeIterator(object):
    """
    Utility iterator class used by KerasImageFileEstimator.fitMultiple to serve models in a thread
    safe manner.

    >>> list(_ThreadSafeIterator(range(5)))
    [0, 1, 2, 3, 4]

    >>> from multiprocessing import Pool
    >>> def f(x):
    >>>     return x
    >>> p = Pool(5)
    >>> p.map(f, _ThreadSafeIterator(range(5)))
    [0, 1, 2, 3, 4]
    """
    def __init__(self, models):
        """
        :param models: iterator of objects to be iterated safely
        """
        self.models = list(models)
        self.counter = 0
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            index = self.counter
            if index >= len(self.models):
                raise StopIteration("No models remaining.")
            self.counter += 1
        return self.models[index]

    def next(self):
        """For python2 compatibility."""
        return self.__next__()


class KerasImageFileEstimator(Estimator, HasInputCol, HasOutputCol, HasLabelCol, HasKerasModel,
                              HasKerasOptimizer, HasKerasLoss, CanLoadImage, HasOutputMode):
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
    def __init__(self, inputCol=None, outputCol=None, outputMode="vector", labelCol=None,
                 modelFile=None, imageLoader=None, kerasOptimizer=None, kerasLoss=None,
                 kerasFitParams=None):
        """
        __init__(self, inputCol=None, outputCol=None, outputMode="vector", labelCol=None,
                 modelFile=None, imageLoader=None, kerasOptimizer=None, kerasLoss=None,
                 kerasFitParams=None)
        """
        # NOTE(phi-dbq): currently we ignore output mode, as the actual output are the
        #                trained models and the Transformers built from them.
        super(KerasImageFileEstimator, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        self._tunable_params = [self.kerasOptimizer, self.kerasLoss, self.kerasFitParams,
                                self.outputCol, self.outputMode]  # model params and output params

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, outputMode="vector", labelCol=None,
                  modelFile=None, imageLoader=None, kerasOptimizer=None, kerasLoss=None,
                  kerasFitParams=None):
        """
        setParams(self, inputCol=None, outputCol=None, outputMode="vector", labelCol=None,
                  modelFile=None, imageLoader=None, kerasOptimizer=None, kerasLoss=None,
                  kerasFitParams=None)
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _validateParams(self, paramMap):
        """
        Check Param values so we can throw errors on the driver, rather than workers.
        :param paramMap: Dict[pyspark.ml.param.Param, object]
        :return: True if parameters are valid
        """

        undefined = set([p for p in self.params if not self.isDefined(p)])
        undefined_tunable = undefined.intersection(self._tunable_params)
        failed_define = [p.name for p in undefined.difference(undefined_tunable)]
        failed_tune = [p.name for p in undefined_tunable if p not in paramMap]
        untunable_overrides = [p.name for p in paramMap if p not in self._tunable_params]

        if failed_define or failed_tune or untunable_overrides:
            msg = "Following Params must be"
            if failed_define:
                msg += " defined: " + str(failed_define)
            if failed_tune:
                msg += " defined or tuned: " + str(failed_tune)
            if untunable_overrides:
                msg += " not tuned: " + str(untunable_overrides)
            raise ValueError(msg)

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
                    _keras_label = row[label_col].toArray()
                except ValueError:
                    raise ValueError("Cannot extract encoded label array")
                localLabels.append(_keras_label)

            if not localLabels:
                raise ValueError("Failed to load any labels from dataset, but labels are required")

            y = np.stack(localLabels, axis=0)
            assert y.shape[0] == X.shape[0], \
                "number of features {} != number of labels {}".format(X.shape[0], y.shape[0])

        return X, y

    def _collectModels(self, kerasModelBytesRDD):
        """
        Collect Keras models on workers to MLlib Models on the driver.
        :param kerasModelBytesRDD: RDD of (param_map, model_bytes) tuples
        :return: generator of (index, MLlib model) tuples
        """
        for (i, param_map, model_bytes) in kerasModelBytesRDD.collect():
            model_filename = kmutil.bytes_to_h5file(model_bytes)
            yield i, self._copyValues(KerasImageFileTransformer(modelFile=model_filename),
                                      extra=param_map)

    def fitMultiple(self, dataset, paramMaps):
        """
        Fits len(paramMaps) models in parallel, one in each Spark task.
        :param dataset: input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`.
                        The column `inputCol` should be of type `sparkdl.image.imageIO.imgSchema`.
        :param paramMaps: non-empty list or tuple of ParamMaps (dict values)
        :return: an iterable which contains one model for each param map. Each call to
                 `next(modelIterator)` will return `(index, model)` where model was fit using
                 `paramMaps[index]`. `index` values may not be sequential.
        .. warning:: This serializes each model into an HDF5 byte file to the driver. If the model
                     file is large, the driver might go out-of-memory. As we cannot assume the
                     existence of a sufficiently large (and writable) file system, users are
                     advised to not train too many models in a single Spark job.
        """
        _ = [self._validateParams(pm) for pm in paramMaps]

        def _get_tunable_name_value_map(param_map, tunable):
            """takes a dictionary {`Param` -> value} and a list [`Param`], select keys that are
            present in both and returns a map of {Param.name -> value}"""
            return {param.name: val for param, val in param_map.items() if param in tunable}

        sc = JVMAPI._curr_sc()
        param_name_maps = [(i, _get_tunable_name_value_map(pm, self._tunable_params))
                           for (i, pm) in enumerate(paramMaps)]
        num_models = len(param_name_maps)
        paramNameMapsRDD = sc.parallelize(param_name_maps, numSlices=num_models)

        # Extract image URI from provided dataset and create features as numpy arrays
        localFeatures, localLabels = self._getNumpyFeaturesAndLabels(dataset)
        localFeaturesBc = sc.broadcast(localFeatures)
        localLabelsBc = None if localLabels is None else sc.broadcast(localLabels)

        # Broadcast Keras model (HDF5) file content as bytes
        modelBytes = self._loadModelAsBytes()
        modelBytesBc = sc.broadcast(modelBytes)

        # Obtain params for this estimator instance
        base_params = _get_tunable_name_value_map(self.extractParamMap(), self._tunable_params)
        baseParamsBc = sc.broadcast(base_params)

        def _local_fit(row):
            """
            Fit locally a model with a combination of this estimator's param,
            with overriding parameters provided by the input.
            :param row: a list or tuple containing index and override_param_map. Index is an int
                        representing the index of parameter map and override_param_map is a dict
                        whose key is a string representing an MLllib Param Name. These are meant
                        to override the base estimator's params.
            :return: tuple of index, override_param_map and serialized Keras HDF5 file bytes
            """
            index, override_param_map = row
            # Update params
            params = baseParamsBc.value
            params.update(override_param_map)

            # Create Keras model
            model = kmutil.bytes_to_model(modelBytesBc.value)
            model.compile(optimizer=params['kerasOptimizer'], loss=params['kerasLoss'])

            # Retrieve features and labels and fit Keras model
            features = localFeaturesBc.value
            labels = None if localLabelsBc is None else localLabelsBc.value
            _fit_params = params['kerasFitParams']
            model.fit(x=features, y=labels, **_fit_params)

            return index, override_param_map, kmutil.model_to_bytes(model)

        kerasModelBytesRDD = paramNameMapsRDD.map(_local_fit)
        models = self._collectModels(kerasModelBytesRDD)

        return _ThreadSafeIterator(models)

    def _loadModelAsBytes(self):
        """
        (usable on driver only)
        Load the Keras model file as a byte string.
        :return: str containing the model data
        """
        with open(self.getModelFile(), mode='rb') as fin:
            fileContent = fin.read()
        return fileContent

    def _fit(self, dataset):
        tuples = self.fitMultiple(dataset, [{}])
        return dict(tuples)[0]
