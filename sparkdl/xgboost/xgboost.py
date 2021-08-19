#
# DATABRICKS CONFIDENTIAL & PROPRIETARY
# __________________
#
# Copyright 2020 Databricks, Inc.
# All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property of Databricks, Inc.
# and its suppliers, if any.  The intellectual and technical concepts contained herein are
# proprietary to Databricks, Inc. and its suppliers and may be covered by U.S. and foreign Patents,
# patents in process, and are protected by trade secret and/or copyright law. Dissemination, use,
# or reproduction of this information is strictly forbidden unless prior written permission is
# obtained from Databricks, Inc.
#
# If you view or obtain a copy of this information and believe Databricks, Inc. may not have
# intended it to be made available, please promptly report it to Databricks Legal Department
# @ legal@databricks.com.
#

# pylint: disable=invalid-name
# pylint: disable=useless-super-delegation
# pylint: disable=import-error
# pylint: disable=no-name-in-module
# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-function-args
# pylint: disable=logging-format-interpolation
# pylint: disable=bare-except
# pylint: disable=ungrouped-imports
# pylint: disable=too-many-ancestors

from pyspark.ml import Estimator, Model
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol, HasWeightCol, \
    HasPredictionCol, HasProbabilityCol, HasRawPredictionCol, HasValidationIndicatorCol
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.util import MLReadable, MLWritable


class _XgboostParams(HasFeaturesCol, HasLabelCol, HasWeightCol, HasPredictionCol,
                     HasValidationIndicatorCol):

    missing = Param(
        parent=Params._dummy(),
        name="missing",
        doc="Specify the missing value in the features, default np.nan. " \
            "We recommend using 0.0 as the missing value for better performance. " \
            "Note: In a spark DataFrame, the inactive values in a sparse vector " \
            "mean 0 instead of missing values, unless missing=0 is specified.")

    callbacks = Param(
        parent=Params._dummy(),
        name="callbacks",
        doc="Refer to XGBoost doc of `xgboost.XGBClassifier.fit()` or " \
            "`xgboost.XGBRegressor.fit()` for this param callbacks. " \
            "The callbacks can be arbitrary functions. It is saved using cloudpickle " \
            "which is not a fully self-contained format. It may fail to load with " \
            "different versions of dependencies.")

    num_workers = Param(
        parent=Params._dummy(),
        name="num_workers",
        doc="The number of XGBoost workers. Each XGBoost worker corresponds to one spark task. " \
            "Note: This parameter is only supported on Databricks Runtime 9.0 ML and above.",
        typeConverter=TypeConverters.toInt
    )
    use_gpu = Param(
        parent=Params._dummy(),
        name="use_gpu",
        doc="A boolean variable. Set use_gpu=true if the executors " \
            "are running on GPU instances. Currently, only one GPU per task is supported. " \
            "Note: This parameter is only supported on Databricks Runtime 9.0 ML and above."
    )
    force_repartition = Param(
        parent=Params._dummy(),
        name="force_repartition",
        doc="A boolean variable. Set force_repartition=true if you " \
            "want to force the input dataset to be repartitioned before XGBoost training. " \
            "Note: The auto repartitioning judgement is not fully accurate, so it is recommended " \
            "to have force_repartition be True. " \
            "Note: This parameter is only supported on Databricks Runtime 9.0 ML and above."
    )
    use_external_storage = Param(
        parent=Params._dummy(),
        name="use_external_storage",
        doc="A boolean variable (that is False by default). External storage is a parameter " \
            "for distributed training that allows external storage (disk) to be used when " \
            "you have an exceptionally large dataset. This should be set to false for " \
            "small datasets. Note that base margin and weighting doesn't work if this is True. " \
            "Also note that you may use precision if you use external storage. " \
            "Note: This parameter is only supported on Databricks Runtime 9.0 ML and above."
    )
    external_storage_precision = Param(
        parent=Params._dummy(),
        name="external_storage_precision",
        doc="The number of significant digits for data storage on disk when using external storage. " \
            "Note: This parameter is only supported on Databricks Runtime 9.0 ML and above.",
        typeConverter=TypeConverters.toInt
    )

    baseMarginCol = Param(
        parent=Params._dummy(),
        name="baseMarginCol",
        doc="Specify the base margins of the training and validation dataset. Set " \
            "this value instead of setting `base_margin` and `base_margin_eval_set` " \
            "in the fit method. Note: this parameter is not available for distributed " \
            "training. " \
            "Note: This parameter is only supported on Databricks Runtime 9.0 ML and above.")


class _XgboostEstimator(Estimator, _XgboostParams, MLReadable, MLWritable):

    def __init__(self):
        raise NotImplementedError()

    def _fit(self, dataset):
        raise NotImplementedError()

    def write(self):
        raise NotImplementedError()

    @classmethod
    def read(cls):
        raise NotImplementedError()


class _XgboostModel(Model, _XgboostParams, MLReadable, MLWritable):

    def __init__(self, xgb_sklearn_model=None):
        raise NotImplementedError()

    def get_booster(self):
        """
        Return the `xgboost.core.Booster` instance.
        """
        raise NotImplementedError()

    def write(self):
        raise NotImplementedError()

    @classmethod
    def read(cls):
        raise NotImplementedError()

    def _transform(self, dataset):
        raise NotImplementedError()


class XgboostRegressorModel(_XgboostModel):
    """
    The model returned by :func:`sparkdl.xgboost.XgboostRegressor.fit`

    .. Note:: This API is experimental.
    """
    pass


class XgboostClassifierModel(_XgboostModel, HasProbabilityCol, HasRawPredictionCol):
    """
    The model returned by :func:`sparkdl.xgboost.XgboostClassifier.fit`

    .. Note:: This API is experimental.
    """
    pass


class XgboostRegressor(_XgboostEstimator):
    """
    XgboostRegressor is a PySpark ML estimator. It implements the XGBoost regression
    algorithm based on XGBoost python library, and it can be used in PySpark Pipeline
    and PySpark ML meta algorithms like CrossValidator/TrainValidationSplit/OneVsRest.

    XgboostRegressor automatically supports most of the parameters in
    `xgboost.XGBRegressor` constructor and most of the parameters used in
    `xgboost.XGBRegressor` fit and predict method (see `API docs <https://xgboost.readthedocs\
    .io/en/latest/python/python_api.html#xgboost.XGBRegressor>`_ for details).

    XgboostRegressor doesn't support setting `gpu_id` but support another param `use_gpu`,
    see doc below for more details.

    XgboostRegressor doesn't support setting `base_margin` explicitly as well, but support
    another param called `baseMarginCol`. see doc below for more details.

    XgboostRegressor doesn't support `validate_features` and `output_margin` param.

    :param callbacks: The export and import of the callback functions are at best effort.
        For details, see :py:attr:`sparkdl.xgboost.XgboostRegressor.callbacks` param doc.
    :param missing: The parameter `missing` in XgboostRegressor has different semantics with
        that in `xgboost.XGBRegressor`. For details, see
        :py:attr:`sparkdl.xgboost.XgboostRegressor.missing` param doc.
    :param validationIndicatorCol: For params related to `xgboost.XGBRegressor` training
        with evaluation dataset's supervision, set
        :py:attr:`sparkdl.xgboost.XgboostRegressor.validationIndicatorCol`
        parameter instead of setting the `eval_set` parameter in `xgboost.XGBRegressor`
        fit method.
    :param weightCol: To specify the weight of the training and validation dataset, set
        :py:attr:`sparkdl.xgboost.XgboostRegressor.weightCol` parameter instead of setting
        `sample_weight` and `sample_weight_eval_set` parameter in `xgboost.XGBRegressor`
        fit method.
    :param xgb_model: Set the value to be the instance returned by
        :func:`sparkdl.xgboost.XgboostRegressorModel.get_booster`.
    :param num_workers: Integer that specifies the number of XGBoost workers to use.
        Each XGBoost worker corresponds to one Spark task. This parameter is only
        supported on Databricks Runtime 9.0 ML and above.
    :param use_gpu: Boolean that specifies whether the executors are running on GPU
        instances. This parameter is only supported on Databricks Runtime 9.0 ML and above.
    :param use_external_storage: Boolean that specifices whether you want to use
        external storage when training in a distributed manner. This allows using disk
        as cache. Setting this to true is useful when you want better memory utilization
        but is not needed for small test datasets. This parameter is only supported on
        Databricks Runtime 9.0 ML and above.
    :param baseMarginCol: To specify the base margins of the training and validation
        dataset, set :py:attr:`sparkdl.xgboost.XgboostRegressor.baseMarginCol` parameter
        instead of setting `base_margin` and `base_margin_eval_set` in the
        `xgboost.XGBRegressor` fit method. Note: this isn't available for distributed
        training. This parameter is only supported on Databricks Runtime 9.0 ML and above.

    .. Note:: The Parameters chart above contains parameters that need special handling.
        For a full list of parameters, see entries with `Param(parent=...` below.

    .. Note:: This API is experimental.

    **Examples**

    >>> from sparkdl.xgboost import XgboostRegressor
    >>> from pyspark.ml.linalg import Vectors
    >>> df_train = spark.createDataFrame([
    ...     (Vectors.dense(1.0, 2.0, 3.0), 0, False, 1.0),
    ...     (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1, False, 2.0),
    ...     (Vectors.dense(4.0, 5.0, 6.0), 2, True, 1.0),
    ...     (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 3, True, 2.0),
    ... ], ["features", "label", "isVal", "weight"])
    >>> df_test = spark.createDataFrame([
    ...     (Vectors.dense(1.0, 2.0, 3.0), ),
    ...     (Vectors.sparse(3, {1: 1.0, 2: 5.5}), )
    ... ], ["features"])
    >>> xgb_regressor = XgboostRegressor(max_depth=5, missing=0.0,
    ... validationIndicatorCol='isVal', weightCol='weight',
    ... early_stopping_rounds=1, eval_metric='rmse')
    >>> xgb_reg_model = xgb_regressor.fit(df_train)
    >>> xgb_reg_model.transform(df_test)

    """

    def __init__(self, **kwargs):
        raise NotImplementedError()


class XgboostClassifier(_XgboostEstimator, HasProbabilityCol, HasRawPredictionCol):
    """
    XgboostClassifier is a PySpark ML estimator. It implements the XGBoost classification
    algorithm based on XGBoost python library, and it can be used in PySpark Pipeline
    and PySpark ML meta algorithms like CrossValidator/TrainValidationSplit/OneVsRest.

    XgboostClassifier automatically supports most of the parameters in
    `xgboost.XGBClassifier` constructor and most of the parameters used in
    `xgboost.XGBClassifier` fit and predict method (see `API docs <https://xgboost.readthedocs\
    .io/en/latest/python/python_api.html#xgboost.XGBClassifier>`_ for details).

    XgboostClassifier doesn't support setting `gpu_id` but support another param `use_gpu`,
    see doc below for more details.

    XgboostClassifier doesn't support setting `base_margin` explicitly as well, but support
    another param called `baseMarginCol`. see doc below for more details.

    XgboostClassifier doesn't support setting `output_margin`, but we can get output margin
    from the raw prediction column. See `rawPredictionCol` param doc below for more details.

    XgboostClassifier doesn't support `validate_features` and `output_margin` param.

    :param callbacks: The export and import of the callback functions are at best effort. For
        details, see :py:attr:`sparkdl.xgboost.XgboostClassifier.callbacks` param doc.
    :param missing: The parameter `missing` in XgboostClassifier has different semantics with
        that in `xgboost.XGBClassifier`. For details, see
        :py:attr:`sparkdl.xgboost.XgboostClassifier.missing` param doc.
    :param rawPredictionCol: The `output_margin=True` is implicitly supported by the
        `rawPredictionCol` output column, which is always returned with the predicted margin
        values.
    :param validationIndicatorCol: For params related to `xgboost.XGBClassifier` training with
        evaluation dataset's supervision,
        set :py:attr:`sparkdl.xgboost.XgboostClassifier.validationIndicatorCol`
        parameter instead of setting the `eval_set` parameter in `xgboost.XGBClassifier`
        fit method.
    :param weightCol: To specify the weight of the training and validation dataset, set
        :py:attr:`sparkdl.xgboost.XgboostClassifier.weightCol` parameter instead of setting
        `sample_weight` and `sample_weight_eval_set` parameter in `xgboost.XGBClassifier`
        fit method.
    :param xgb_model: Set the value to be the instance returned by
        :func:`sparkdl.xgboost.XgboostClassifierModel.get_booster`.
    :param num_workers: Integer that specifies the number of XGBoost workers to use.
        Each XGBoost worker corresponds to one Spark task. This parameter is only
        supported on Databricks Runtime 9.0 ML and above.
    :param use_gpu: Boolean that specifies whether the executors are running on GPU
        instances. This parameter is only supported on Databricks Runtime 9.0 ML and above.
    :param use_external_storage: Boolean that specifices whether you want to use
        external storage when training in a distributed manner. This allows using disk
        as cache. Setting this to true is useful when you want better memory utilization
        but is not needed for small test datasets. This parameter is only supported on
        Databricks Runtime 9.0 ML and above.
    :param baseMarginCol: To specify the base margins of the training and validation
        dataset, set :py:attr:`sparkdl.xgboost.XgboostClassifier.baseMarginCol` parameter
        instead of setting `base_margin` and `base_margin_eval_set` in the
        `xgboost.XGBClassifier` fit method. Note: this isn't available for distributed
        training. This parameter is only supported on Databricks Runtime 9.0 ML and above.

    .. Note:: The Parameters chart above contains parameters that need special handling.
        For a full list of parameters, see entries with `Param(parent=...` below.

    .. Note:: This API is experimental.

    **Examples**

    >>> from sparkdl.xgboost import XgboostClassifier
    >>> from pyspark.ml.linalg import Vectors
    >>> df_train = spark.createDataFrame([
    ...     (Vectors.dense(1.0, 2.0, 3.0), 0, False, 1.0),
    ...     (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1, False, 2.0),
    ...     (Vectors.dense(4.0, 5.0, 6.0), 0, True, 1.0),
    ...     (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, True, 2.0),
    ... ], ["features", "label", "isVal", "weight"])
    >>> df_test = spark.createDataFrame([
    ...     (Vectors.dense(1.0, 2.0, 3.0), ),
    ... ], ["features"])
    >>> xgb_classifier = XgboostClassifier(max_depth=5, missing=0.0,
    ...     validationIndicatorCol='isVal', weightCol='weight',
    ...     early_stopping_rounds=1, eval_metric='logloss')
    >>> xgb_clf_model = xgb_classifier.fit(df_train)
    >>> xgb_clf_model.transform(df_test).show()

    """

    def __init__(self, **kwargs):
        raise NotImplementedError()
