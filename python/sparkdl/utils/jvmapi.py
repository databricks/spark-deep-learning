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
import logging

from pyspark import SparkContext, SQLContext
from pyspark.sql.column import Column

# pylint: disable=W0212
PYTHON_INTERFACE_CLASSNAME = "com.databricks.sparkdl.python.PythonInterface"
MODEL_FACTORY_CLASSNAME = "com.databricks.sparkdl.python.GraphModelFactory"

logger = logging.getLogger('sparkdl')


def _curr_sql_ctx(sqlCtx=None):
    _sql_ctx = sqlCtx if sqlCtx is not None else SQLContext._instantiatedContext
    logger.info("Spark SQL Context = %s", _sql_ctx)
    return _sql_ctx


def _curr_sc():
    return SparkContext._active_spark_context


def _curr_jvm():
    return _curr_sc()._jvm


def forClass(javaClassName, sqlCtx=None):
    """
    Loads the JVM API object (lazily, because the spark context needs to be initialized
    first).
    """
    # (tjh) suspect the SQL context is doing crazy things at import, because I was
    # experiencing some issues here.
    # You cannot simply call the creation of the the class on the _jvm
    # due to classloader issues with Py4J.
    jvm_thread = _curr_jvm().Thread.currentThread()
    jvm_class = jvm_thread.getContextClassLoader().loadClass(javaClassName)
    return jvm_class.newInstance().sqlContext(_curr_sql_ctx(sqlCtx)._ssql_ctx)


def pyUtils():
    """
    Exposing Spark PythonUtils
    spark/core/src/main/scala/org/apache/spark/api/python/PythonUtils.scala
    """
    return _curr_jvm().PythonUtils


def default():
    """ Default JVM Python Interface class """
    return forClass(javaClassName=PYTHON_INTERFACE_CLASSNAME)


def createTensorFramesModelBuilder():
    """ Create TensorFrames model builder using the Scala API """
    return forClass(javaClassName=MODEL_FACTORY_CLASSNAME)


def listToMLlibVectorUDF(col):
    """ Map struct column from list to MLlib vector """
    return Column(default().listToMLlibVectorUDF(col._jc))  # pylint: disable=W0212


def registerPipeline(name, ordered_udf_names):
    """
    Given a sequence of @ordered_udf_names f1, f2, ..., fn
    Create a pipelined UDF as fn(...f2(f1()))
    """
    assert len(ordered_udf_names) > 1, \
        "must provide more than one ordered udf names"
    return default().registerPipeline(name, ordered_udf_names)


def registerUDF(name, function_body, schema):
    """ Register a single UDF """
    return _curr_sql_ctx().registerFunction(name, function_body, schema)
