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

"""
Internal utilities module.

This should not get exposed outside.
"""
import logging

from pyspark import SparkContext
from pyspark.sql.column import Column

ENTRYPOINT_CLASSNAME = "com.databricks.sparkdl.python.PythonInterface"

logger = logging.getLogger('sparkdl')

def _java_api_sql(javaClassName, sqlCtx = None):
    """
    Loads the PythonInterface object (lazily, because the spark context needs to be initialized
    first).

    WARNING: this works by setting a hidden variable called SQLContext._active_sql_context
    """
    # I suspect the SQL context is doing crazy things at import, because I was
    # experiencing some issues here.
    from pyspark.sql import SQLContext
    _sc = SparkContext._active_spark_context
    logger.info("Spark context = " + str(_sc))
    if sqlCtx is None:
        _sql = SQLContext._instantiatedContext
    else:
        _sql = sqlCtx
    _jvm = _sc._jvm
    # You cannot simply call the creation of the the class on the _jvm due to classloader issues
    # with Py4J.
    return _jvm.Thread.currentThread().getContextClassLoader().loadClass(javaClassName) \
        .newInstance().sqlContext(_sql._ssql_ctx)

def _api():
    return _java_api_sql(ENTRYPOINT_CLASSNAME)

def list_to_vector_udf(col):
    jc = _api().listToVectorFunction(col._jc)
    return Column(jc)
