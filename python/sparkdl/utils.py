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

"""
Internal utilities module.

This should not get exposed outside.
"""
import logging

from pyspark import SparkContext, SQLContext
from pyspark.sql.column import Column

logger = logging.getLogger('sparkdl')

class JVMAPI(object):
    # pylint: disable=W0212
    ENTRYPOINT_CLASSNAME = "com.databricks.sparkdl.python.PythonInterface"

    @classmethod
    def _curr_sql_ctx(cls, sqlCtx=None):
        _sql_ctx = sqlCtx if sqlCtx is not None else SQLContext._instantiatedContext
        logger.info("Spark SQL Context = " + str(_sql_ctx))
        return _sql_ctx

    @classmethod
    def _curr_sc(cls):
        return SparkContext._active_spark_context

    @classmethod
    def _curr_jvm(cls):
        return cls._curr_sc()._jvm

    @classmethod
    def for_class(cls, javaClassName, sqlCtx=None):
        """
        Loads the PythonInterface object (lazily, because the spark context needs to be initialized
        first).
        """
        # (tjh) suspect the SQL context is doing crazy things at import, because I was
        # experiencing some issues here.
        # You cannot simply call the creation of the the class on the _jvm
        # due to classloader issues with Py4J.
        jvm_thread = cls._curr_jvm().Thread.currentThread()
        jvm_class = jvm_thread.getContextClassLoader().loadClass(javaClassName)
        return jvm_class.newInstance().sqlContext(cls._curr_sql_ctx(sqlCtx)._ssql_ctx)

    @classmethod
    def default(cls):
        return cls.for_class(javaClassName=cls.ENTRYPOINT_CLASSNAME)

    @classmethod
    def pyutils(cls):
        return cls._curr_jvm().PythonUtils

def list_to_vector_udf(col):
    return Column(JVMAPI.default().listToMLlibVectorUDF(col._jc))  # pylint: disable=W0212

def pipelined_udf(name, ordered_udf_names):
    """ Given a sequence of @ordered_udf_names f1, f2, ..., fn
        Create a pipelined UDF as fn(...f2(f1()))
    """
    assert len(ordered_udf_names) > 1, \
        "must provide more than one ordered udf names"
    JVMAPI.default().registerPipeline(name, ordered_udf_names)
