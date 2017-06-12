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
from . import jvmapi as JVMAPI

logger = logging.getLogger('sparkdl')

def list_to_vector_udf(col):
    return Column(JVMAPI.default().listToVectorFunction(col._jc))  # pylint: disable=W0212

def pipelined_udf(name, ordered_udf_names):
    """ Given a sequence of @ordered_udf_names f1, f2, ..., fn
        Create a pipelined UDF as fn(...f2(f1()))
    """
    assert len(ordered_udf_names) > 1, \
        "must provide more than one ordered udf names"
    JVMAPI.default().pipeline(name, JVMAPI.pyutils().toSeq(ordered_udf_names))

def pyutils():
    return JVMAPI._curr_jvm().PythonUtils
