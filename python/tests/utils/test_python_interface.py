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
import sys
import traceback

from pyspark import SparkContext, SQLContext
from pyspark.sql.column import Column
from sparkdl.utils import jvmapi as JVMAPI
from ..tests import SparkDLTestCase


class PythonAPITest(SparkDLTestCase):

    def test_using_api(self):
        """ Must be able to load the API """
        try:
            print(JVMAPI.default())
        except BaseException:
            traceback.print_exc(file=sys.stdout)
            self.fail("failed to load certain classes")

        kls_name = str(JVMAPI.forClass(javaClassName=JVMAPI.PYTHON_INTERFACE_CLASSNAME))
        self.assertEqual(kls_name.split('@')[0], JVMAPI.PYTHON_INTERFACE_CLASSNAME)
