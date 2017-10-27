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
import sparkdl

if sys.version_info[:2] <= (2, 6):
    try:
        import unittest2 as unittest
    except ImportError:
        sys.stderr.write('Please install unittest2 to test with Python 2.6 or earlier')
        sys.exit(1)
else:
    import unittest

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession


class SparkDLTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sc = SparkContext('local[*]', cls.__name__)
        cls.sql = SQLContext(cls.sc)
        cls.session = SparkSession.builder.getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.session.stop()
        cls.session = None
        cls.sc.stop()
        cls.sc = None
        cls.sql = None

    def assertDfHasCols(self, df, cols = []):
        map(lambda c: self.assertIn(c, df.columns), cols)
