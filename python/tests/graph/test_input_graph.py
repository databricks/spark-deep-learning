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
from __future__ import absolute_import, division, print_function

from collections import namedtuple

import numpy as np
# Use this to create parameterized test cases
from parameterized import parameterized

from ..tests import PythonUnitTestCase
from .base_utils import GenTestCases

#========================================================================
# Don't have to modify the content below

_TEST_CASES_GENERATORS = []


def _REGISTER_(obj):
    _TEST_CASES_GENERATORS.append(obj)


#========================================================================
# Register all test objects here
_REGISTER_(GenTestCases(vec_size=23, test_batch_size=71))
_REGISTER_(GenTestCases(vec_size=13, test_batch_size=23))
_REGISTER_(GenTestCases(vec_size=5, test_batch_size=17))
#========================================================================

_ALL_TEST_CASES = []
for obj in _TEST_CASES_GENERATORS:
    obj.build_input_graphs()
    _ALL_TEST_CASES += obj.test_cases
    obj.tear_down_env()


class TFInputGraphTest(PythonUnitTestCase):
    @parameterized.expand(_ALL_TEST_CASES)
    def test_tf_input_graph(self, ref_out, tgt_out, description):
        """ Test build TFInputGraph from various methods """
        self.assertTrue(np.allclose(ref_out, tgt_out), msg=description)
