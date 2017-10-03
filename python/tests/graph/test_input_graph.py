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
import itertools

import numpy as np
# Use this to create parameterized test cases
from parameterized import parameterized

from ..tests import PythonUnitTestCase
from .base_test_generators import GenTestCases

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
_CLEAN_UP_TASKS = []

for obj in _TEST_CASES_GENERATORS:
    obj.build_input_graphs()
    _ALL_TEST_CASES += obj.test_cases
    _CLEAN_UP_TASKS.append(obj.tear_down_env)


class TFInputGraphTest(PythonUnitTestCase):
    @classmethod
    def tearDownClass(cls):
        for clean_fn in _CLEAN_UP_TASKS:
            clean_fn()

    @parameterized.expand(_ALL_TEST_CASES)
    def test_tf_input_graph(self, test_fn, description):  # pylint: disable=unused-argument
        """ Test build TFInputGraph from various sources """
        bool_result, err_msg = test_fn()
        self.assertTrue(bool_result, msg=err_msg)
