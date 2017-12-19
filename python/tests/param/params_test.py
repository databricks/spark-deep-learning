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
# Use this to create parameterized test cases
from parameterized import parameterized

from sparkdl.param.converters import SparkDLTypeConverters

from ..tests import PythonUnitTestCase

TestCase = namedtuple('TestCase', ['data', 'description'])

_shared_invalid_test_cases = [
    TestCase(data=['a1', 'b2'], description='required pair but got single element'),
    TestCase(data=('c3', 'd4'), description='required pair but got single element'),
    TestCase(data=[('a', 1), ('b', 2)], description='only accept dict, but got list'),
    TestCase(data={1: 'a', 2.0: 'b'}, description='wrong mapping type'),
    TestCase(data={'a': 1.0, 'b': 2}, description='wrong mapping type'),
]
_col2tnsr_test_cases = _shared_invalid_test_cases + [
    TestCase(data={'colA': 'tnsrOpA', 'colB': 'tnsrOpB'},
             description='tensor name required'),
]
_tnsr2col_test_cases = _shared_invalid_test_cases + [
    TestCase(data={'tnsrOpA': 'colA', 'tnsrOpB': 'colB'},
             description='tensor name required'),
]


class ParamsConverterTest(PythonUnitTestCase):
    """
    Test MLlib Params introduced in Spark Deep Learning Pipeline
    Additional test cases are attached via the meta class `TestGenMeta`.
    """

    def test_tf_input_mapping_converter(self):
        """ Test valid input mapping conversion """
        valid_tnsr_input = {'colA': 'tnsrOpA:0', 'colB': 'tnsrOpB:0'}
        valid_input_mapping_result = [('colA', 'tnsrOpA:0'), ('colB', 'tnsrOpB:0')]

        res = SparkDLTypeConverters.asColumnToTensorNameMap(valid_tnsr_input)
        self.assertEqual(valid_input_mapping_result, res)

    def test_tf_output_mapping_converter(self):
        """ Test valid output mapping conversion """
        valid_tnsr_output = {'tnsrOpA:0': 'colA', 'tnsrOpB:0': 'colB'}
        valid_output_mapping_result = [('tnsrOpA:0', 'colA'), ('tnsrOpB:0', 'colB')]

        res = SparkDLTypeConverters.asTensorNameToColumnMap(valid_tnsr_output)
        self.assertEqual(valid_output_mapping_result, res)

    @parameterized.expand(_col2tnsr_test_cases)
    def test_invalid_input_mapping(self, data, description):
        """ Test invalid column name to tensor name mapping """
        with self.assertRaises(TypeError, msg=description):
            SparkDLTypeConverters.asColumnToTensorNameMap(data)

    @parameterized.expand(_tnsr2col_test_cases)
    def test_invalid_output_mapping(self, data, description):
        """ Test invalid tensor name to column name mapping """
        with self.assertRaises(TypeError, msg=description):
            SparkDLTypeConverters.asTensorNameToColumnMap(data)
