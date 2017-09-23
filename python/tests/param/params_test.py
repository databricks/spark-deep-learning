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
from six import with_metaclass

from sparkdl.param.converters import SparkDLTypeConverters

from ..tests import PythonUnitTestCase


class TestGenMeta(type):
    """
    This meta-class add test cases to the main unit-test class.
    """

    def __new__(mcs, name, bases, attrs):
        _add_invalid_col2tnsr_mapping_tests()
        attrs.update(_TEST_FUNCTIONS_REGISTRY)
        return super(TestGenMeta, mcs).__new__(mcs, name, bases, attrs)

# Stores test function name mapped to implementation body
_TEST_FUNCTIONS_REGISTRY = {}

TestCase = namedtuple('TestCase', ['data', 'reason'])


def _add_invalid_col2tnsr_mapping_tests():
    """ Create a list of test cases and construct individual test functions for each case """
    test_cases = [TestCase(data=['a1', 'b2'], reason='required pair but get single element'),
                  TestCase(data=('c3', 'd4'), reason='required pair but get single element'),
                  TestCase(data=[('a', 1), ('b', 2)], reason='only accept dict, but get list'),]

    # Add tests for `asColumnToTensorNameMap`
    for idx, test_case in enumerate(test_cases):

        def test_fn_impl(self):
            with self.assertRaises(TypeError, msg=test_case.reason):
                SparkDLTypeConverters.asColumnToTensorNameMap(test_case.data)

        test_fn_name = 'test_invalid_col2tnsr_{}'.format(idx)
        test_fn_impl.__name__ = test_fn_name
        _desc = 'Test invalid column => tensor name mapping: {}'
        test_fn_impl.__doc__ = _desc.format(test_case.reason)
        _TEST_FUNCTIONS_REGISTRY[test_fn_name] = test_fn_impl

    # Add tests for `asTensorNameToColumnMap`
    for idx, test_case in enumerate(test_cases):

        def test_fn_impl(self):  # pylint: disable=function-redefined
            with self.assertRaises(TypeError, msg=test_case.reason):
                SparkDLTypeConverters.asTensorNameToColumnMap(test_case.data)

        test_fn_name = 'test_invalid_tnsr2col_{}'.format(idx)
        test_fn_impl.__name__ = test_fn_name
        _desc = 'Test invalid tensor name => column mapping: {}'
        test_fn_impl.__doc__ = _desc.format(test_case.reason)
        _TEST_FUNCTIONS_REGISTRY[test_fn_name] = test_fn_impl


class ParamsConverterTest(with_metaclass(TestGenMeta, PythonUnitTestCase)):
    """ Test MLlib Params introduced in Spark Deep Learning Pipeline """
    # pylint: disable=protected-access

    @classmethod
    def setUpClass(cls):
        print(repr(cls), cls)

    def test_tf_input_mapping_converter(self):
        """ Test valid input mapping conversion """
        valid_tnsr_input = {'colA': 'tnsrOpA:0', 'colB': 'tnsrOpB:0'}
        valid_op_input = {'colA': 'tnsrOpA', 'colB': 'tnsrOpB'}
        valid_input_mapping_result = [('colA', 'tnsrOpA:0'), ('colB', 'tnsrOpB:0')]

        for valid_input_mapping in [valid_op_input, valid_tnsr_input]:
            res = SparkDLTypeConverters.asColumnToTensorNameMap(valid_input_mapping)
            self.assertEqual(valid_input_mapping_result, res)

    def test_tf_output_mapping_converter(self):
        """ Test valid output mapping conversion """
        valid_tnsr_output = {'tnsrOpA:0': 'colA', 'tnsrOpB:0': 'colB'}
        valid_op_output = {'tnsrOpA': 'colA', 'tnsrOpB': 'colB'}
        valid_output_mapping_result = [('tnsrOpA:0', 'colA'), ('tnsrOpB:0', 'colB')]

        for valid_output_mapping in [valid_tnsr_output, valid_op_output]:
            res = SparkDLTypeConverters.asTensorNameToColumnMap(valid_output_mapping)
            self.assertEqual(valid_output_mapping_result, res)
