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
    To add test cases, implement the test logic in a function
    >>> `def _my_test_impl(): ...`
    then call the following function
    >>>  `_register_test_case(fn_impl=_my_test_impl, name=..., doc=...)`
    """
    def __new__(mcs, name, bases, attrs):
        _add_invalid_col2tnsr_mapping_tests()
        attrs.update(_TEST_FUNCTIONS_REGISTRY)
        return super(TestGenMeta, mcs).__new__(mcs, name, bases, attrs)


# Stores test function name mapped to implementation body
_TEST_FUNCTIONS_REGISTRY = {}

TestCase = namedtuple('TestCase', ['data', 'description'])

def _register_test_case(fn_impl, name, doc):
    """ Add an individual test case """
    fn_impl.__name__ = name
    fn_impl.__doc__ = doc
    _TEST_FUNCTIONS_REGISTRY[name] = fn_impl

def _add_invalid_col2tnsr_mapping_tests():
    """ Create a list of test cases and construct individual test functions for each case """
    shared_test_cases = [
        TestCase(data=['a1', 'b2'], description='required pair but get single element'),
        TestCase(data=('c3', 'd4'), description='required pair but get single element'),
        TestCase(data=[('a', 1), ('b', 2)], description='only accept dict, but get list'),
        TestCase(data={1: 'a', 2.0: 'b'}, description='wrong mapping type'),
        TestCase(data={'a': 1.0, 'b': 2}, description='wrong mapping type'),
    ]

    # Specify test cases for `asColumnToTensorNameMap`
    # Add additional test cases specific to this one
    col2tnsr_test_cases = shared_test_cases + [
        TestCase(data={'colA': 'tnsrOpA', 'colB': 'tnsrOpB'},
                 description='strict tensor name required'),
    ]
    _fn_name_template = 'test_invalid_col2tnsr_{idx}'
    _fn_doc_template = 'Test invalid column => tensor name mapping: {description}'

    for idx, test_case in enumerate(col2tnsr_test_cases):
        # Add the actual test logic here
        def test_fn_impl(self):
            with self.assertRaises(TypeError, msg=test_case.description):
                SparkDLTypeConverters.asColumnToTensorNameMap(test_case.data)

        _name = _fn_name_template.format(idx=idx)
        _doc = _fn_doc_template.format(description=test_case.description)
        _register_test_case(fn_impl=test_fn_impl, name=_name, doc=_doc)


    # Specify tests for `asTensorNameToColumnMap`
    tnsr2col_test_cases = shared_test_cases + [
        TestCase(data={'tnsrOpA': 'colA', 'tnsrOpB': 'colB'},
                 description='strict tensor name required'),
    ]
    _fn_name_template = 'test_invalid_tnsr2col_{idx}'
    _fn_doc_template = 'Test invalid tensor name => column mapping: {description}'

    for idx, test_case in enumerate(tnsr2col_test_cases):
        # Add the actual test logic here
        def test_fn_impl(self):  # pylint: disable=function-redefined
            with self.assertRaises(TypeError, msg=test_case.description):
                SparkDLTypeConverters.asTensorNameToColumnMap(test_case.data)

        _name = _fn_name_template.format(idx=idx)
        _doc = _fn_doc_template.format(description=test_case.description)
        _register_test_case(fn_impl=test_fn_impl, name=_name, doc=_doc)


class ParamsConverterTest(with_metaclass(TestGenMeta, PythonUnitTestCase)):
    """
    Test MLlib Params introduced in Spark Deep Learning Pipeline
    Additional test cases are attached via the meta class `TestGenMeta`.
    """
    # pylint: disable=protected-access

    @classmethod
    def setUpClass(cls):
        print(repr(cls), cls)

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
