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

from six import with_metaclass

from sparkdl.param.converters import SparkDLTypeConverters as conv

from ..tests import PythonUnitTestCase


class TestGenInvalidMeta(type):
    def __new__(cls, name, bases, attrs):
        """ implement test cases here """
        test_cases = [['a1', 'b2'], ('c3', 'd4'), [('a', 1), ('b', 2)]]

        def _check_col2tnsr(case):
            def impl(self):
                with self.assertRaises(TypeError):
                    conv.asColumnToTensorNameMap(case)
            return impl

        def _check_tnsr2col(case):
            def impl(self):
                with self.assertRaises(TypeError):
                    conv.asTensorNameToColumnMap(case)
            return impl

        def _add_test_fn(fn_name, fn_impl):
            fn_impl.__name__ = fn_name
            attrs[fn_name] = fn_impl

        for idx, case in enumerate(test_cases):
            _add_test_fn('test_invalid_col2tnsr_{}'.format(idx),
                         _check_col2tnsr(case))
            _add_test_fn('test_invalid_tnsr2col_{}'.format(idx),
                         _check_tnsr2col(case))

        return super(TestGenInvalidMeta, cls).__new__(cls, name, bases, attrs)


class ParamsConverterTest(with_metaclass(TestGenInvalidMeta, PythonUnitTestCase)):
    # pylint: disable=protected-access

    @classmethod
    def setUpClass(cls):
        print(repr(cls), cls)

    def test_tf_input_mapping_converter(self):
        valid_tnsr_input = {'colA': 'tnsrOpA:0', 'colB': 'tnsrOpB:0'}
        valid_op_input = {'colA': 'tnsrOpA', 'colB': 'tnsrOpB'}
        valid_input_mapping_result = [('colA', 'tnsrOpA:0'), ('colB', 'tnsrOpB:0')]

        for valid_input_mapping in [valid_op_input, valid_tnsr_input]:
            res = conv.asColumnToTensorNameMap(valid_input_mapping)
            self.assertEqual(valid_input_mapping_result, res)

    def test_tf_output_mapping_converter(self):
        valid_tnsr_output = {'tnsrOpA:0': 'colA', 'tnsrOpB:0': 'colB'}
        valid_op_output = {'tnsrOpA': 'colA', 'tnsrOpB': 'colB'}
        valid_output_mapping_result = [('tnsrOpA:0', 'colA'), ('tnsrOpB:0', 'colB')]

        for valid_output_mapping in [valid_tnsr_output, valid_op_output]:
            res = conv.asTensorNameToColumnMap(valid_output_mapping)
            self.assertEqual(valid_output_mapping_result, res)
