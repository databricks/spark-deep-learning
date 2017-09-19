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

if sys.version_info[:2] <= (2, 6):
    try:
        import unittest2 as unittest
    except ImportError:
        sys.stderr.write('Please install unittest2 to test with Python 2.6 or earlier')
        sys.exit(1)
else:
    import unittest

from sparkdl.param.converters import SparkDLTypeConverters as conv

class ParamsConverterTest(unittest.TestCase):
    # pylint: disable=protected-access

    def test_tf_input_mapping_converter(self):
        valid_tnsr_input = {'colA': 'tnsrOpA:0',
                            'colB': 'tnsrOpB:0'}
        valid_op_input = {'colA': 'tnsrOpA',
                          'colB': 'tnsrOpB'}
        valid_input_mapping_result = [('colA', 'tnsrOpA:0'),
                                      ('colB', 'tnsrOpB:0')]

        for valid_input_mapping in [valid_op_input, valid_tnsr_input]:
            res = conv.asColumnToTensorNameMap(valid_input_mapping)
            self.assertEqual(valid_input_mapping_result, res)

    def test_tf_output_mapping_converter(self):
        valid_tnsr_output = {'tnsrOpA:0': 'colA',
                             'tnsrOpB:0': 'colB'}
        valid_op_output = {'tnsrOpA': 'colA',
                           'tnsrOpB': 'colB'}
        valid_output_mapping_result = [('tnsrOpA:0', 'colA'),
                                       ('tnsrOpB:0', 'colB')]

        for valid_output_mapping in [valid_tnsr_output, valid_op_output]:
            res = conv.asTensorNameToColumnMap(valid_output_mapping)
            self.assertEqual(valid_output_mapping_result, res)


    def test_invalid_input_mapping(self):
        for invalid in [['a1', 'b2'], ('c3', 'd4'), [('a', 1), ('b', 2)],
                        {1: 'a', 2.0: 'b'}, {'a': 1, 'b': 2.0}]:
            with self.assertRaises(TypeError):
                conv.asColumnToTensorNameMap(invalid)
                conv.asTensorNameToColumnMap(invalid)

        with self.assertRaises(TypeError):
            # Wrong containter type: only accept dict
            conv.asColumnToTensorNameMap([('colA', 'tnsrA:0'), ('colB', 'tnsrB:0')])
            conv.asTensorNameToColumnMap([('colA', 'tnsrA:0'), ('colB', 'tnsrB:0')])
            conv.asColumnToTensorNameMap([('tnsrA:0', 'colA'), ('tnsrB:0', 'colB')])
            conv.asTensorNameToColumnMap([('tnsrA:0', 'colA'), ('tnsrB:0', 'colB')])
