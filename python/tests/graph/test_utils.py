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

import tensorflow as tf

import sparkdl.graph.utils as tfx

from ..tests import PythonUnitTestCase

TestCase = namedtuple('TestCase', ['data', 'description'])


def _gen_tensor_op_string_input_tests():
    op_name = 'someOp'
    for tnsr_idx in [0, 1, 2, 3, 5, 8, 15, 17]:
        tnsr_name = '{}:{}'.format(op_name, tnsr_idx)
        yield TestCase(data=(op_name, tfx.op_name(tnsr_name)),
                       description='test tensor name to op name')
        yield TestCase(data=(tnsr_name, tfx.tensor_name(tnsr_name)),
                       description='test tensor name to tensor name')


def _gen_invalid_tensor_or_op_input_with_wrong_types():
    for wrong_val in [7, 1.2, tf.Graph()]:
        yield TestCase(data=wrong_val, description='wrong type {}'.format(type(wrong_val)))


def _gen_invalid_tensor_or_op_with_graph_pairing():
    tnsr = tf.constant(1427.08, name='someConstOp')
    other_graph = tf.Graph()
    op_name = tnsr.op.name

    # Test get_tensor and get_op with non-associated tensor/op and graph inputs
    _comm_suffix = ' with wrong graph'
    yield TestCase(data=lambda: tfx.get_op(tnsr, other_graph),
                   description='test get_op from tensor' + _comm_suffix)
    yield TestCase(data=lambda: tfx.get_tensor(tnsr, other_graph),
                   description='test get_tensor from tensor' + _comm_suffix)
    yield TestCase(data=lambda: tfx.get_op(tnsr.name, other_graph),
                   description='test get_op from tensor name' + _comm_suffix)
    yield TestCase(data=lambda: tfx.get_tensor(tnsr.name, other_graph),
                   description='test get_tensor from tensor name' + _comm_suffix)
    yield TestCase(data=lambda: tfx.get_op(tnsr.op, other_graph),
                   description='test get_op from op' + _comm_suffix)
    yield TestCase(data=lambda: tfx.get_tensor(tnsr.op, other_graph),
                   description='test get_tensor from op' + _comm_suffix)
    yield TestCase(data=lambda: tfx.get_op(op_name, other_graph),
                   description='test get_op from op name' + _comm_suffix)
    yield TestCase(data=lambda: tfx.get_tensor(op_name, other_graph),
                   description='test get_tensor from op name' + _comm_suffix)


def _gen_valid_tensor_op_input_combos():
    op_name = 'someConstOp'
    tnsr_name = '{}:0'.format(op_name)
    tnsr = tf.constant(1427.08, name=op_name)
    graph = tnsr.graph

    # Test for op_name
    yield TestCase(data=(op_name, tfx.op_name(tnsr)),
                   description='get op name from tensor (no graph)')
    yield TestCase(data=(op_name, tfx.op_name(tnsr, graph)),
                   description='get op name from tensor (with graph)')
    yield TestCase(data=(op_name, tfx.op_name(tnsr_name)),
                   description='get op name from tensor name (no graph)')
    yield TestCase(data=(op_name, tfx.op_name(tnsr_name, graph)),
                   description='get op name from tensor name (with graph)')
    yield TestCase(data=(op_name, tfx.op_name(tnsr.op)),
                   description='get op name from op (no graph)')
    yield TestCase(data=(op_name, tfx.op_name(tnsr.op, graph)),
                   description='get op name from op (with graph)')
    yield TestCase(data=(op_name, tfx.op_name(op_name)),
                   description='get op name from op name (no graph)')
    yield TestCase(data=(op_name, tfx.op_name(op_name, graph)),
                   description='get op name from op name (with graph)')

    # Test for tensor_name
    yield TestCase(data=(tnsr_name, tfx.tensor_name(tnsr)),
                   description='get tensor name from tensor (no graph)')
    yield TestCase(data=(tnsr_name, tfx.tensor_name(tnsr, graph)),
                   description='get tensor name from tensor (with graph)')
    yield TestCase(data=(tnsr_name, tfx.tensor_name(tnsr_name)),
                   description='get tensor name from tensor name (no graph)')
    yield TestCase(data=(tnsr_name, tfx.tensor_name(tnsr_name, graph)),
                   description='get tensor name from tensor name (with graph)')
    yield TestCase(data=(tnsr_name, tfx.tensor_name(tnsr.op)),
                   description='get tensor name from op (no graph)')
    yield TestCase(data=(tnsr_name, tfx.tensor_name(tnsr.op, graph)),
                   description='get tensor name from op (with graph)')
    yield TestCase(data=(tnsr_name, tfx.tensor_name(tnsr_name)),
                   description='get tensor name from op name (no graph)')
    yield TestCase(data=(tnsr_name, tfx.tensor_name(tnsr_name, graph)),
                   description='get tensor name from op name (with graph)')

    # Test for get_tensor
    yield TestCase(data=(tnsr, tfx.get_tensor(tnsr, graph)),
                   description='get tensor from tensor')
    yield TestCase(data=(tnsr, tfx.get_tensor(tnsr_name, graph)),
                   description='get tensor from tensor name')
    yield TestCase(data=(tnsr, tfx.get_tensor(tnsr.op, graph)),
                   description='get tensor from op')
    yield TestCase(data=(tnsr, tfx.get_tensor(op_name, graph)),
                   description='get tensor from op name')

    # Test for get_op
    yield TestCase(data=(tnsr.op, tfx.get_op(tnsr, graph)),
                   description='get op from tensor')
    yield TestCase(data=(tnsr.op, tfx.get_op(tnsr_name, graph)),
                   description='get op from tensor name')
    yield TestCase(data=(tnsr.op, tfx.get_op(tnsr.op, graph)),
                   description='get op from op')
    yield TestCase(data=(tnsr.op, tfx.get_op(op_name, graph)),
                   description='test op from op name')


class TFeXtensionGraphUtilsTest(PythonUnitTestCase):
    @parameterized.expand(_gen_tensor_op_string_input_tests)
    def test_valid_tensor_op_name_inputs(self, data, description):
        """ Must get correct names from valid graph element names """
        name_a, name_b = data
        self.assertEqual(name_a, name_b, msg=description)

    @parameterized.expand(_gen_invalid_tensor_or_op_input_with_wrong_types)
    def test_invalid_tensor_name_inputs_with_wrong_types(self, data, description):
        """ Must fail when provided wrong types """
        with self.assertRaises(TypeError, msg=description):
            tfx.tensor_name(data)

    @parameterized.expand(_gen_invalid_tensor_or_op_input_with_wrong_types)
    def test_invalid_op_name_inputs_with_wrong_types(self, data, description):
        """ Must fail when provided wrong types """
        with self.assertRaises(TypeError, msg=description):
            tfx.op_name(data)

    @parameterized.expand(_gen_invalid_tensor_or_op_input_with_wrong_types)
    def test_invalid_op_inputs_with_wrong_types(self, data, description):
        """ Must fail when provided wrong types """
        with self.assertRaises(TypeError, msg=description):
            tfx.get_op(data, tf.Graph())

    @parameterized.expand(_gen_invalid_tensor_or_op_input_with_wrong_types)
    def test_invalid_tensor_inputs_with_wrong_types(self, data, description):
        """ Must fail when provided wrong types """
        with self.assertRaises(TypeError, msg=description):
            tfx.get_tensor(data, tf.Graph())

    @parameterized.expand(_gen_valid_tensor_op_input_combos)
    def test_valid_tensor_op_object_inputs(self, data, description):
        """ Must get correct graph elements from valid graph elements or their names """
        tfobj_or_name_a, tfobj_or_name_b = data
        self.assertEqual(tfobj_or_name_a, tfobj_or_name_b, msg=description)

    @parameterized.expand(_gen_invalid_tensor_or_op_with_graph_pairing)
    def test_invalid_tensor_op_object_graph_pairing(self, data, description):
        """ Must fail with non-associated tensor/op and graph inputs """
        with self.assertRaises((KeyError, AssertionError, TypeError), msg=description):
            data()
