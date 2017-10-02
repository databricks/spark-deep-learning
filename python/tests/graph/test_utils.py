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
    for tnsr_idx in range(17):
        tnsr_name = '{}:{}'.format(op_name, tnsr_idx)
        yield TestCase(data=(op_name, tfx.op_name(tnsr_name)),
                       description='must get the same op name from its tensor')
        yield TestCase(data=(tnsr_name, tfx.tensor_name(tnsr_name)),
                       description='must get the tensor name from its operation')


def _gen_invalid_tensor_op_input_with_wrong_types():
    for wrong_val in [7, 1.2, tf.Graph()]:
        yield TestCase(data=wrong_val, description='wrong type {}'.format(type(wrong_val)))


def _gen_valid_tensor_op_objects():
    op_name = 'someConstOp'
    tnsr_name = '{}:0'.format(op_name)
    tnsr = tf.constant(1427.08, name=op_name)
    graph = tnsr.graph

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

    yield TestCase(data=(tnsr, tfx.get_tensor(tnsr, graph)),
                   description='get tensor from tensor (with graph)')
    yield TestCase(data=(tnsr, tfx.get_tensor(tnsr_name, graph)),
                   description='get tensor from tensor name (with graph)')
    yield TestCase(data=(tnsr, tfx.get_tensor(tnsr.op, graph)),
                   description='get tensor from op (with graph)')
    yield TestCase(data=(tnsr, tfx.get_tensor(op_name, graph)),
                   description='get tensor from op name (with graph)')

    yield TestCase(data=(tnsr.op, tfx.get_op(tnsr, graph)),
                   description='get op from tensor (with graph)')
    yield TestCase(data=(tnsr.op, tfx.get_op(tnsr_name, graph)),
                   description='get op from tensor name (with graph)')
    yield TestCase(data=(tnsr.op, tfx.get_op(tnsr.op, graph)),
                   description='get op from op (with graph)')
    yield TestCase(data=(tnsr.op, tfx.get_op(op_name, graph)),
                   description='get op from op name (with graph)')

    yield TestCase(data=(graph, tfx.get_op(tnsr, graph).graph),
                   description='get graph from retrieved op (with graph)')
    yield TestCase(data=(graph, tfx.get_tensor(tnsr, graph).graph),
                   description='get graph from retrieved tensor (with graph)')


class TFeXtensionGraphUtilsTest(PythonUnitTestCase):
    @parameterized.expand(_gen_tensor_op_string_input_tests)
    def test_valid_graph_element_names(self, data, description):
        """ Must get correct names from valid graph element names """
        name_a, name_b = data
        self.assertEqual(name_a, name_b, msg=description)

    @parameterized.expand(_gen_invalid_tensor_op_input_with_wrong_types)
    def test_wrong_tensor_types(self, data, description):
        """ Must fail when provided wrong types """
        with self.assertRaises(TypeError):
            tfx.tensor_name(data, msg=description)

    @parameterized.expand(_gen_invalid_tensor_op_input_with_wrong_types)
    def test_wrong_op_types(self, data, description):
        """ Must fail when provided wrong types """
        with self.assertRaises(TypeError):
            tfx.op_name(data, msg=description)

    @parameterized.expand(_gen_valid_tensor_op_objects)
    def test_get_graph_elements(self, data, description):
        """ Must get correct graph elements from valid graph elements or their names """
        tfobj_or_name_a, tfobj_or_name_b = data
        self.assertEqual(tfobj_or_name_a, tfobj_or_name_b, msg=description)
