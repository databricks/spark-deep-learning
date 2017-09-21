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

import tensorflow as tf

import sparkdl.graph.utils as tfx

from ..tests import PythonUnitTestCase

class TFeXtensionGraphUtilsTest(PythonUnitTestCase):

    def test_infer_graph_element_names(self):
        for tnsr_idx in range(17):
            op_name = 'someOp'
            tnsr_name = '{}:{}'.format(op_name, tnsr_idx)
            self.assertEqual(op_name, tfx.as_op_name(tnsr_name))
            self.assertEqual(tnsr_name, tfx.as_tensor_name(tnsr_name))

        with self.assertRaises(TypeError):
            for wrong_value in [7, 1.2, tf.Graph()]:
                tfx.as_op_name(wrong_value)
                tfx.as_tensor_name(wrong_value)

    def test_get_graph_elements(self):
        op_name = 'someConstOp'
        tnsr_name = '{}:0'.format(op_name)
        tnsr = tf.constant(1427.08, name=op_name)
        graph = tnsr.graph

        self.assertEqual(op_name, tfx.as_op_name(tnsr))
        self.assertEqual(op_name, tfx.op_name(graph, tnsr))
        self.assertEqual(tnsr_name, tfx.as_tensor_name(tnsr))
        self.assertEqual(tnsr_name, tfx.tensor_name(graph, tnsr))
        self.assertEqual(tnsr, tfx.get_tensor(graph, tnsr))
        self.assertEqual(tnsr.op, tfx.get_op(graph, tnsr))
        self.assertEqual(graph, tfx.get_op(graph, tnsr).graph)
        self.assertEqual(graph, tfx.get_tensor(graph, tnsr).graph)
