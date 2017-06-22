#
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

from __future__ import print_function

from keras.applications import InceptionV3
import tensorflow as tf

from sparkdl.graph.builder import IsolatedSession, GraphFunction

from ..tests import SparkDLTestCase

class GraphFunctionSerializationTest(SparkDLTestCase):

    def test_serialization(self):
        """ Must be able to serialize and deserialize """

        with IsolatedSession() as issn:
            x = tf.placeholder(tf.double, shape=[], name="x")
            z = tf.add(x, 3, name='z')
            gfn = issn.asGraphFunction([x], [z])
        
        gfn.dump("/tmp/test.gfn")
        gfn_reloaded = GraphFunction.fromSerialized("/tmp/test.gfn")

        self.assertEqual(str(gfn.graph_def), str(gfn_reloaded.graph_def))
        self.assertEqual(gfn.input_names, gfn_reloaded.input_names)
        self.assertEqual(gfn.output_names, gfn_reloaded.output_names)

    def test_large_serialization(self):
        """ Must be able to serialize and deserialize large graphs """

        gfn = GraphFunction.fromKeras(InceptionV3(weights="imagenet"))
        gfn.dump("/tmp/test_large.gfn")
        gfn_reloaded = GraphFunction.fromSerialized("/tmp/test_large.gfn")

        self.assertEqual(str(gfn.graph_def), str(gfn_reloaded.graph_def))
        self.assertEqual(gfn.input_names, gfn_reloaded.input_names)
        self.assertEqual(gfn.output_names, gfn_reloaded.output_names)
