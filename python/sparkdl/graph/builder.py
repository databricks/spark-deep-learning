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
import json
import logging
import os
import shutil
import six
from tempfile import mkdtemp

import keras.backend as K
from keras.models import Model as KerasModel, load_model
import tensorflow as tf

import tensorframes.core as tfrm

import sparkdl.graph.utils as tfx
from sparkdl.utils import jvmapi as JVMAPI

logger = logging.getLogger('sparkdl')

class IsolatedSession(object):
    """
    Building TensorFlow graph and export as either UDF or GraphFunction

    It provides a GraphBuilder object which provides
    - importing existing GraphFunction object as a subgraph
    - exporting current graph as an GraphFunction object

    This is a thin layer on top of `tf.Session`.

    :param g: Graph, use the provided TensorFlow graph as default graph
    :param keras: bool, whether to also let Keras TensorFlow backend use this session
    """
    def __init__(self, graph=None, keras=False):
        self.graph = graph or tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        if keras:
            self.keras_prev_sess = K.get_session()
        else:
            self.keras_prev_sess = None

    def __enter__(self):
        self.sess.as_default()
        self.sess.__enter__()
        if self.keras_prev_sess is not None:
            K.set_session(self.sess)
        return self

    def __exit__(self, *args):
        if self.keras_prev_sess is not None:
            K.set_session(self.keras_prev_sess)
        self.sess.__exit__(*args)

    def run(self, *args, **kargs):
        return self.sess.run(*args, **kargs)

    def asUDF(self, udf_name, fetches, feed_dict=None, blocked=False, register=True):
        """ 
        Make the TensorFlow graph as a SQL UDF 

        :param udf_name: str, name of the SQL UDF 
        :param fetches: list, output tensors of the graph
        :param feed_dict: dict, a dictionary that maps graph elements to input values
        :param blocked: bool, whether the TensorFrame execution should be blocked based or row based
        :param register: bool, whether this UDF should be registered
        :return: JVM function handle object
        """ 
        # pylint: disable=W0212
        # TODO: work with registered expansions
        jvm_builder = JVMAPI.forClass(JVMAPI.MODEL_FACTORY_CLASSNAME)
        tfrm._add_graph(self.graph, jvm_builder)

        fetch_names = [tfx.tensor_name(self.graph, tnsr) for tnsr in fetches]
        fetch_shapes = [tfx.get_shape(self.graph, tnsr) for tnsr in fetches]
        placeholder_names = []
        placeholder_shapes = []

        for node in self.graph.as_graph_def(add_shapes=True).node:
            if len(node.input) == 0 and str(node.op) == 'Placeholder':
                tnsr_name = tfx.tensor_name(self.graph, node.name)
                tnsr = self.graph.get_tensor_by_name(tnsr_name)
                try:
                    tnsr_shape = tfx.get_shape(self.graph, tnsr)
                    placeholder_names.append(tnsr_name)
                    placeholder_shapes.append(tnsr_shape)
                except ValueError:
                    pass

        jvm_builder.shape(fetch_names + placeholder_names, fetch_shapes + placeholder_shapes)
        jvm_builder.fetches(fetch_names)
        placeholder_op_names = [tfx.op_name(self.graph, tnsr_name) for tnsr_name in placeholder_names]
        tfrm._add_inputs(jvm_builder, feed_dict, placeholder_op_names)

        if register:
            return jvm_builder.registerUDF(udf_name, blocked)
        else:
            return jvm_builder.makeUDF(udf_name, blocked)
                
    def asGraphFunction(self, inputs, outputs, strip_and_freeze=True):
        """
        Export the graph in this session as a GraphFunction object

        :param inputs: list, graph elements representing the inputs
        :param outputs: list, graph elements representing the outputs
        :param strip_and_freeze: bool, should we remove unused part of the graph and freee its values
        """
        if strip_and_freeze:
            gdef = tfx.strip_and_freeze_until(outputs, self.graph, self.sess)
        else:
            gdef = self.graph.as_graph_def(add_shapes=True)
        return GraphFunction(graph_def=gdef,
                             input_names=[tfx.validated_input(self.graph, elem) for elem in inputs],
                             output_names=[tfx.validated_output(self.graph, elem) for elem in outputs])

    def importGraphFunction(self, gfn, input_map=None, name="GFN-IMPORT", **gdef_kargs):
        """
        Import a GraphFunction object into the current session

        :param gfn: GraphFunction, an object representing a TensorFlow graph and its inputs and outputs
        :param input_map: dict, mapping from input names to existing graph elements
        :param name: str, the scope for all the variables in the GraphFunction's elements
        :param gdef_kargs: other keyword elements for TensorFlow's `import_graph_def`
        """
        try:
            del gdef_kargs["return_elements"]
        except KeyError:
            pass
        if input_map is not None:
            assert set(input_map.keys()) <= set(gfn.input_names), \
                "cannot locate provided input elements in the graph"

        input_names = gfn.input_names
        output_names = gfn.output_names
        if name is not None:
            name = name.strip()
            if len(name) > 0:
                output_names = [
                    name + '/' + op_name for op_name in gfn.output_names]
                input_names = [
                    name + '/' + op_name for op_name in gfn.input_names]

        # When importing, provide the original output op names
        tf.import_graph_def(gfn.graph_def,
                            input_map=input_map,
                            return_elements=gfn.output_names,
                            name=name,
                            **gdef_kargs)
        feeds = [tfx.get_tensor(self.graph, name) for name in input_names]
        fetches = [tfx.get_tensor(self.graph, name) for name in output_names]
        return (feeds, fetches)


class GraphFunction(object):
    """
    Represent a TensorFlow graph with its GraphDef, input and output operation names.

    :param graph_def: GraphDef, a static ProtocolBuffer object holding informations of a TensorFlow graph
    :param input_names: names to the input graph elements (must be of Placeholder type)
    :param output_names: names to the output graph elements
    """

    def __init__(self, graph_def, input_names, output_names):
        """
        :param graph_def: GraphDef object
        :param input_names: list of input (operation) names (must be typed `Placeholder`)
        :param output_names: list of output (operation) names
        """
        self.graph_def = graph_def
        self.input_names = input_names
        self.output_names = output_names

    def dump(self, fpath):
        """
        Store the GraphFunction to a file

        :param fpath: str or path, path to the serialized GraphFunction
        """
        assert isinstance(fpath, six.string_types)

        gdef_bytes_fpath = "{}.gdef.bytes".format(fpath)
        with open(gdef_bytes_fpath, 'wb') as fout:
            gdef_bytes = self.graph_def.SerializeToString()
            fout.write(gdef_bytes)

        serialized = {"graph_def_file": gdef_bytes_fpath,
                      "inputs": self.input_names,
                      "outputs": self.output_names}
        with open(fpath, 'w') as fout:
            json.dump(serialized, fout)

    @classmethod
    def fromSerialized(cls, fpath):
        """
        Load an existing GraphFunction from file.

        :param fpath: str or path, path to the serialized GraphFunction
        """
        with open(str(fpath)) as fin:
            serialized = json.load(fin)
        assert set(['inputs', 'graph_def_file', 'outputs']) <= set(serialized.keys())

        gdef_bytes_fpath = serialized["graph_def_file"]
        assert os.path.exists(gdef_bytes_fpath), \
            "TensorFlow GraphDef binary file must be found"

        with open(gdef_bytes_fpath, 'rb') as fin:
            gdef_bytes = fin.read()
            gdef = tf.GraphDef.FromString(gdef_bytes)  # pylint: disable=E1101

        gfn = cls(graph_def=gdef,
                  input_names=serialized["inputs"],
                  output_names=serialized["outputs"])

        return gfn

    @classmethod
    def fromKeras(cls, model_or_file_path):
        """ Build a GraphFunction from a Keras model
        """
        def load_model_file(file_path):
            assert file_path.endswith('.h5'), \
                'Keras model must be specified as HDF5 file'

            with IsolatedSession(keras=True) as issn:
                K.set_learning_phase(0) # Testing phase
                model = load_model(file_path)
                gfn = issn.asGraphFunction(model.inputs, model.outputs)

            return gfn

        if isinstance(model_or_file_path, KerasModel):
            model = model_or_file_path
            _tmpdir = mkdtemp(prefix='kera-')
            try:  # Save to tempdir and restore in a new session
                model_path = os.path.join(_tmpdir, "model.h5")
                model.save(model_path, overwrite=True)
                gfn = load_model_file(str(model_path))
            finally:
                shutil.rmtree(_tmpdir, ignore_errors=True)
            return gfn
        elif isinstance(model_or_file_path, six.string_types):
            return load_model_file(model_or_file_path)
        else:
            raise TypeError("input must be a Keras model of a file path")

    @classmethod
    def fromList(cls, functions):
        """
        Takes multiple graph functions and merges them into a single graph function.
        It is assumed that there is only one input and one output in the intermediary layers

        :param functions: a list of tuples (scope name, GraphFunction object).
        """
        assert len(functions) >= 1, ("must provide at least one function", functions)
        if 1 == len(functions):
            return functions[0]
        for (scope_in, gfn_in), (scope_out, gfn_out) in zip(functions[:-1], functions[1:]):
            assert len(gfn_in.output_names) == len(gfn_out.input_names), \
                "graph function link {} -> {} require compatible layers".format(scope_in, scope_out)
            if len(gfn_out.input_names) != 1:
                raise NotImplementedError("Only support single input/output for intermediary layers")

        # Acquire initial placeholders' properties
        with IsolatedSession() as issn:
            _, first_gfn = functions[0]
            feeds, _ = issn.importGraphFunction(first_gfn, name='')
            first_input_info = []
            for tnsr in feeds:
                name = tfx.op_name(issn.graph, tnsr)
                first_input_info.append((tnsr.dtype, tnsr.shape, name))

        # Build a linear chain of all the provide functions
        with IsolatedSession() as issn:
            first_inputs = [tf.placeholder(dtype, shape, name)
                            for (dtype, shape, name) in first_input_info]
            prev_outputs = first_inputs

            for idx, (scope, gfn) in enumerate(functions):
                # Give a scope to each function to avoid name conflict
                if scope is None or len(scope.strip()) == 0:
                    scope = 'GFN-BLK-{}'.format(idx)
                _msg = 'merge: stage {}, scope {}'.format(idx, scope)
                logger.info(_msg)
                input_map = dict(zip(gfn.input_names, prev_outputs))
                _, fetches = issn.importGraphFunction(
                    gfn, name=scope, input_map=input_map)
                prev_outputs = fetches

            # Add a non-scoped output name as the output node
            last_output_names = functions[-1][1].output_names
            last_outputs = []
            for tnsr, name in zip(prev_outputs, last_output_names):
                last_outputs.append(tf.identity(tnsr, name=name))

            gfn = issn.asGraphFunction(first_inputs, last_outputs)

        return gfn
