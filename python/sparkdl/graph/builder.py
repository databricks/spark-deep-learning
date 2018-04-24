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
import logging
import os
import shutil
from tempfile import mkdtemp

import keras.backend as K
from keras.models import Model as KerasModel, load_model
import six
import tensorflow as tf

import sparkdl.graph.utils as tfx

logger = logging.getLogger('sparkdl')


class IsolatedSession(object):
    """
    Provide an isolated session to work with mixed Keras and TensorFlow
    graph segments.

    It provides utility functions to
    - importing existing `GraphFunction` object as a subgraph
    - exporting current graph as an `GraphFunction` object

    This is a thin layer on top of `tf.Session`.

    :param graph: `tf.Graph`, use the provided TensorFlow graph as default graph
    :param using_keras: bool, when set to True, attach Keras TensorFlow backend to this session.
                        In this case, all Keras models loaded in this session will be accessible
                        as a subgraph of of `graph`
    """

    def __init__(self, graph=None, using_keras=False):
        self.graph = graph or tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        if using_keras:
            self.using_keras = True
            self.keras_prev_sess = K.get_session()
        else:
            self.using_keras = False
            self.keras_prev_sess = None

    def __enter__(self):
        self.sess.__enter__()
        if self.using_keras:
            K.set_session(self.sess)
        return self

    def __exit__(self, *args):
        if self.using_keras:
            K.set_session(self.keras_prev_sess)
        self.sess.__exit__(*args)

    def run(self, *args, **kargs):
        """
        This method delegate the TensorFlow graph execution
        to the underlying tf.Session object to perform
        one step of graph computation.

        All the parameters are defined according to `tf.Session.run`
        Reference: https://www.tensorflow.org/api_docs/python/tf/Session#run
        """
        return self.sess.run(*args, **kargs)

    def asGraphFunction(self, inputs, outputs, strip_and_freeze=True):
        """
        Export the graph in this session as a :py:class:`GraphFunction` object

        :param inputs: list, graph elements representing the inputs
        :param outputs: list, graph elements representing the outputs
        :param strip_and_freeze: bool, should we remove unused part of the graph and freeze its
        values
        """
        if strip_and_freeze:
            gdef = tfx.strip_and_freeze_until(outputs, self.graph, self.sess)
        else:
            gdef = self.graph.as_graph_def(add_shapes=True)
        input_names = [tfx.validated_input(elem, self.graph) for elem in inputs]
        output_names = [tfx.validated_output(elem, self.graph) for elem in outputs]
        return GraphFunction(graph_def=gdef, input_names=input_names, output_names=output_names)

    def importGraphFunction(self, gfn, input_map=None, prefix="GFN-IMPORT", **gdef_kargs):
        """
        Import a GraphFunction object into the current session.
        The API is similar to :py:meth:`tf.import_graph_def`

        .. _a link: https://www.tensorflow.org/api_docs/python/tf/import_graph_def

        :param gfn: GraphFunction, an object representing a TensorFlow graph and its inputs and
        outputs
        :param input_map: dict, mapping from input names to existing graph elements
        :param prefix: str, the scope for all the variables in the :py:class:`GraphFunction`
        elements

                       .. _a link: https://www.tensorflow.org/programmers_guide/variable_scope

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
        scope_name = prefix
        if prefix:
            scope_name = prefix.strip()
            if scope_name:
                output_names = [scope_name + '/' + op_name for op_name in gfn.output_names]
                input_names = [scope_name + '/' + op_name for op_name in gfn.input_names]

        # When importing, provide the original output op names
        tf.import_graph_def(gfn.graph_def,
                            input_map=input_map,
                            return_elements=gfn.output_names,
                            name=scope_name,
                            **gdef_kargs)
        feeds = [tfx.get_tensor(name, self.graph) for name in input_names]
        fetches = [tfx.get_tensor(name, self.graph) for name in output_names]
        return (feeds, fetches)


class GraphFunction(object):
    """
    Represent a TensorFlow graph with its GraphDef, input and output operation names.

    :param graph_def: GraphDef, a static ProtocolBuffer object holding information of a
    TensorFlow graph
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

    @classmethod
    def _fromKerasModelFile(cls, file_path):
        """
        Load a Keras model from a file path into a `GraphFunction`.

        :param file_path: the (HDF5) file path
        """
        assert file_path.endswith('.h5'), \
            'Keras model must be specified as HDF5 file'

        with IsolatedSession(using_keras=True) as issn:
            K.set_learning_phase(0)  # Testing phase
            model = load_model(file_path)
            gfn = issn.asGraphFunction(model.inputs, model.outputs)

        return gfn

    @classmethod
    def fromKeras(cls, model_or_file_path):
        """
        Build a GraphFunction from a Keras model

        :param model_or_file_path: KerasModel or str, either a Keras model or the file path name
        to one
        """
        if isinstance(model_or_file_path, KerasModel):
            model = model_or_file_path
            _tmpdir = mkdtemp(prefix='kera-')
            try:  # Save to tempdir and restore in a new session
                model_path = os.path.join(_tmpdir, "model.h5")
                model.save(model_path, overwrite=True)
                gfn = cls._fromKerasModelFile(str(model_path))
            finally:
                shutil.rmtree(_tmpdir, ignore_errors=True)
            return gfn
        elif isinstance(model_or_file_path, six.string_types):
            return cls._fromKerasModelFile(model_or_file_path)
        else:
            raise TypeError("input must be a Keras model of a file path")

    @classmethod
    def fromList(cls, functions):
        """
        Construct a single GraphFunction from a list of graph functions.
        Each function in the list corresponds to a stage.

        Each function is also scoped by a scope name, in order to avoid
        variable name conflict and also to make the graph cleaner for visualization.
        If a scope name is not provided, we generate one as `GFN-BLK-<stage_index>`.

        The inputs and outputs are picked out of the scopes, so that users
        will still be able to call the function with the expected inputs/outputs names.

        It is assumed that there is only one input and one output in the intermediary layers

        :param functions: a list of tuples (scope name, GraphFunction object).
        """
        assert len(functions) >= 1, ("must provide at least one function", functions)
        if len(functions) == 1:
            return functions[0]
        # Check against each intermediary layer input output function pairs
        for (scope_in, gfn_in), (scope_out, gfn_out) in zip(functions[:-1], functions[1:]):
            # For stage F => G, the composition G(F(.)) must work, which means
            # the number of outputs for F is equal to the number of inputs for G
            assert len(gfn_in.output_names) == len(gfn_out.input_names), \
                "graph function link {} -> {} require compatible layers".format(scope_in, scope_out)
            # We currently only support single input/output for intermediary stages
            # The functions could still take multi-dimensional tensor, but only one
            if len(gfn_out.input_names) != 1:
                raise NotImplementedError(
                    "Only support single input/output for intermediary layers")

        # Acquire initial placeholders' properties
        # We want the input names of the merged function are not under scoped
        # In this way users of the merged function could still use the input names
        # of the first function to get the correct input tensors.
        first_input_info = []
        with IsolatedSession() as issn:
            _, first_gfn = functions[0]
            feeds, _ = issn.importGraphFunction(first_gfn, prefix='')
            for tnsr in feeds:
                name = tfx.op_name(tnsr, issn.graph)
                first_input_info.append((tnsr.dtype, tnsr.shape, name))
            # TODO: make sure that this graph is not reused to prevent name conflict
            # Report error if the graph is not manipulated by anyone else
            # https://www.tensorflow.org/api_docs/python/tf/Graph#finalize
            issn.graph.finalize()

        # Build a linear chain of all the provide functions
        with IsolatedSession() as issn:
            first_inputs = [tf.placeholder(dtype, shape, name)
                            for (dtype, shape, name) in first_input_info]
            prev_outputs = first_inputs

            for idx, (scope, gfn) in enumerate(functions):
                # Give a scope to each function to avoid name conflict
                if scope is None or len(scope.strip()) == 0:    # pylint: disable=len-as-condition
                    # TODO: refactor above and test: if not (scope and scope.strip())
                    scope = 'GFN-BLK-{}'.format(idx)
                _msg = 'merge: stage {}, scope {}'.format(idx, scope)
                logger.info(_msg)
                input_map = dict(zip(gfn.input_names, prev_outputs))
                _, fetches = issn.importGraphFunction(
                    gfn, prefix=scope, input_map=input_map)
                prev_outputs = fetches

            # Add a non-scoped output name as the output node
            # So that users can still use the output name of the last function's output
            # to fetch the correct output tensors
            last_output_names = functions[-1][1].output_names
            last_outputs = []
            for tnsr, name in zip(prev_outputs, last_output_names):
                last_outputs.append(tf.identity(tnsr, name=name))

            gfn = issn.asGraphFunction(first_inputs, last_outputs)

        return gfn
