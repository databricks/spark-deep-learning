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

import joblib as jl
import logging
from pathlib import Path
import shutil
import six
from tempfile import mkdtemp
import webbrowser

import keras.backend as K
import tensorflow as tf
import tensorframes.core as tfrm

from .utils import jvmapi as JVMAPI

logger = logging.getLogger('sparkdl')

class GraphBuilder(object):
    """
    Building TensorFlow graph and export as either UDF or GraphFunction

    When working with various pieces of TensorFlow, one is faced with
    figuring out providing one of the four variants
    (`tensor` OR `operation`, `name` OR `graph element`).

    The various combination makes it hard to figuring out the best way.
    We provide some methods to map whatever we have as input to
    one of the four target variants.

    This is a thin layer on top of `tf.Session`.
    """

    def __init__(self, graph):
        assert isinstance(graph, tf.Graph)
        self.sess = tf.Session(graph=graph)
        self.graph = graph

    def get_shape(self, tnsr_or_name):
        """ Return the shape of the tensor as a list """
        _shape = self.get_tensor(tnsr_or_name).get_shape().as_list()
        return [-1 if x is None else x for x in _shape]

    def get_op(self, tfobj_or_name):
        """ Get the graph element `Operation` from a given name or graph element """
        if isinstance(tfobj_or_name, tf.Operation):
            return tfobj_or_name
        name = tfobj_or_name
        if isinstance(tfobj_or_name, tf.Tensor):
            name = tfobj_or_name.name
        if not isinstance(name, six.string_types):
            raise TypeError('invalid op request for {} of {}'.format(name, type(name)))
        name = name.split(":")[0]
        op = self.graph.get_operation_by_name(name)
        assert op is not None, \
            'cannot locate op {} in current graph'.format(name)
        return op

    def get_tensor(self, tfobj_or_name):
        if isinstance(tfobj_or_name, tf.Tensor):
            return tfobj_or_name
        name = tfobj_or_name
        if isinstance(tfobj_or_name, tf.Operation):
            name = tfobj_or_name.name
        if not isinstance(name, six.string_types):
            raise TypeError('invalid tensor request for {} of {}'.format(name, type(name)))
        name_parts = name.split(":")
        if len(name_parts) < 2:
            name += ":0"
        tnsr = self.graph.get_tensor_by_name(name)
        assert tnsr is not None, \
            'cannot locate tensor {} in current graph'.format(name)
        return tnsr

    def op_name(self, tfobj_or_name):
        return self.get_op(tfobj_or_name).name

    def tensor_name(self, tfobj_or_name):
        return self.get_tensor(tfobj_or_name).name

    def valid_output(self, tfobj_or_name):
        return self.op_name(tfobj_or_name)

    def valid_input(self, tfobj_or_name):
        name = self.op_name(tfobj_or_name)
        op = self.graph.get_operation_by_name(name)
        assert 'Placeholder' == op.type, \
            ('input must be Placeholder, but get', op.type)
        return name

    def show_tf_graph(self, max_const_size=32, show_in_browser=True):
        """
        Visualize TensorFlow graph as a static TensorBoard page.
        Notice that in order to view it directly, the user must have
        a working Chrome browser.

        The page directly embed GraphDef prototxt so that the page (and an active Internet connection)
        is only needed to view the content. There is NO need to fire up a web server in the backend.

        :param max_const_size: int, if a constant is way too long, clip it in the plot
        :param show_in_browser: bool, indicate if we want to launch a browser to show the generated HTML page.
        :return: str, the generated HTML code (which might be quite large)
        """
        _tfb_url_prefix = "https://tensorboard.appspot.com"
        _tfb_url = "{}/tf-graph-basic.build.html".format(_tfb_url_prefix)

        def strip_consts(gdef, max_const_size=32):
            """Strip large constant values from graph_def."""
            strip_def = tf.GraphDef()
            for n0 in gdef.node:
                n = strip_def.node.add()  # pylint: disable=E1101
                n.MergeFrom(n0)
                if n.op == 'Const':
                    tensor = n.attr['value'].tensor
                    nbytes = len(tensor.tensor_content)
                    if nbytes > max_const_size:
                        tensor.tensor_content = str.encode("<stripped {} bytes>".format(nbytes))
            return strip_def

        strip_def = strip_consts(self.graph.as_graph_def(), max_const_size)
        html_code = """
            <script>
              function load() {{
                document.getElementById("{id}").pbtxt = {data};
              }}
            </script>
            <link rel="import" href="{tfb}" onload=load()>
            <div style="height:600px">
              <tf-graph-basic id="{id}"></tf-graph-basic>
            </div>
        """.format(data=repr(str(strip_def)),
                   id='gfn-sess-graph',
                   tfb=_tfb_url)

        # Construct the graph def board and open it
        fp_out = Path.cwd() / "graph_def.html"
        with open(str(fp_out), 'wb') as fout:
            try:
                fout.write(html_code)
            except TypeError:
                fout.write(html_code.encode('utf8'))

        print("file @", str(fp_out))
        if show_in_browser:
            webbrowser.get("chrome").open("file://{}".format(str(fp_out)))
        return html_code


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
        jvm_builder = JVMAPI.for_class("com.databricks.sparkdl.python.GraphModelFactory")
        tfrm._add_graph(self.graph, jvm_builder)

        fetch_names = [self.tensor_name(tnsr) for tnsr in fetches]
        fetch_shapes = [self.get_shape(tnsr) for tnsr in fetches]
        placeholder_names = []
        placeholder_shapes = []

        for node in self.graph.as_graph_def(add_shapes=True).node:
            if len(node.input) == 0 and str(node.op) == 'Placeholder':
                tnsr_name = self.tensor_name(node.name)
                tnsr = self.graph.get_tensor_by_name(tnsr_name)
                try:
                    tnsr_shape = self.get_shape(tnsr)
                    placeholder_names.append(tnsr_name)
                    placeholder_shapes.append(tnsr_shape)
                except ValueError:
                    pass

        jvm_builder.shape(fetch_names + placeholder_names, fetch_shapes + placeholder_shapes)
        jvm_builder.fetches(fetch_names)
        placeholder_op_names = [self.op_name(tnsr_name) for tnsr_name in placeholder_names]
        tfrm._add_inputs(jvm_builder, feed_dict, placeholder_op_names)

        if register:
            return jvm_builder.registerUDF(udf_name, blocked)
        else:
            return jvm_builder.makeUDF(udf_name, blocked)

    def asGraphFunction(self, inputs, outputs, strip_and_freeze=True):
        """
        Build a GraphFunction object

        :param inputs: list, graph elements representing the inputs
        :param outputs: list, graph elements representing the outputs
        :param strip_and_freeze: bool, should we remove unused part of the graph and freee its values
        """
        if strip_and_freeze:
            gdef = self.strip_and_freeze_until(outputs)
        else:
            gdef = self.graph.as_graph_def(add_shapes=True)
        return GraphFunction(graph_def=gdef,
                             input_names=[self.valid_input(i) for i in inputs],
                             output_names=[self.valid_output(o) for o in outputs])


    def strip_and_freeze_until(self, fetches):
        """
        Converting all variables into constants

        :param fetches: list, graph elements representing the outputs of the graph
        :return: GraphDef, the GraphDef object with cleanup procedure applied
        """
        gdef_frozen = tf.graph_util.convert_variables_to_constants(
            self.sess,
            self.graph.as_graph_def(add_shapes=True),
            [self.op_name(tnsr) for tnsr in fetches])
        return gdef_frozen


    def import_graph_function(self, gfn, input_map=None, name="GFN-IMPORT", **gdef_kargs):
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
        feeds = list(map(self.get_tensor, input_names))
        fetches = list(map(self.get_tensor, output_names))
        return (feeds, fetches)


class GraphBuilderSession(object):
    """
    GraphBuilderSession, a thin wrapper for TensorFlow's `Session`.
    It provides a GraphBuilder object which provides
    - various helper functions for locating and naming graph elements
    - importing existing GraphFunction object as a subgraph
    - exporting current graph as an GraphFunction object

    :param g: Graph, use the provided TensorFlow graph as default graph
    :param wrap_keras: bool, whether to also let Keras TensorFlow backend use this session
    """
    def __init__(self, g=None, wrap_keras=False):
        self.builder = GraphBuilder(g or tf.Graph())
        if wrap_keras:
            self.keras_prev_sess = K.get_session()
        else:
            self.keras_prev_sess = None

    def __enter__(self):
        self.builder.sess.as_default()
        self.builder.sess.__enter__()
        if self.keras_prev_sess is not None:
            K.set_session(self.builder.sess)
        return self.builder

    def __exit__(self, *args):
        if self.keras_prev_sess is not None:
            K.set_session(self.keras_prev_sess)
        self.builder.sess.__exit__(*args)



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

    @classmethod
    def from_file(cls, fpath):
        """
        Load an existing GraphFunction from file.
        This implementation uses `joblib` to provide good I/O performance

        :param fpath: str or path, path to the serialized GraphFunction
        """
        _st = jl.load(fpath)
        assert set(['inputs', 'graph_def_bytes', 'outputs']) <= set(_st.keys())
        gdef = tf.GraphDef.FromString(_st["graph_def_bytes"])  # pylint: disable=E1101
        return cls(graph_def=gdef,
                   input_names=_st["inputs"],
                   output_names=_st["outputs"])

    def dump(self, fpath):
        """
        Store the GraphFunction to a file

        :param fpath: str or path, path to the serialized GraphFunction
        """
        _st = {"graph_def_bytes": self.graph_def.SerializeToString(),
               "inputs": self.input_names,
               "outputs": self.output_names}
        assert isinstance(fpath, six.string_types)
        if not fpath.endswith("jl"):
            fpath += ".jl"
        jl.dump(_st, fpath)
