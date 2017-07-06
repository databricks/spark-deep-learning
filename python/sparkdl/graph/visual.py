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
from tempfile import NamedTemporaryFile
import webbrowser

import tensorflow as tf

import sparkdl.graph.utils as tfx

logger = logging.getLogger('sparkdl')

TFB_URL_PREFIX = "https://tensorboard.appspot.com"
TFB_URL = "{}/tf-graph-basic.build.html".format(TFB_URL_PREFIX)


def write_visualization_html(graph, html_file_path=None, max_const_size=32, show_in_browser=True):
    """
    Visualize TensorFlow graph as a static TensorBoard page.
    Notice that in order to view it directly, the user must have
    a working Chrome browser.

    The page directly embed GraphDef prototxt so that the page (and an active Internet connection)
    is only needed to view the content. There is NO need to fire up a web server in the backend.

    :param graph: tf.Graph, a TensorFlow Graph object
    :param html_file_path: str, path to the HTML output file
    :param max_const_size: int, if a constant is way too long, clip it in the plot
    :param show_in_browser: bool, indicate if we want to launch a browser to show the generated HTML page.
    """
    if isinstance(graph, tf.GraphDef):
        gdef = graph
    else:
        graph = tfx.validated_graph(graph)
        gdef = graph.as_graph_def()

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

    strip_gdef = strip_consts(gdef, max_const_size)
    html_header = """
        <style>
          body, html { height: 100%; }
          div#graphdef { height: 100%; }
        </style>
    """
    html_code = html_header + """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="{tfb}" onload=load()>
        <div id="graphdef">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_gdef)),
               id='gfn-sess-graph',
               tfb=TFB_URL)

    if not html_file_path:
        html_file_path = NamedTemporaryFile(prefix="tf_graph_def", suffix=".html").name

    # Construct the graph def board and open it
    with open(str(html_file_path), 'wb') as fout:
        try:
            fout.write(html_code)
        except TypeError:
            fout.write(html_code.encode('utf8'))

    logger.info("html output file @ " + str(html_file_path))
    if show_in_browser:
        webbrowser.get("chrome").open("file://{}".format(str(html_file_path)))
