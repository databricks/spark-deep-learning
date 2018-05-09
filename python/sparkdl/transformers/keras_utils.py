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

import keras.backend as K
import tensorflow as tf


# pylint: disable=too-few-public-methods
class KSessionWrap:
    """
    Runs operations in Keras in an isolated manner: the current graph and the current session
    are not modified by anything done in this block:

    with KSessionWrap() as (current_session, current_graph):
    ... do some things that call Keras
    """

    def __init__(self, graph=None):
        self.requested_graph = graph

    def __enter__(self):
        # pylint: disable=attribute-defined-outside-init
        self.old_session = K.get_session()
        self.g = self.requested_graph or tf.Graph()     # pylint: disable=invalid-name
        self.current_session = tf.Session(graph=self.g)
        # pylint: enable=attribute-defined-outside-init
        K.set_session(self.current_session)
        return self.current_session, self.g

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the previous session
        K.set_session(self.old_session)
