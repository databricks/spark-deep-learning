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
from pathlib import Path
import shutil
import six
from tempfile import mkdtemp

import keras.backend as K
from keras.models import Model as KerasModel, load_model
import tensorflow as tf

from .graph_builder import GraphBuilderSession
from .image.imageIO import SparkMode

logger = logging.getLogger('sparkdl')

class GraphFunctionFactory(object):
    """ 
    Build various pieces of the function
    """

    @classmethod
    def build_spimage_converter(cls, img_dtype):
        """ 
        Convert a imageIO byte encoded image into a image tensor suitable as input to ConvNets
        The name of the input must be a subset of those specified in `image.imageIO.imgSchema`.

        :param img_dtype: the type of data the underlying image bytes represent
        """
        with GraphBuilderSession() as builder:
            # Flat image data -> image dimensions
            # This has to conform to `imageIO.imgSchema`
            height = tf.placeholder(tf.int32, [], name="height")
            width = tf.placeholder(tf.int32, [], name="width")
            num_channels = tf.placeholder(tf.int32, [], name="nChannels")
            image_buffer = tf.placeholder(tf.string, [], name="data")

            # The image is packed into bytes with height as leading dimension
            # This is the default behavior of Python Image Library
            shape = tf.reshape(tf.stack([height, width, num_channels], axis=0), 
                               shape=(3,), name='shape')
            if SparkMode.RGB == img_dtype:
                image_uint8 = tf.decode_raw(image_buffer, tf.uint8, name="decode_raw")
                image_float = tf.to_float(image_uint8)
            else:
                assert img_dtype == SparkMode.RGB_FLOAT32, \
                    "Unsupported dtype for image: {}".format(img_dtype)
                image_float = tf.decode_raw(image_buffer, tf.float32, name="decode_raw")

            image_reshaped = tf.reshape(image_float, shape, name="reshaped")
            image_input = tf.expand_dims(image_reshaped, 0, name="image_input")
            gfn = builder.asGraphFunction([height, width, image_buffer, num_channels], [image_input])

        return gfn

    @classmethod
    def build_identity(cls):
        with GraphBuilderSession() as builder:
            pred_input = tf.placeholder(tf.float32, [None, None])
            final_output = tf.identity(pred_input, name='output')
            gfn = builder.asGraphFunction([pred_input], [final_output])

        return gfn

    @classmethod
    def build_flattener(cls):
        with GraphBuilderSession() as builder:
            mat_input = tf.placeholder(tf.float32, [None, None])
            mat_output = tf.identity(tf.reshape(mat_input, shape=[-1]), name='output')
            gfn = builder.asGraphFunction([mat_input], [mat_output])

        return gfn
    
    @classmethod
    def import_bare_keras(cls, model_or_file_path):
        """ Build an UDF from a Keras model
        """
        if isinstance(model_or_file_path, KerasModel):
            model = model_or_file_path
            model_path = Path(mkdtemp(prefix='kera-')) / "model.h5"
            # Save to tempdir and restore in a new session
            model.save(str(model_path), overwrite=True)
            is_temp_model = True
        else:
            model_path = model_or_file_path
            is_temp_model = False

        # Keras load function requires path string
        if not isinstance(model_path, six.string_types):
            model_path = str(model_path)

        with GraphBuilderSession(wrap_keras=True) as builder:
            K.set_session(builder.sess)
            K.set_learning_phase(0) # Testing phase
            model = load_model(model_path)
            if is_temp_model:
                shutil.rmtree(str(Path(model_path).parent), ignore_errors=True)

            gfn = builder.asGraphFunction(model.inputs, model.outputs)

        return gfn

    @classmethod
    def pipeline(cls, functions):
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
        with GraphBuilderSession() as builder:
            _, first_gfn = functions[0]
            feeds, _ = builder.import_graph_function(first_gfn, name='')
            first_input_info = []
            for tnsr in feeds:
                name = builder.op_name(tnsr)
                first_input_info.append((tnsr.dtype, tnsr.shape, name))

        # Build a linear chain of all the provide functions
        with GraphBuilderSession() as builder: 
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
                _, fetches = builder.import_graph_function(
                    gfn, name=scope, input_map=input_map)
                prev_outputs = fetches

            # Add a non-scoped output name as the output node
            last_output_names = functions[-1][1].output_names
            last_outputs = []
            for tnsr, name in zip(prev_outputs, last_output_names):
                last_outputs.append(tf.identity(tnsr, name=name))

            gfn = builder.asGraphFunction(first_inputs, last_outputs)

        return gfn
