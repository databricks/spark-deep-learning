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

import numpy as np
import tensorflow as tf
import tensorframes as tfs

from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params
from pyspark.sql.functions import udf

from sparkdl.image.imageIO import imageSchema, sparkModeLookup, SparkMode
from sparkdl.param import (
    keyword_only, HasInputCol, HasOutputCol, SparkDLTypeConverters, HasOutputMode)
import sparkdl.transformers.utils as utils
import sparkdl.utils.jvmapi as JVMAPI
import sparkdl.graph.utils as tfx

class TFImageTransformer(Transformer, HasInputCol, HasOutputCol, HasOutputMode):
    """
    Applies the Tensorflow graph to the image column in DataFrame.

    Restrictions of the current API:

    * Does not use minibatches, which is a major low-hanging fruit for performance.
    * Only one output node can be specified.
    * The output is expected to be an image or a 1-d vector.
    * All images in the dataframe are expected be of the same numerical data type
      (i.e. the dtype of the values in the numpy array representation is the same.)

    We assume all graphs have a "minibatch" dimension (i.e. an unknown leading
    dimension) in the tensor shapes.

    .. note:: The input tensorflow graph should have appropriate weights constantified,
              since a new session is created inside this transformer.
    """

    USER_GRAPH_NAMESPACE = 'given'
    NEW_OUTPUT_PREFIX = 'sdl_flattened'

    graph = Param(Params._dummy(), "graph", "A TensorFlow computation graph",
                  typeConverter=SparkDLTypeConverters.toTFGraph)
    inputTensor = Param(Params._dummy(), "inputTensor",
                        "A TensorFlow tensor object or name representing the input image",
                        typeConverter=SparkDLTypeConverters.toStringOrTFTensor)
    outputTensor = Param(Params._dummy(), "outputTensor",
                         "A TensorFlow tensor object or name representing the output",
                         typeConverter=SparkDLTypeConverters.toStringOrTFTensor)

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, graph=None,
                 inputTensor=utils.IMAGE_INPUT_PLACEHOLDER_NAME, outputTensor=None,
                 outputMode="vector"):
        """
        __init__(self, inputCol=None, outputCol=None, graph=None,
                 inputTensor=utils.IMAGE_INPUT_PLACEHOLDER_NAME, outputTensor=None,
                 outputMode="vector")
        """
        super(TFImageTransformer, self).__init__()
        self._setDefault(inputTensor=utils.IMAGE_INPUT_PLACEHOLDER_NAME)
        self._setDefault(outputMode="vector")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, graph=None,
                  inputTensor=utils.IMAGE_INPUT_PLACEHOLDER_NAME, outputTensor=None,
                  outputMode="vector"):
        """
        setParams(self, inputCol=None, outputCol=None, graph=None,
                  inputTensor=utils.IMAGE_INPUT_PLACEHOLDER_NAME, outputTensor=None,
                  outputMode="vector")
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setGraph(self, value):
        return self._set(graph=value)

    def setInputTensor(self, value):
        return self._set(inputTensor=value)

    def setOutputTensor(self, value):
        return self._set(outputTensor=value)

    def getGraph(self):
        return self.getOrDefault(self.graph)

    def getInputTensor(self):
        tensor_or_name = self.getOrDefault(self.inputTensor)
        if isinstance(tensor_or_name, tf.Tensor):
            return tensor_or_name
        else:
            return self.getGraph().get_tensor_by_name(tensor_or_name)

    def getOutputTensor(self):
        tensor_or_name = self.getOrDefault(self.outputTensor)
        if isinstance(tensor_or_name, tf.Tensor):
            return tensor_or_name
        else:
            return self.getGraph().get_tensor_by_name(tensor_or_name)

    def _transform(self, dataset):
        graph = self.getGraph()
        composed_graph = self._addReshapeLayers(graph, self._getImageDtype(dataset))
        final_graph = self._stripGraph(composed_graph)

        with final_graph.as_default():
            image = dataset[self.getInputCol()]
            image_df_exploded = (dataset
              .withColumn("__sdl_image_height", image.height)
              .withColumn("__sdl_image_width", image.width)
              .withColumn("__sdl_image_nchannels", image.nChannels)
              .withColumn("__sdl_image_data", image.data)
            )

            final_output_name = self._getFinalOutputTensorName()
            output_tensor = final_graph.get_tensor_by_name(final_output_name)
            final_df = (
                tfs.map_rows([output_tensor], image_df_exploded,
                             feed_dict={
                                 "height": "__sdl_image_height",
                                 "width": "__sdl_image_width",
                                 "num_channels": "__sdl_image_nchannels",
                                 "image_buffer": "__sdl_image_data"})
                .drop("__sdl_image_height", "__sdl_image_width", "__sdl_image_nchannels",
                      "__sdl_image_data")
            )

            tfs_output_name = tfx.op_name(final_graph, output_tensor)
            original_output_name = self._getOriginalOutputTensorName()
            output_shape = final_graph.get_tensor_by_name(original_output_name).shape
            output_mode = self.getOrDefault(self.outputMode)
            # TODO: support non-1d tensors (return np.array).
            if output_mode == "image":
                return self._convertOutputToImage(final_df, tfs_output_name, output_shape)
            else:
                assert output_mode == "vector", "Unknown output mode: %s" % output_mode
                return self._convertOutputToVector(final_df, tfs_output_name)

    def _getImageDtype(self, dataset):
        # This may not be the best way to get the type of image, but it is one way.
        # Assumes that the dtype for all images is the same in the given dataframe.
        pdf = dataset.select(self.getInputCol()).take(1)
        img = pdf[0][self.getInputCol()]
        img_type = sparkModeLookup[img.mode]
        return img_type.dtype

    def _addReshapeLayers(self, tf_graph, dtype="uint8"):
        input_tensor_name = self.getInputTensor().name

        gdef = tf_graph.as_graph_def(add_shapes=True)
        g = tf.Graph()
        with g.as_default():
            # Flat image data -> image dimensions
            height = tf.placeholder(tf.int32, [], name="height")
            width = tf.placeholder(tf.int32, [], name="width")
            num_channels = tf.placeholder(tf.int32, [], name="num_channels")
            image_buffer = tf.placeholder(tf.string, [], name="image_buffer")
            # Note: the shape argument is required for tensorframes as it uses a
            # slightly older version of tensorflow.
            shape = tf.reshape(tf.stack([height, width, num_channels], axis=0), shape=(3,),
                               name='shape')
            if dtype == "uint8":
                image_uint8 = tf.decode_raw(image_buffer, tf.uint8, name="decode_raw")
                image_float = tf.to_float(image_uint8)
            else:
                assert dtype == SparkMode.FLOAT32, "Unsupported dtype for image: %s" % dtype
                image_float = tf.decode_raw(image_buffer, tf.float32, name="decode_raw")
            image_reshaped = tf.reshape(image_float, shape, name="reshaped")
            image_reshaped_expanded = tf.expand_dims(image_reshaped, 0, name="expanded")

            # Add on the original graph
            tf.import_graph_def(gdef, input_map={input_tensor_name: image_reshaped_expanded},
                                return_elements=[self.getOutputTensor().name],
                                name=self.USER_GRAPH_NAMESPACE)

            # Flatten the output for tensorframes
            output_node = g.get_tensor_by_name(self._getOriginalOutputTensorName())
            _ = tf.reshape(output_node[0],  # batch-size = 1,
                           shape=[-1], name=self._getFinalOutputOpName())
        return g

    # Sometimes the tf graph contains a bunch of stuff that doesn't lead to the
    # output. TensorFrames does not like that, so we strip out the parts that
    # are not necessary for the computation at hand.
    def _stripGraph(self, tf_graph):
        gdef = tfx.strip_and_freeze_until([self._getFinalOutputOpName()], tf_graph)
        g = tf.Graph()
        with g.as_default():
            tf.import_graph_def(gdef, name='')
        return g

    def _getOriginalOutputTensorName(self):
        return self.USER_GRAPH_NAMESPACE + '/' + self.getOutputTensor().name

    def _getFinalOutputTensorName(self):
        return self.NEW_OUTPUT_PREFIX + '_' + self.getOutputTensor().name

    def _getFinalOutputOpName(self):
        return tfx.as_op_name(self._getFinalOutputTensorName())

    def _convertOutputToImage(self, df, tfs_output_col, output_shape):
        assert len(output_shape) == 4, str(output_shape) + " does not have 4 dimensions"
        height = int(output_shape[1])
        width = int(output_shape[2])
        def to_image(orig_image, numeric_data):
            # Assume the returned image has float pixels but same #channels as input
            mode = orig_image.mode if orig_image.mode == "float32" else "RGB-float32"
            return [mode, height, width, orig_image.nChannels,
                    bytearray(np.array(numeric_data).astype(np.float32).tobytes())]
        to_image_udf = udf(to_image, imageSchema)
        return (
            df.withColumn(self.getOutputCol(),
                          to_image_udf(df[self.getInputCol()], df[tfs_output_col]))
              .drop(tfs_output_col)
        )

    def _convertOutputToVector(self, df, tfs_output_col):
        """
        Converts the output python list to MLlib Vector.
        """
        return (
            df.withColumn(self.getOutputCol(), JVMAPI.listToMLlibVectorUDF(df[tfs_output_col]))
              .drop(tfs_output_col)
        )
