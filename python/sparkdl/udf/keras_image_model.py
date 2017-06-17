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

from ..graph.builder import GraphFunction, IsolatedSession
from ..graph.pieces import buildSpImageConverter, buildFlattener
from ..image.imageIO import imageSchema
from ..utils import jvmapi as JVMAPI

# We need this to be a simple `namedtuple` for serialization purposes
# This way it will not also unpickle the extra structures

logger = logging.getLogger('sparkdl')

ENTRYPOINT_CLASSNAME = "com.databricks.sparkdl.python.GraphModelFactory"

def _serialize_and_reload_with(preprocessor):
    """
    Load a preprocessor function (image_file_path => image_tensor)
    """
    def udf_impl(spimg):
        import numpy as np
        from PIL import Image
        from tempfile import NamedTemporaryFile
        from sparkdl.image.imageIO import imageArrayToStruct, imageType
        
        pil_mode = imageType(spimg).pilMode        
        img_shape = (spimg.width, spimg.height)
        img = Image.frombytes(pil_mode, img_shape, bytes(spimg.data))
        # Warning: must use lossless format to guarantee consistency
        temp_fp = NamedTemporaryFile(suffix='.png')
        img.save(temp_fp, 'PNG')
        img_arr_reloaded = preprocessor(temp_fp.name)
        assert isinstance(img_arr_reloaded, np.ndarray), \
            "expect preprocessor to return a numpy array"        
        img_arr_reloaded = img_arr_reloaded.astype(np.uint8)
        return imageArrayToStruct(img_arr_reloaded)

    return udf_impl

def registerKerasImageUDF(udf_name, keras_model_or_file_path, preprocessor=None):
    """
    Create an UserDefinedFunction from a Keras model

    registerKerasImageUDF(`udf_name`, "path/to/my/keras/model.h5", `preprocessor`)

    :param udf_name: str, name of the UserDefinedFunction
    :param keras_model_file_path: str, path to the HDF5 keras model file
    :param preprocessor: function, optional, 
    :return: list, image as a DataFrame Row compatible list    .
    """
    ordered_udf_names = []
    keras_udf_name = udf_name
    if preprocessor is not None:
        # Spill the image structure to file and reload it 
        # with the user provided preprocessing funcition
        preproc_udf_name = '{}__preprocess'.format(udf_name)
        ordered_udf_names.append(preproc_udf_name)
        JVMAPI.registerUDF(
            preproc_udf_name,
            _serialize_and_reload_with(preprocessor),
            imageSchema)
        keras_udf_name = '{}__model_predict'.format(udf_name)

    stages = [('spimg', buildSpImageConverter("RGB")),
              ('model', GraphFunction.fromKeras(keras_model_or_file_path)),
              ('final', buildFlattener())]
    gfn = GraphFunction.fromList(stages)

    with IsolatedSession() as issn:
        _, fetches = issn.importGraphFunction(gfn, name='')
        issn.asUDF(keras_udf_name, fetches)
        ordered_udf_names.append(keras_udf_name)

    if len(ordered_udf_names) > 1:
        msg = "registering pipelined UDF {udf} with stages {udfs}"
        msg = msg.format(udf=udf_name, udfs=ordered_udf_names)
        logger.info(msg)
        JVMAPI.registerPipeline(udf_name, ordered_udf_names)

    return gfn
