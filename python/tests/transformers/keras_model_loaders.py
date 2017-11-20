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

"""
TODO: Load keras model that operates on 1D input...

NOTE: all we need is to store a model file, can probably use some canonical dataset of
vector features...
"""

def loadAndPreprocessKerasInceptionV3(raw_uri):

    # this is the canonical way to load and prep images in keras
    uri = raw_uri[5:] if raw_uri.startswith("file:/") else raw_uri
    image = img_to_array(load_img(uri, target_size=InceptionV3Constants.INPUT_SHAPE))
    image = np.expand_dims(image, axis=0)
    return preprocess_input(image)

def prepInceptionV3KerasModelFile(fileName):
    model_dir_tmp = tempfile.mkdtemp("sparkdl_keras_tests", dir="/tmp")
    path = model_dir_tmp + "/" + fileName

    height, width = InceptionV3Constants.INPUT_SHAPE
    input_shape = (height, width, 3)
    model = InceptionV3(weights="imagenet", include_top=True, input_shape=input_shape)
    model.save(path)
    return path