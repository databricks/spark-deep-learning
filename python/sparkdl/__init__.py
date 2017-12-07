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


# hack to import copy-pasted image schema (to be removed in Spark2.3)
# TODO remove in Spark2.3
import os
import pyspark.ml
dir_path = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(dir_path)
pyspark.ml.__path__.append(os.path.join(parentdir, "pyspark", "ml"))

from pyspark.ml.image import ImageSchema

from .graph.input import TFInputGraph
from .transformers.keras_image import KerasImageFileTransformer
from .transformers.named_image import DeepImagePredictor, DeepImageFeaturizer
from .transformers.tf_image import TFImageTransformer
from .transformers.tf_tensor import TFTransformer
from .transformers.utils import imageInputPlaceholder


__all__ = [
    'imageSchema', 'imageType', 'readImages',
    'TFImageTransformer', 'TFInputGraph', 'TFTransformer',
    'DeepImagePredictor', 'DeepImageFeaturizer', 'KerasImageFileTransformer', 'KerasTransformer',
    'imageInputPlaceholder']
