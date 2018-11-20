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

import os
from .named_image_test import NamedImageTransformerBaseTestCase


class NamedImageTransformerInceptionV3Test(NamedImageTransformerBaseTestCase):

    # TODO(ML-5165) Enable these tests in a separate target
    __test__ = os.getenv('RUN_ONLY_LIGHT_TESTS', False) != "True"
    name = "InceptionV3"
    featurizerCompareDigitsExact = 4
