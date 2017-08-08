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
from __future__ import absolute_import, division, print_function

import logging
import numpy as np
import tensorflow as tf
import tensorframes as tfs

from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params
from pyspark.sql.functions import udf

from sparkdl.graph.builder import IsolatedSession
import sparkdl.graph.utils as tfx
from sparkdl.transformers.param import (
    keyword_only, HasInputMapping, HasOutputMapping, SparkDLTypeConverters,
    HasTFGraph, HasTFHParams)

__all__ = ['TFTransformer']

logger = logging.getLogger('sparkdl')

class TFTransformer(Transformer, HasTFGraph, HasTFHParams, HasInputMapping, HasOutputMapping):
    """
    Applies the TensorFlow graph to the array column in DataFrame.

    Restrictions of the current API:

    We assume that
    - All graphs have a "minibatch" dimension (i.e. an unknown leading
      dimension) in the tensor shapes.
    - Input DataFrame has an array column where all elements have the same length
    """

    @keyword_only
    def __init__(self, inputMapping=None, outputMapping=None, tfGraph=None, hparams=None):
        """
        __init__(self, inputMapping=None, outputMapping=None, tfGraph=None, hparams=None)
        """
        super(TFTransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputMapping=None, outputMapping=None, tfGraph=None, hparams=None):
        """
        setParams(self, inputMapping=None, outputMapping=None, tfGraph=None, hparams=None)
        """
        super(TFTransformer, self).__init__()
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        df = dataset
        output_renaming = {}

        with IsolatedSession(graph=self.getTFGraph()) as issn:
            feeds = []
            for input_colname, tnsr in self.getInputMapping():
                feeds.append(tfx.get_tensor(issn.graph, tnsr))
                tf_expected_colname = tfx.op_name(issn.graph, tnsr)
                df = df.withColumnRenamed(input_colname, tf_expected_colname)

            fetches = []
            for tnsr, output_colname in self.getOutputMapping():
                fetches.append(tfx.get_tensor(issn.graph, tnsr))
                tf_expected_colname = tfx.op_name(issn.graph, tnsr)
                output_renaming[tf_expected_colname] = output_colname

            gfn = issn.asGraphFunction(feeds, fetches, strip_and_freeze=True)

        analyzed_df = tfs.analyze(df)

        with IsolatedSession() as issn:
            _, fetches = issn.importGraphFunction(gfn, prefix='')
            out_df = tfs.map_blocks(fetches, analyzed_df)

            for old_colname, new_colname in output_renaming.items():
                out_df = out_df.withColumnRenamed(old_colname, new_colname)

        return out_df
