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
    HasTFGraph, HasTFHParams, HasTFCheckpointDir, HasExportDir, HasTagSet, HasSignatureDefKey)

__all__ = ['TFTransformer']

logger = logging.getLogger('sparkdl')

class TFTransformer(Transformer, HasTFCheckpointDir, HasTFGraph,
                    HasExportDir, HasTagSet, HasSignatureDefKey,
                    HasTFHParams, HasInputMapping, HasOutputMapping):
    """
    Applies the TensorFlow graph to the array column in DataFrame.

    Restrictions of the current API:

    We assume that
    - All graphs have a "minibatch" dimension (i.e. an unknown leading
      dimension) in the tensor shapes.
    - Input DataFrame has an array column where all elements have the same length
    """

    @keyword_only
    def __init__(self, tfCheckpointDir=None, tfGraph=None,
                 exportDir=None, tagSet=None, signatureDefKey=None,
                 inputMapping=None, outputMapping=None, tfHParms=None):
        """
        __init__(self, tfCheckpointDir=None, tfGraph=None,
                 exportDir=None, tagSet=None, signatureDefKey=None,
                 inputMapping=None, outputMapping=None, tfHParms=None)
        """
        super(TFTransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)


    @keyword_only
    def setParams(self, tfCheckpointDir=None, tfGraph=None,
                  exportDir=None, tagSet=None, signatureDefKey=None,
                  inputMapping=None, outputMapping=None, tfHParms=None):
        """
        setParams(self, tfCheckpointDir=None, tfGraph=None,
                  exportDir=None, tagSet=None, signatureDefKey=None,
                  inputMapping=None, outputMapping=None, tfHParms=None)
        """
        super(TFTransformer, self).__init__()
        kwargs = self._input_kwargs
        return self._set(**kwargs)


    def _convertInternal(self):
        assert self.isDefined(self.inputMapping) and self.isDefined(self.outputMapping), \
            "inputMapping and outputMapping must be defined"

        _maybe_graph = self.getTFGraph()
        _maybe_meta_graph_def = None
        with IsolatedSession(graph=_maybe_graph) as issn:
            if self.isDefined(self.exportDir):
                assert _maybe_graph is None
                assert not self.isDefined(self.tfCheckpointDir)
                tag_set = self.getTagSet().split(',')
                _maybe_meta_graph_def = tf.saved_model.loader.load(
                    issn.sess, tag_set, self.getExportDir())
            elif self.isDefined(self.tfCheckpointDir):
                assert _maybe_graph is None
                ckpt_dir = self.getTFCheckpointDir()
                ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
                print('using checkpoint path from {} as {}'.format(ckpt_dir, ckpt_path))
                saver = tf.train.import_meta_graph("{}.meta".format(ckpt_path), clear_devices=True)
                saver.restore(issn.sess, ckpt_path)
                _maybe_meta_graph_def = saver.export_meta_graph(clear_devices=True)
            else:
                assert _maybe_graph is not None

            sig_def = None
            if self.isDefined(self.signatureDefKey):
                sig_def_key = self.getSignatureDefKey()
                if sig_def_key is not None:
                    meta_graph_def = _maybe_meta_graph_def
                    assert meta_graph_def is not None
                    #print('sigdef:', meta_graph_def.signature_def)
                    sig_def = tf.contrib.saved_model.get_signature_def_by_key(
                        meta_graph_def, sig_def_key)
                    assert sig_def is not None

            feeds = []
            _input_mapping = {}
            for input_colname, tnsr_or_sig in self.getInputMapping():
                if sig_def:
                    tnsr = sig_def.inputs[tnsr_or_sig].name
                    _input_mapping[input_colname] = tfx.op_name(issn.graph, tnsr)
                else:
                    tnsr = tnsr_or_sig
                feeds.append(tfx.get_tensor(issn.graph, tnsr))

            if sig_def:
                self.setInputMapping(_input_mapping)

            fetches = []
            # By default the output columns will have the name of their
            # corresponding `tf.Graph` operation names.
            # We have to convert them to the user specified output names
            self.output_renaming = {}
            for tnsr_or_sig, output_colname in self.getOutputMapping():
                if sig_def:
                    tnsr = sig_def.outputs[tnsr_or_sig].name
                else:
                    tnsr = tnsr_or_sig
                fetches.append(tfx.get_tensor(issn.graph, tnsr))
                tf_expected_colname = tfx.op_name(issn.graph, tnsr)
                self.output_renaming[tf_expected_colname] = output_colname

            # Consolidate the input format into a serialized format
            self.gfn = issn.asGraphFunction(feeds, fetches, strip_and_freeze=True)


    def _transform(self, dataset):
        self._convertInternal()

        with IsolatedSession() as issn:
            analyzed_df = tfs.analyze(dataset)
            _, fetches = issn.importGraphFunction(self.gfn, prefix='')
            feed_dict = dict([(tnsr_name, col_name) for col_name, tnsr_name in self.getInputMapping()])
            out_df = tfs.map_blocks(fetches, analyzed_df, feed_dict=feed_dict)

            # We still have to rename output columns
            for old_colname, new_colname in self.output_renaming.items():
                if old_colname != new_colname:
                    out_df = out_df.withColumnRenamed(old_colname, new_colname)

        return out_df
