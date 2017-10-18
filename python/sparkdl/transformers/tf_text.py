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
from pyspark.ml import Transformer
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import udf
from pyspark.sql import functions as f
from pyspark.sql.types import *
from pyspark.sql.functions import lit
from sparkdl.param.shared_params import HasEmbeddingSize, HasSequenceLength
from sparkdl.param import (
    keyword_only, HasInputCol, HasOutputCol)
import re

import sparkdl.utils.jvmapi as JVMAPI


class TFTextTransformer(Transformer, HasInputCol, HasOutputCol, HasEmbeddingSize, HasSequenceLength):
    """
    Convert sentence/document to a 2-D Array eg. [[word embedding],[....]]  in DataFrame which can be processed
    directly by tensorflow or keras who's backend is tensorflow.

    Processing Steps:

    * Using Word2Vec compute Map(word -> vector) from input column, then broadcast the map.
    * Process input column (which is text),split it with white space, replace word with vector, padding the result to
      the same size.
    * Create a new dataframe with columns like new 2-D array , vocab_size, embedding_size
    * return then new dataframe
    """
    VOCAB_SIZE = 'vocab_size'
    EMBEDDING_SIZE = 'embedding_size'

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, embeddingSize=100, sequenceLength=64):
        super(TFTextTransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, embeddingSize=100, sequenceLength=64):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        sc = JVMAPI._curr_sc()

        word2vec = Word2Vec(vectorSize=self.getEmbeddingSize(), minCount=1, inputCol=self.getInputCol(),
                            outputCol="word_embedding")

        vectorsDf = word2vec.fit(
            dataset.select(f.split(self.getInputCol(), "\\s+").alias(self.getInputCol()))).getVectors()

        """
          It's strange here that after calling getVectors the df._sc._jsc will lose and this is
          only happens when you run it with ./python/run-tests.sh script.
          We add this code to make it pass the test. However it seems this will hit
          "org.apache.spark.SparkException: EOF reached before Python server acknowledged" error.
        """
        if vectorsDf._sc._jsc is None:
            vectorsDf._sc._jsc = sc._jsc

        word_embedding = dict(vectorsDf.rdd.map(
            lambda p: (p.word, p.vector.values.tolist())).collect())

        word_embedding["unk"] = np.zeros(self.getEmbeddingSize()).tolist()
        local_word_embedding = sc.broadcast(word_embedding)

        def convert_word_to_index(s):
            def _pad_sequences(sequences, maxlen=None):
                new_sequences = []

                if len(sequences) <= maxlen:
                    for i in range(maxlen - len(sequences)):
                        new_sequences.append(np.zeros(self.getEmbeddingSize()).tolist())
                    return sequences + new_sequences
                else:
                    return sequences[0:maxlen]

            new_q = [local_word_embedding.value[word] for word in re.split(r"\s+", s) if
                     word in local_word_embedding.value.keys()]
            result = _pad_sequences(new_q, maxlen=self.getSequenceLength())
            return result

        cwti_udf = udf(convert_word_to_index, ArrayType(ArrayType(FloatType())))
        doc_martic = (dataset.withColumn(self.getOutputCol(), cwti_udf(self.getInputCol()).alias(self.getOutputCol()))
                      .withColumn(self.VOCAB_SIZE, lit(len(word_embedding)))
                      .withColumn(self.EMBEDDING_SIZE, lit(self.getEmbeddingSize()))
                      )

        return doc_martic
