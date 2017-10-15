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
import cPickle as pickle
import shutil
import threading

from tensorflowonspark import TFNode

from sparkdl.estimators.tf_text_file_estimator import TFTextFileEstimator, KafkaMockServer
from sparkdl.transformers.tf_text import TFTextTransformer
from sparkdl.tf_fun import map_fun
from ..tests import TFoSBaseSparkTest
from ..tests import SparkDLTestCase


class TFTextTransformerTest(SparkDLTestCase):
    def test_convertText(self):
        input_col = "text"
        output_col = "sentence_matrix"

        documentDF = self.session.createDataFrame([
            ("Hi I heard about Spark", 1),
            ("I wish Java could use case classes", 0),
            ("Logistic regression models are neat", 2)
        ], ["text", "preds"])

        # transform text column to sentence_matrix column which contains 2-D array.
        transformer = TFTextTransformer(
            inputCol=input_col, outputCol=output_col, embeddingSize=100, sequenceLength=64)

        df = transformer.transform(documentDF)
        data = df.collect()
        self.assertEquals(len(data), 3)
        for row in data:
            self.assertEqual(len(row[output_col]), 64)
            self.assertEqual(len(row[output_col][0]), 100)


class TFTextFileEstimatorTest(SparkDLTestCase):
    def test_trainText(self):
        import os
        if os.path.exists(KafkaMockServer()._kafka_mock_server_tmp_file_):
            shutil.rmtree(KafkaMockServer()._kafka_mock_server_tmp_file_)

        input_col = "text"
        output_col = "sentence_matrix"

        documentDF = self.session.createDataFrame([
            ("Hi I heard about Spark", 1),
            ("I wish Java could use case classes", 0),
            ("Logistic regression models are neat", 2)
        ], ["text", "preds"])

        # transform text column to sentence_matrix column which contains 2-D array.
        transformer = TFTextTransformer(
            inputCol=input_col, outputCol=output_col, embeddingSize=100, sequenceLength=64)

        df = transformer.transform(documentDF)

        # create a estimator to training where map_fun contains tensorflow's code
        estimator = TFTextFileEstimator(inputCol="sentence_matrix", outputCol="sentence_matrix", labelCol="preds",
                                        kafkaParam={"bootstrap_servers": ["127.0.0.1"], "topic": "test",
                                                    "group_id": "sdl_1", "test_mode": True},
                                        fitParam=[{"epochs": 5, "batch_size": 64}, {"epochs": 5, "batch_size": 1}],
                                        runningMode="Normal",
                                        mapFnParam=map_fun)
        estimator.fit(df).collect()


class TFTextFileEstimatorOnTFoSTest(TFoSBaseSparkTest):
    def trainText(self):
        """
         To make this test work,Please:
          1. Start a Spark standalone cluster and export MASTER to your env,
          2. Make sure spark-deep-learning assembly  in spark classpath.
          3. Change method 'trainText' to 'test_trainText'
        """
        input_col = "text"
        output_col = "sentence_matrix"

        documentDF = self.session.createDataFrame([
            ("Hi I heard about Spark", 1),
            ("I wish Java could use case classes", 0),
            ("Logistic regression models are neat", 2)
        ], ["text", "preds"])

        # transform text column to sentence_matrix column which contains 2-D array.
        transformer = TFTextTransformer(
            inputCol=input_col, outputCol=output_col, embeddingSize=100, sequenceLength=64)

        df = transformer.transform(documentDF)

        def map_fun(args={}, ctx=None, _read_data=None):
            import time
            self.assertTrue(ctx is not None)
            self.assertTrue(_read_data is None)
            self.assertTrue(args["params"]["fitParam"][0]["cluster_size"] == 2)
            clusterMode = ctx is not None
            if clusterMode and ctx.job_name == "ps":
                time.sleep((ctx.worker_num + 1) * 5)

            if clusterMode:
                cluster, server = TFNode.start_cluster_server(ctx, 1)

            data = TFNode.DataFeed(ctx.mgr, True)
            batch1 = data.next_batch(1)
            self.assertTrue(len(batch1) == 1)
            self.assertTrue(len(batch1[0]) == 64)
            self.assertTrue(len(batch1[0][0]) == 100)
            # consume all
            data.next_batch(100)

        estimator = TFTextFileEstimator(inputCol="sentence_matrix", outputCol="sentence_matrix", labelCol="preds",
                                        fitParam=[
                                            {"epochs": 1, "cluster_size": 2, "batch_size": 1, "model": "/tmp/model"}],
                                        runningMode="TFoS",
                                        mapFnParam=map_fun)
        estimator.fit(df).collect()


class MockKakfaServerTest(SparkDLTestCase):
    def test_mockKafkaServerProduce(self):
        dataset = self.session.createDataFrame([
            ("Hi I heard about Spark", 1),
            ("I wish Java could use case classes", 0),
            ("Logistic regression models are neat", 2)
        ], ["text", "preds"])

        def _write_data():
            def _write_partition(index, d_iter):
                producer = KafkaMockServer(index)
                try:
                    for d in d_iter:
                        producer.send("", pickle.dumps(d))
                    producer.send("", pickle.dumps("_stop_"))
                    producer.flush()
                finally:
                    producer.close()
                return []

            dataset.rdd.mapPartitionsWithIndex(_write_partition).count()

        _write_data()

        def _consume():
            consumer = KafkaMockServer()
            stop_count = 0
            while True:
                messages = consumer.poll(timeout_ms=1000, max_records=64)
                group_msgs = []
                for tp, records in messages.items():
                    for record in records:
                        try:
                            msg_value = pickle.loads(record.value)
                            print(msg_value)
                            if msg_value == "_stop_":
                                stop_count += 1
                            else:
                                group_msgs.append(msg_value)
                        except:
                            pass
                if stop_count >= 8:
                    break
            self.assertEquals(stop_count, 8)

            t = threading.Thread(target=_consume)
            t.start()
            t2 = threading.Thread(target=_consume)
            t2.start()
