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

import shutil
import threading
import sys
from sparkdl.estimators.tf_text_file_estimator import TFTextFileEstimator, KafkaMockServer
from sparkdl.transformers.tf_text import TFTextTransformer
from ..tests import SparkDLTestCase

if sys.version_info[:2] <= (2, 7):
    import cPickle as pickle
else:
    import _pickle as pickle


def map_fun(args={}, ctx=None, _read_data=None):
    import tensorflow as tf
    EMBEDDING_SIZE = args["embedding_size"]
    params = args['params']['fitParam']
    SEQUENCE_LENGTH = 64

    def feed_dict(batch):
        # Convert from dict of named arrays to two numpy arrays of the proper type
        features = []
        for i in batch:
            features.append(i['sentence_matrix'])

        # print("{} {}".format(feature, features))
        return features

    encoder_variables_dict = {
        "encoder_w1": tf.Variable(
            tf.random_normal([SEQUENCE_LENGTH * EMBEDDING_SIZE, 256]), name="encoder_w1"),
        "encoder_b1": tf.Variable(tf.random_normal([256]), name="encoder_b1"),
        "encoder_w2": tf.Variable(tf.random_normal([256, 128]), name="encoder_w2"),
        "encoder_b2": tf.Variable(tf.random_normal([128]), name="encoder_b2")
    }

    def encoder(x, name="encoder"):
        with tf.name_scope(name):
            encoder_w1 = encoder_variables_dict["encoder_w1"]
            encoder_b1 = encoder_variables_dict["encoder_b1"]

            layer_1 = tf.nn.sigmoid(tf.matmul(x, encoder_w1) + encoder_b1)

            encoder_w2 = encoder_variables_dict["encoder_w2"]
            encoder_b2 = encoder_variables_dict["encoder_b2"]

            layer_2 = tf.nn.sigmoid(tf.matmul(layer_1, encoder_w2) + encoder_b2)
            return layer_2

    def decoder(x, name="decoder"):
        with tf.name_scope(name):
            decoder_w1 = tf.Variable(tf.random_normal([128, 256]))
            decoder_b1 = tf.Variable(tf.random_normal([256]))

            layer_1 = tf.nn.sigmoid(tf.matmul(x, decoder_w1) + decoder_b1)

            decoder_w2 = tf.Variable(
                tf.random_normal([256, SEQUENCE_LENGTH * EMBEDDING_SIZE]))
            decoder_b2 = tf.Variable(
                tf.random_normal([SEQUENCE_LENGTH * EMBEDDING_SIZE]))

            layer_2 = tf.nn.sigmoid(tf.matmul(layer_1, decoder_w2) + decoder_b2)
            return layer_2

    tf.reset_default_graph
    sess = tf.Session()

    input_x = tf.placeholder(tf.float32, [None, SEQUENCE_LENGTH, EMBEDDING_SIZE], name="input_x")
    flattened = tf.reshape(input_x,
                           [-1, SEQUENCE_LENGTH * EMBEDDING_SIZE])

    encoder_op = encoder(flattened)

    tf.add_to_collection('encoder_op', encoder_op)

    y_pred = decoder(encoder_op)

    y_true = flattened

    with tf.name_scope("xent"):
        consine = tf.div(tf.reduce_sum(tf.multiply(y_pred, y_true), 1),
                         tf.multiply(tf.sqrt(tf.reduce_sum(tf.multiply(y_pred, y_pred), 1)),
                                     tf.sqrt(tf.reduce_sum(tf.multiply(y_true, y_true), 1))))
        xent = tf.reduce_sum(tf.subtract(tf.constant(1.0), consine))
        tf.summary.scalar("xent", xent)

    with tf.name_scope("train"):
        # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(xent)
        train_step = tf.train.RMSPropOptimizer(0.01).minimize(xent)

    summ = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())

    for i in range(params["epochs"]):
        print("epoll {}".format(i))
        for data in _read_data(max_records=params["batch_size"]):
            batch_data = feed_dict(data)
            sess.run(train_step, feed_dict={input_x: batch_data})

    sess.close()


class TFTextFileEstimatorTest(SparkDLTestCase):
    def test_trainText(self):
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
        import tempfile
        mock_kafka_file = tempfile.mkdtemp()
        # create a estimator to training where map_fun contains tensorflow's code
        estimator = TFTextFileEstimator(inputCol="sentence_matrix", outputCol="sentence_matrix", labelCol="preds",
                                        kafkaParam={"bootstrap_servers": ["127.0.0.1"], "topic": "test",
                                                    "mock_kafka_file": mock_kafka_file,
                                                    "group_id": "sdl_1", "test_mode": True},
                                        runningMode="Normal",
                                        fitParam=[{"epochs": 5, "batch_size": 64}, {"epochs": 5, "batch_size": 1}],
                                        mapFnParam=map_fun)
        estimator.fit(df).collect()
        shutil.rmtree(mock_kafka_file)


class MockKakfaServerTest(SparkDLTestCase):
    def test_mockKafkaServerProduce(self):
        import tempfile
        mock_kafka_file = tempfile.mkdtemp()
        dataset = self.session.createDataFrame([
            ("Hi I heard about Spark", 1),
            ("I wish Java could use case classes", 0),
            ("Logistic regression models are neat", 2)
        ], ["text", "preds"])

        def _write_data():
            def _write_partition(index, d_iter):
                producer = KafkaMockServer(index, mock_kafka_file)
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
            consumer = KafkaMockServer(0, mock_kafka_file)
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
            import time
            time.sleep(10)
            shutil.rmtree(mock_kafka_file)
