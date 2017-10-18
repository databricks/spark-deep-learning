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

# pylint: disable=protected-access
from __future__ import absolute_import, division, print_function

import logging
import threading
import time
import os
import sys

from kafka import KafkaConsumer
from kafka import KafkaProducer
from pyspark.ml import Estimator
from tensorflowonspark import TFCluster

from sparkdl.param import (
    keyword_only, HasLabelCol, HasInputCol, HasOutputCol)
from sparkdl.param.shared_params import KafkaParam, FitParam, MapFnParam, RunningMode
import sparkdl.utils.jvmapi as JVMAPI

if sys.version_info[:2] <= (2, 7):
    import cPickle as pickle
else:
    import _pickle as pickle

__all__ = ['TFTextFileEstimator']

logger = logging.getLogger('sparkdl')


class TFTextFileEstimator(Estimator, HasInputCol, HasOutputCol, HasLabelCol, KafkaParam, FitParam, RunningMode,
                          MapFnParam):
    """
    Build a Estimator from tensorflow or keras when backend is tensorflow.

    First,assume we have data in dataframe like following.

    .. code-block:: python
            documentDF = self.session.createDataFrame([
                                                        ("Hi I heard about Spark", 1),
                                                        ("I wish Java could use case classes", 0),
                                                        ("Logistic regression models are neat", 2)
                                                        ], ["text", "preds"])

            transformer = TFTextTransformer(
                                            inputCol=input_col,
                                            outputCol=output_col)

            df = transformer.transform(documentDF)

     TFTextTransformer will transform text column to  `output_col`, which is 2-D array.

     Then we create a tensorflow function.

     .. code-block:: python
         def map_fun(args={}, ctx=None, _read_data=None):
            import tensorflow as tf
            EMBEDDING_SIZE = args["embedding_size"]
            feature = args['feature']
            label = args['label']
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

     _read_data is a data generator. args provide hyper parameteres configured in this estimator.

     here is how to use _read_data:

     .. code-block:: python
        for data in _read_data(max_records=params.batch_size):
            batch_data = feed_dict(data)
            sess.run(train_step, feed_dict={input_x: batch_data})

     finally we can create  TFTextFileEstimator to train our model:

     .. code-block:: python
            estimator = TFTextFileEstimator(inputCol="sentence_matrix",
                                            outputCol="sentence_matrix", labelCol="preds",
                                            kafkaParam={"bootstrap_servers": ["127.0.0.1"], "topic": "test",
                                                    "group_id": "sdl_1"},
                                            fitParam=[{"epochs": 5, "batch_size": 64}, {"epochs": 5, "batch_size": 1}],
                                            mapFnParam=map_fun)
            estimator.fit(df)

    """

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, labelCol=None, kafkaParam=None, fitParam=None,
                 runningMode="Normal", mapFnParam=None):
        super(TFTextFileEstimator, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, labelCol=None, kafkaParam=None, fitParam=None,
                  runningMode="Normal", mapFnParam=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def fit(self, dataset, params=None):
        self._validateParams()
        if params is None:
            paramMaps = self.getFitParam()
        elif isinstance(params, (list, tuple)):
            if len(params) == 0:
                paramMaps = [dict()]
            else:
                self._validateFitParams(params)
                paramMaps = params
        elif isinstance(params, dict):
            paramMaps = [params]
        else:
            raise ValueError("Params must be either a param map or a list/tuple of param maps, "
                             "but got %s." % type(params))
        if self.getRunningMode() == "TFoS":
            return self._fitInCluster(dataset, paramMaps)
        else:
            return self._fitInParallel(dataset, paramMaps)

    def _validateParams(self):
        """
        Check Param values so we can throw errors on the driver, rather than workers.
        :return: True if parameters are valid
        """
        if not self.isDefined(self.inputCol):
            raise ValueError("Input column must be defined")
        if not self.isDefined(self.outputCol):
            raise ValueError("Output column must be defined")
        return True

    def _clusterModelDefaultValue(self, sc, args):
        if "cluster_size" not in args:
            executors = sc._conf.get("spark.executor.instances")
            num_executors = int(executors) if executors is not None else 1
            args['cluster_size'] = num_executors
            num_ps = 1
        if "num_ps" not in args:
            args['num_ps'] = 1
        if "tensorboard" not in args:
            args['tensorboard'] = None
        return args

    def _fitInCluster(self, dataset, paramMaps):
        sc = JVMAPI._curr_sc()

        temp_item = dataset.take(1)[0]
        vocab_s = temp_item["vocab_size"]
        embedding_size = temp_item["embedding_size"]

        baseParamMap = self.extractParamMap()
        baseParamDict = dict([(param.name, val) for param, val in baseParamMap.items()])

        args = self._clusterModelDefaultValue(sc, paramMaps[0])
        args["feature"] = self.getInputCol()
        args["label"] = self.getLabelCol()
        args["vacab_size"] = vocab_s
        args["embedding_size"] = embedding_size
        args["params"] = baseParamDict

        cluster = TFCluster.run(sc, self.getMapFnParam(), args, args['cluster_size'], args['num_ps'],
                                args['tensorboard'],
                                TFCluster.InputMode.SPARK)
        cluster.train(dataset.rdd, args["epochs"])
        cluster.shutdown()

    def _fitInParallel(self, dataset, paramMaps):

        inputCol = self.getInputCol()
        labelCol = self.getLabelCol()

        from time import gmtime, strftime
        kafaParams = self.getKafkaParam()
        topic = kafaParams["topic"] + "_" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        group_id = kafaParams["group_id"]
        bootstrap_servers = kafaParams["bootstrap_servers"]
        kafka_test_mode = kafaParams["test_mode"] if "test_mode" in kafaParams else False
        mock_kafka_file = kafaParams["mock_kafka_file"] if kafka_test_mode else None

        def _write_data():
            def _write_partition(index, d_iter):
                producer = KafkaMockServer(index, mock_kafka_file) if kafka_test_mode else KafkaProducer(
                    bootstrap_servers=bootstrap_servers)
                try:
                    for d in d_iter:
                        producer.send(topic, pickle.dumps(d))
                    producer.send(topic, pickle.dumps("_stop_"))
                    producer.flush()
                finally:
                    producer.close()
                return []

            dataset.rdd.mapPartitionsWithIndex(_write_partition).count()

        if kafka_test_mode:
            _write_data()
        else:
            t = threading.Thread(target=_write_data)
            t.start()

        stop_flag_num = dataset.rdd.getNumPartitions()
        temp_item = dataset.take(1)[0]
        vocab_s = temp_item["vocab_size"]
        embedding_size = temp_item["embedding_size"]

        sc = JVMAPI._curr_sc()

        paramMapsRDD = sc.parallelize(paramMaps, numSlices=len(paramMaps))

        # Obtain params for this estimator instance
        baseParamMap = self.extractParamMap()
        baseParamDict = dict([(param.name, val) for param, val in baseParamMap.items()])
        baseParamDictBc = sc.broadcast(baseParamDict)

        def _local_fit(override_param_map):
            # Update params
            params = baseParamDictBc.value
            params["fitParam"] = override_param_map

            def _read_data(max_records=64):
                consumer = KafkaMockServer(0, mock_kafka_file) if kafka_test_mode else KafkaConsumer(topic,
                                                                                                     group_id=group_id,
                                                                                                     bootstrap_servers=bootstrap_servers,
                                                                                                     auto_offset_reset="earliest",
                                                                                                     enable_auto_commit=False
                                                                                                     )
                try:
                    stop_count = 0
                    fail_msg_count = 0
                    while True:
                        if kafka_test_mode:
                            time.sleep(1)
                        messages = consumer.poll(timeout_ms=1000, max_records=max_records)
                        group_msgs = []
                        for tp, records in messages.items():
                            for record in records:
                                try:
                                    msg_value = pickle.loads(record.value)
                                    if msg_value == "_stop_":
                                        stop_count += 1
                                    else:
                                        group_msgs.append(msg_value)
                                except:
                                    fail_msg_count += 0
                                    pass
                        if len(group_msgs) > 0:
                            yield group_msgs

                        if kafka_test_mode:
                            print(
                                "stop_count = {} "
                                "group_msgs = {} "
                                "stop_flag_num = {} "
                                "fail_msg_count = {}".format(stop_count,
                                                             len(group_msgs),
                                                             stop_flag_num,
                                                             fail_msg_count))

                        if stop_count >= stop_flag_num and len(group_msgs) == 0:
                            break
                finally:
                    consumer.close()

                self.getMapFnParam()(args={"feature": inputCol,
                                           "label": labelCol,
                                           "vacab_size": vocab_s,
                                           "embedding_size": embedding_size,
                                           "params": params}, ctx=None, _read_data=_read_data,
                                     )

        return paramMapsRDD.map(lambda paramMap: (paramMap, _local_fit(paramMap)))

    def _fit(self, dataset):  # pylint: disable=unused-argument
        err_msgs = ["This function should not have been called",
                    "Please contact library maintainers to file a bug"]
        raise NotImplementedError('\n'.join(err_msgs))


class KafkaMockServer(object):
    """
      Restrictions of KafkaMockServer:
       * Make sure all data have been writen before consume.
       * Poll function will just ignore max_records and just return all data in queue.
    """
    import tempfile
    _kafka_mock_server_tmp_file_ = None
    sended = False

    def __init__(self, index=0, mock_kafka_file=None):
        super(KafkaMockServer, self).__init__()
        self.index = index
        self.queue = []
        self._kafka_mock_server_tmp_file_ = mock_kafka_file
        if not os.path.exists(self._kafka_mock_server_tmp_file_):
            os.mkdir(self._kafka_mock_server_tmp_file_)

    def send(self, topic, msg):
        self.queue.append(pickle.loads(msg))

    def flush(self):
        with open(self._kafka_mock_server_tmp_file_ + "/" + str(self.index), "wb") as f:
            pickle.dump(self.queue, f)
        self.queue = []

    def close(self):
        pass

    def poll(self, timeout_ms, max_records):
        if self.sended:
            return {}

        records = []
        for file in os.listdir(self._kafka_mock_server_tmp_file_):
            with open(self._kafka_mock_server_tmp_file_ + "/" + file, "wb") as f:
                tmp = pickle.load(f)
                records += tmp
        result = {}
        couter = 0
        for i in records:
            obj = MockRecord()
            obj.value = pickle.dumps(i)
            couter += 1
            result[str(couter) + "_"] = [obj]
        self.sended = True
        return result


class MockRecord(list):
    pass
