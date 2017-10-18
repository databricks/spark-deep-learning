from pyspark.sql import SparkSession
from sparkdl.estimators.tf_text_file_estimator import TFTextFileEstimator, KafkaMockServer
from sparkdl.transformers.tf_text import TFTextTransformer


def map_fun(args={}, ctx=None, _read_data=None):
    from tensorflowonspark import TFNode
    from datetime import datetime
    import math
    import numpy
    import tensorflow as tf
    import time

    print(args)

    EMBEDDING_SIZE = args["embedding_size"]
    feature = args['feature']
    label = args['label']
    params = args['params']['fitParam'][0]
    SEQUENCE_LENGTH = 64

    clusterMode = False if ctx is None else True

    if clusterMode and ctx.job_name == "ps":
        time.sleep((ctx.worker_num + 1) * 5)

    if clusterMode:
        cluster, server = TFNode.start_cluster_server(ctx, 1)

    def feed_dict(batch):
        # Convert from dict of named arrays to two numpy arrays of the proper type
        features = []
        for i in batch:
            features.append(i['sentence_matrix'])

        # print("{} {}".format(feature, features))
        return features

    def build_graph():
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
        global_step = tf.Variable(0)
        init_op = tf.global_variables_initializer()
        return input_x, init_op, train_step, xent, global_step, summ

    def train_with_cluster(input_x, init_op, train_step, xent, global_step, summ):

        logdir = TFNode.hdfs_path(ctx, params['model']) if clusterMode else None
        sv = tf.train.Supervisor(is_chief=ctx.task_index == 0,
                                 logdir=logdir,
                                 init_op=init_op,
                                 summary_op=None,
                                 saver=None,
                                 global_step=global_step,
                                 stop_grace_secs=300,
                                 save_model_secs=10)
        with sv.managed_session(server.target) as sess:
            tf_feed = TFNode.DataFeed(ctx.mgr, True)
            step = 0

            while not sv.should_stop() and not tf_feed.should_stop() and step < 100:
                data = tf_feed.next_batch(params["batch_size"])
                batch_data = feed_dict(data)
                step += 1
                _, x, g = sess.run([train_step, xent, global_step], feed_dict={input_x: batch_data})
                print("global_step:{} xent:{}".format(g, x))

            if sv.should_stop() or step >= args.steps:
                tf_feed.terminate()
        sv.stop()

    def train(input_x, init_op, train_step, xent, global_step, summ):

        with tf.Session() as sess:
            sess.run(init_op)
            ## for i in range(echo)
            for data in _read_data(max_records=params["batch_size"]):
                batch_data = feed_dict(data)
                _, x, g = sess.run([train_step, xent, global_step], feed_dict={input_x: batch_data})
                print("global_step:{} xent:{}".format(x, g))

    if clusterMode and ctx.job_name == "ps":
        server.join()
    elif clusterMode and ctx.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % ctx.task_index,
                cluster=cluster)):
            input_x, init_op, train_step, xent, global_step, summ = build_graph()
        train_with_cluster(input_x, init_op, train_step, xent, global_step, summ)
    else:
        input_x, init_op, train_step, xent, global_step, summ = build_graph()
        train(input_x, init_op, train_step, xent, global_step, summ)


input_col = "text"
output_col = "sentence_matrix"

session = SparkSession.builder.master("spark://allwefantasy:7077").appName("test").getOrCreate()
documentDF = session.createDataFrame([
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
                                fitParam=[{"epochs": 1, "cluster_size": 2, "batch_size": 1, "model": "/tmp/model"}],
                                runningMode="TFoS",
                                mapFnParam=map_fun)
estimator.fit(df).collect()
