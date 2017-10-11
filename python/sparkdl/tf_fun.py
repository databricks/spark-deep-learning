def map_fun(_read_data, **args):
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
