# Tensorflow Prism

TFPrism is a library that transforms your tensorflow graph to
automatically do data parallelism for training. All you need to do to
modify your single-cpu tensorflow code to run on a cluster is to send
your training op and feed_dict through the library.


# Example code

    train_step = tf.train.GradientDescentOptimizer(0.9).minimize(loss)

    with tf.Session('grpc://mycluster.example.com:5600') as sess:
        train_step, node_copier = tfprism.distribute_graph_on_all_tasks(train_step, sess)
        sess.run(init_op)

        for batch in batches:
            sess.run(
                train_step,
                feed_dict=node_copier.mangle_feed_dict(batch))
