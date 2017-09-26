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

# Installation

    pip install .

# Training server / cluster management

The example code above assumes that there is a tensorflow cluster
running a set of worker tasks and parameter server tasks, apropriately
named "/job:worker" "/job:ps" respectively. To set up this can be a
bit tiresome, and if all you want is to quickly get a cluster up and
running and parallelize your code, you can use the cluster management
tool provided with tfprism.

To install the cluster management tools, you need to do

    apt install parallel
    pip install .[server]

on each node in your cluster. Once you have done so you can run

    tfprism cluster start server1,server2,...serverN

to start your cluster. You need to be able to ssh without passwords
(using public key auth) to all servers listed. After this you can
connect to port grpc://server1:5600 using tensorflow.
