import tensorflow as tf
import numpy as np
 
class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, vocab_size,
      embedding_size, filter_sizes, num_filters):

    	# Placeholders for input, output and dropout
		self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
		with tf.device('/cpu:0'), tf.name_scope("embedding"):
		    W = tf.Variable(
		        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
		        name="W")
		    # [None,sequence_length,dim]
		    self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
		    # expansion is because the input to convolution is a 4-D tensor
		    self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
		    with tf.name_scope("conv-maxpool-%s" % filter_size):
		        # Convolution Layer
		        filter_shape = [filter_size, embedding_size, 1, num_filters]
		        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
		        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
		        self.conv = tf.nn.conv2d(
		            self.embedded_chars_expanded,
		            W,
		            strides=[1, 1, 1, 1],
		            padding="VALID",
		            name="conv")
		        # Apply nonlinearity
		        self.h = tf.nn.relu(tf.nn.bias_add(self.conv, b), name="relu")
		        # Max-pooling over the outputs
		        pooled = tf.nn.max_pool(
		            self.h,
		            ksize=[1, sequence_length - filter_size + 1, 1, 1],
		            strides=[1, 1, 1, 1],
		            padding='VALID',
		            name="pool")
		        pooled_outputs.append(pooled)
 
		# Combine all the pooled features
		num_filters_total = num_filters * len(filter_sizes)
		self.h_pool = tf.concat(3, pooled_outputs)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])


with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=100,
            vocab_size=2000,
            embedding_size=50,
            filter_sizes=[2,3,4,5],
            num_filters=5)

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        x_batch = np.asarray([np.random.randint(1999,size=100)])
        print x_batch.shape

        feed_dict = {
          cnn.input_x: x_batch
        }

        # Training loop. For each batch...
        e = sess.run(cnn.h_pool_flat,feed_dict=feed_dict)
        print e.shape
        print e