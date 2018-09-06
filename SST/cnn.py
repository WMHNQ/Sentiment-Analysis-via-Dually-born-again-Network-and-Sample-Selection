import tensorflow as tf
import numpy as np


class CNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, word_embedding, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        self.word_embedding = tf.constant(word_embedding, name='word_embedding')
        self.embedded = tf.nn.embedding_lookup(self.word_embedding, self.input_x)
        self.embedded_expanded = tf.expand_dims(self.embedded, -1)
        self.embedded_expanded = tf.cast(self.embedded_expanded, tf.float32)
        embedding_size = word_embedding.shape[1]
        print('embedding_size {}'.format(embedding_size))


        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.softmax = tf.nn.softmax(self.scores,name = "softmax")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
class CNN_self(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, word_embedding, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_ypre = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.input_ytrue = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        self.word_embedding = tf.constant(word_embedding, name='word_embedding')
        self.embedded = tf.nn.embedding_lookup(self.word_embedding, self.input_x)
        self.embedded_expanded = tf.expand_dims(self.embedded, -1)
        self.embedded_expanded = tf.cast(self.embedded_expanded, tf.float32)
        embedding_size = word_embedding.shape[1]
        print('embedding_size {}'.format(embedding_size))

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output1"):
            W1 = tf.get_variable(
                "W1",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b1")
            l2_loss += tf.nn.l2_loss(W1)
            l2_loss += tf.nn.l2_loss(b1)
            self.scores1 = tf.nn.xw_plus_b(self.h_drop, W1, b1, name="scores1")
            self.predictions1 = tf.argmax(self.scores1, 1, name="predictions1")
            self.softmax1 = tf.nn.softmax(self.scores1, name="softmax1")
        with tf.name_scope("output2"):
            W2 = tf.get_variable(
                "W2",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b2")
            l2_loss += tf.nn.l2_loss(W2)
            l2_loss += tf.nn.l2_loss(b2)
            self.scores2 = tf.nn.xw_plus_b(self.h_drop, W2, b2, name="scores2")
            self.predictions2 = tf.argmax(self.scores2, 1, name="predictions2")
            self.softmax2 = tf.nn.softmax(self.scores2, name="softmax2")
            self.ensemble_softmax = tf.concat([self.softmax1, self.softmax2], 1)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores1, labels=self.input_ypre)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores2, labels=self.input_ytrue)
            self.loss = tf.reduce_mean(losses1)+ tf.reduce_mean(losses2) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions1 = tf.equal(self.predictions1, tf.argmax(self.input_ypre, 1))
            self.accuracy1 = tf.reduce_mean(tf.cast(correct_predictions1, "float"), name="accuracy1")
            correct_predictions2 = tf.equal(self.predictions2, tf.argmax(self.input_ytrue, 1))
            self.accuracy2 = tf.reduce_mean(tf.cast(correct_predictions2, "float"), name="accuracy1")

class CNN_self2(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, word_embedding, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_ypre = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.input_ytrue = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        self.word_embedding = tf.constant(word_embedding, name='word_embedding')
        self.embedded = tf.nn.embedding_lookup(self.word_embedding, self.input_x)
        self.embedded_expanded = tf.expand_dims(self.embedded, -1)
        self.embedded_expanded = tf.cast(self.embedded_expanded, tf.float32)
        embedding_size = word_embedding.shape[1]
        print('embedding_size {}'.format(embedding_size))

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output1"):
            W1 = tf.get_variable(
                "W1",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b1")
            l2_loss += tf.nn.l2_loss(W1)
            l2_loss += tf.nn.l2_loss(b1)
            self.scores1 = tf.nn.xw_plus_b(self.h_drop, W1, b1, name="scores1")
            self.predictions1 = tf.argmax(self.scores1, 1, name="predictions1")
            self.softmax1 = tf.nn.softmax(self.scores1, name="softmax1")

            self.scores2 = tf.nn.xw_plus_b(self.h_drop, W1, b1, name="scores2")
            self.predictions2 = tf.argmax(self.scores2, 1, name="predictions2")
            self.softmax2 = tf.nn.softmax(self.scores2, name="softmax2")
            self.ensemble_softmax = tf.concat([self.softmax1, self.softmax2], 1)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores1, labels=self.input_ypre)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores2, labels=self.input_ytrue)
            self.loss = tf.reduce_mean(losses1)+ tf.reduce_mean(losses2) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions1 = tf.equal(self.predictions1, tf.argmax(self.input_ypre, 1))
            self.accuracy1 = tf.reduce_mean(tf.cast(correct_predictions1, "float"), name="accuracy1")
            correct_predictions2 = tf.equal(self.predictions2, tf.argmax(self.input_ytrue, 1))
            self.accuracy2 = tf.reduce_mean(tf.cast(correct_predictions2, "float"), name="accuracy1")


class CNN_weighttloss(object):
    def __init__(
            self, sequence_length, num_classes, word_embedding, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")  # X - The Data
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")  # Y - The Lables
        self.input_yw = tf.placeholder(tf.float32, [None, num_classes], name="input_yw")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")  # Dropout

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        self.word_embedding = tf.constant(word_embedding, name='word_embedding')
        self.embedded = tf.nn.embedding_lookup(self.word_embedding, self.input_x)
        self.embedded_expanded = tf.expand_dims(self.embedded, -1)
        self.embedded_expanded = tf.cast(self.embedded_expanded, tf.float32)
        embedding_size = word_embedding.shape[1]
        print('embedding_size {}'.format(embedding_size))

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.softmax = tf.nn.softmax(self.scores, name="softmax")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            weight = tf.multiply(self.input_y,self.input_yw )
            weight = tf.reduce_sum(weight,1)
            losses = tf.multiply(losses,weight )
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        print("(!!) LOADED LSTM-CNN! :)")
        # embed()


class CNN_born_weight(object):#haimeigai
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, word_embedding, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_ypre = tf.placeholder(tf.float32, [None, num_classes], name="input_yprey")
        self.input_ytrue = tf.placeholder(tf.float32, [None, num_classes], name="input_ytrue")
        self.input_yw = tf.placeholder(tf.float32, [None, num_classes], name="input_yw")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        self.word_embedding = tf.constant(word_embedding, name='word_embedding')
        self.embedded = tf.nn.embedding_lookup(self.word_embedding, self.input_x)
        self.embedded_expanded = tf.expand_dims(self.embedded, -1)
        self.embedded_expanded = tf.cast(self.embedded_expanded, tf.float32)
        embedding_size = word_embedding.shape[1]
        print('embedding_size {}'.format(embedding_size))

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output1"):
            W1 = tf.get_variable(
                "W1",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b1")
            l2_loss += tf.nn.l2_loss(W1)
            l2_loss += tf.nn.l2_loss(b1)
            self.scores1 = tf.nn.xw_plus_b(self.h_drop, W1, b1, name="scores1")
            self.predictions1 = tf.argmax(self.scores1, 1, name="predictions1")
            self.softmax1 = tf.nn.softmax(self.scores1, name="softmax1")
        with tf.name_scope("output2"):
            W2 = tf.get_variable(
                "W2",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b2")
            l2_loss += tf.nn.l2_loss(W2)
            l2_loss += tf.nn.l2_loss(b2)
            self.scores2 = tf.nn.xw_plus_b(self.h_drop, W2, b2, name="scores2")
            self.predictions2 = tf.argmax(self.scores2, 1, name="predictions2")
            self.softmax2 = tf.nn.softmax(self.scores2, name="softmax2")
            self.ensemble_softmax = tf.concat([self.softmax1, self.softmax2], 1)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores1, labels=self.input_ypre)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores2, labels=self.input_ytrue)
            weight = tf.multiply(self.input_ytrue, self.input_yw)
            weight = tf.reduce_sum(weight, 1)
            losses = tf.multiply(losses1 + losses2, weight)

            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions1 = tf.equal(self.predictions1, tf.argmax(self.input_ypre, 1))
            self.accuracy1 = tf.reduce_mean(tf.cast(correct_predictions1, "float"), name="accuracy1")
            correct_predictions2 = tf.equal(self.predictions2, tf.argmax(self.input_ytrue, 1))
            self.accuracy2 = tf.reduce_mean(tf.cast(correct_predictions2, "float"), name="accuracy1")
class CNN_born_learnweight(object):#haimeigai
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, word_embedding, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_ypre = tf.placeholder(tf.float32, [None, num_classes], name="input_yprey")
        self.input_ytrue = tf.placeholder(tf.float32, [None, num_classes], name="input_ytrue")
        self.input_yw1 = tf.placeholder(tf.float32, [None, num_classes], name="input_yw1")
        self.input_yw2 = tf.placeholder(tf.float32, [None, num_classes], name="input_yw2")
        self.input_yw3 = tf.placeholder(tf.float32, [None, num_classes], name="input_yw3")
        self.balance_gate = tf.placeholder(tf.float32, [1], name="balance_gate")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        self.word_embedding = tf.constant(word_embedding, name='word_embedding')
        self.embedded = tf.nn.embedding_lookup(self.word_embedding, self.input_x)
        self.embedded_expanded = tf.expand_dims(self.embedded, -1)
        self.embedded_expanded = tf.cast(self.embedded_expanded, tf.float32)
        embedding_size = word_embedding.shape[1]
        print('embedding_size {}'.format(embedding_size))

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output1"):
            W1 = tf.get_variable(
                "W1",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b1")
            l2_loss += tf.nn.l2_loss(W1)
            l2_loss += tf.nn.l2_loss(b1)
            self.scores1 = tf.nn.xw_plus_b(self.h_drop, W1, b1, name="scores1")
            self.predictions1 = tf.argmax(self.scores1, 1, name="predictions1")
            self.softmax1 = tf.nn.softmax(self.scores1, name="softmax1")
        with tf.name_scope("output2"):
            W2 = tf.get_variable(
                "W2",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b2")
            l2_loss += tf.nn.l2_loss(W2)
            l2_loss += tf.nn.l2_loss(b2)
            self.scores2 = tf.nn.xw_plus_b(self.h_drop, W2, b2, name="scores2")
            self.predictions2 = tf.argmax(self.scores2, 1, name="predictions2")
            self.softmax2 = tf.nn.softmax(self.scores2, name="softmax2")
            self.ensemble_softmax = tf.concat([self.softmax1, self.softmax2], 1)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores1, labels=self.input_ypre)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores2, labels=self.input_ytrue)
            weight1 = tf.multiply(self.input_ytrue, self.input_yw1)
            weight2 = tf.multiply(self.input_ytrue, self.input_yw2)
            weight3 = tf.multiply(self.input_ytrue, self.input_yw3)
            weight1 = tf.reduce_sum(weight1, 1)
            weight2 = tf.reduce_sum(weight2, 1)
            weight3 = tf.reduce_sum(weight3, 1)
            W1 = tf.Variable(tf.random_normal([1, 1]), name="W1")
            W2 = tf.Variable(tf.random_normal([1, 1]), name="W2")
            W3 = tf.Variable(tf.random_normal([1, 1]), name="W3")
            weight = tf.sigmoid(weight1 * W1 + weight2 * W2 + weight3 * W3)
            losses = tf.multiply(losses1+losses2, weight) + self.balance_gate * (1 - weight)

            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions1 = tf.equal(self.predictions1, tf.argmax(self.input_ypre, 1))
            self.accuracy1 = tf.reduce_mean(tf.cast(correct_predictions1, "float"), name="accuracy1")
            correct_predictions2 = tf.equal(self.predictions2, tf.argmax(self.input_ytrue, 1))
            self.accuracy2 = tf.reduce_mean(tf.cast(correct_predictions2, "float"), name="accuracy1")
class CNN_learn_weighttloss(object):
    def __init__(
            self, sequence_length, num_classes, word_embedding, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")  # X - The Data
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")  # Y - The Lables
        self.input_yw1 = tf.placeholder(tf.float32, [None, num_classes], name="input_yw1")
        self.input_yw2 = tf.placeholder(tf.float32, [None, num_classes], name="input_yw2")
        self.input_yw3 = tf.placeholder(tf.float32, [None, num_classes], name="input_yw3")
        self.balance_gate = tf.placeholder(tf.float32, [1], name="balance_gate")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")  # Dropout

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        self.word_embedding = tf.constant(word_embedding, name='word_embedding')
        self.embedded = tf.nn.embedding_lookup(self.word_embedding, self.input_x)
        self.embedded_expanded = tf.expand_dims(self.embedded, -1)
        self.embedded_expanded = tf.cast(self.embedded_expanded, tf.float32)
        embedding_size = word_embedding.shape[1]
        print('embedding_size {}'.format(embedding_size))

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.softmax = tf.nn.softmax(self.scores, name="softmax")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            weight1 = tf.multiply(self.input_y, self.input_yw1)
            weight2 = tf.multiply(self.input_y, self.input_yw2)
            weight3 = tf.multiply(self.input_y, self.input_yw3)
            weight1 = tf.reduce_sum(weight1, 1)
            weight2 = tf.reduce_sum(weight2, 1)
            weight3 = tf.reduce_sum(weight3, 1)
            W1 = tf.Variable(tf.random_normal([1, 1]), name="W1")
            W2 = tf.Variable(tf.random_normal([1, 1]), name="W2")
            W3 = tf.Variable(tf.random_normal([1, 1]), name="W3")
            weight = tf.sigmoid(weight1 * W1 + weight2 * W2 + weight3 * W3)
            losses = tf.multiply(losses, weight) + self.balance_gate * (1 - weight)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        print("(!!) LOADED LSTM-CNN! :)")
        # embed()