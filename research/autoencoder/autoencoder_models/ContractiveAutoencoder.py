#!/usr/bin/env python
"""Contractive Autoencoder Implementation for Tensorflow Research Models."""
import tensorflow as tf
import numpy as np


def sparse_normal(w, mu=0., sigma=0.01, density=0.01):
    """SI: Initializes a tensor with few non-zero values in N(mu, sigma)."""
    vals = np.random.normal(
        scale=sigma,
        loc=np.zeros(w.get_shape(), dtype=np.float32)
        )
    indices = np.random.sample(vals.shape) > density
    vals[indices] = 0.
    w.assign(vals)
    return w


class ContractiveAutoencoder(object):
    """
    Contractive Autoencoder.

    Uses a special regularization method to enhance learned features.
    """

    def __init__(
        self,
        n_input,
        n_hidden,
        transfer_function=tf.sigmoid,
        optimizer=tf.train.AdamOptimizer(),
        dropout_probability=1.,
        contraction_level=1.,
        density=0.01,
        tied_weights=True
            ):
        """Constructor."""
        if dropout_probability > 1. or dropout_probability < 0.:
            raise ValueError('Dropout Probability must be in [0,1]')
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.dropout_probability = dropout_probability
        self.keep_prob = tf.placeholder(tf.float32)
        self.density = density
        self.tied_weights = tied_weights

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [self.n_input])
        self.linear_hidden = tf.add(
            tf.matmul(
                tf.nn.dropout(tf.expand_dims(self.x, 0), self.keep_prob),
                self.weights['w1']
                ),
            self.weights['b1']
            )
        self.hidden = self.transfer(self.linear_hidden)
        self.reconstruction = tf.add(
            tf.matmul(self.hidden, self.weights['w2']), self.weights['b2']
            )

        # cost
        """
        Compute the contraction penalty in the way that Ian describes:
        https://groups.google.com/d/topic/pylearn-dev/iY7swxgn-xI/discussion
        """
        g = tf.gradients(tf.reduce_sum(self.hidden), self.linear_hidden)[0]
        g = tf.square(g)
        _w = tf.square(tf.reduce_sum(self.weights['w1'], axis=0))
        j = tf.multiply(g, _w)
        contraction_loss = tf.reduce_sum(j)
        if self.transfer == tf.sigmoid:
            reconstruction_loss = 0.5 * tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.x, logits=self.reconstruction[0]
                    )
                )
        else:
            reconstruction_loss = 0.5 * tf.reduce_sum(
                tf.pow(tf.subtract(self.reconstruction, self.x), 2.0)
                )
        self.cost = reconstruction_loss + contraction_level * contraction_loss
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        """Initialize the network weights and biases."""
        with tf.variable_scope('ContractiveAutoencoder'):
            all_weights = {}
            w1 = tf.Variable(
                tf.zeros([self.n_input, self.n_hidden]), dtype=tf.float32
                )
            all_weights['w1'] = sparse_normal(w1, density=self.density)
            all_weights['b1'] = tf.Variable(
                tf.zeros([self.n_hidden], dtype=tf.float32)
                )
            if self.tied_weights:
                all_weights['w2'] = tf.transpose(all_weights['w1'])
            else:
                all_weights['w2'] = tf.Variable(
                    tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32)
                    )
            all_weights['b2'] = tf.Variable(
                tf.zeros([self.n_input], dtype=tf.float32)
                )
            return all_weights

    def partial_fit(self, X):
        """Run training for one epoch."""
        cost = 0.
        for x in X:
            cost += self.sess.run(
                (self.cost, self.optimizer),
                feed_dict={self.x: x, self.keep_prob: self.dropout_probability}
                )[0]
        return cost

    def calc_total_cost(self, X):
        """Calculate total cost on example set X."""
        cost = 0.
        for x in X:
            print(x.shape)
            cost += self.sess.run(
                self.cost,
                feed_dict={self.x: x, self.keep_prob: 1.0}
                )
        return cost

    def transform(self, X):
        """Encode the example set X."""
        return np.stack([
            self.sess.run(
                self.hidden,
                feed_dict={self.x: x, self.keep_prob: 1.0}
                )
            for x in X
            ])

    def generate(self, hidden=None):
        """Decode the example string."""
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return np.stack([
            self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})
            ])

    def reconstruct(self, X):
        """Encode and decode the example set X."""
        return np.stack([
            self.sess.run(
                self.reconstruction,
                feed_dict={self.x: x, self.keep_prob: 1.0})
            for x in X
            ])

    def getWeights(self):
        """Getter for Weights."""
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        """Getter for Biases."""
        return self.sess.run(self.weights['b1'])


if __name__ == '__main__':
    try:
        ContractiveAutoencoder(784, 100)
    except Exception as ex:
        print(ex)
