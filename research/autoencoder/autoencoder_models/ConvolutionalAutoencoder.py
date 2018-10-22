#!/usr/bin/env python
"""
Tutorial on how to create a convolutional autoencoder w/ Tensorflow.

Parag K. Mital, Jan 2016
"""
import tensorflow as tf
import numpy as np
import math


def corrupt(x):
    """Take an input tensor and add uniform masking.

    Parameters
    ----------
    x : Tensor/Placeholder
        Input to corrupt.

    Returns
    -------
    x_corrupted : Tensor
        50 pct of values corrupted.

    """
    return tf.multiply(
        x,
        tf.cast(
            tf.random_uniform(
                shape=tf.shape(x), minval=0, maxval=2, dtype=tf.int32),
            tf.float32))


def lrelu(x, leak=0.2, name="lrelu"):
    """Leaky rectifier.

    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.

    Returns
    -------
    x : Tensor
        Output of the nonlinearity.

    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


class ConvolutionalAutoencoder(object):
    """Convolutional Autoencoder."""

    def __init__(self,
                 input_shape=[None, 784],
                 n_filters=[1, 10, 10, 10],
                 filter_sizes=[3, 3, 3, 3],
                 corruption=True,
                 transfer_function=lrelu,
                 optimizer=tf.train.AdamOptimizer(0.01)):
        """Build a deep denoising autoencoder w/ tied weights.

        Parameters
        ----------
        input_shape : list, optional
        n_filters : list, optional
        filter_sizes : list, optional

        Returns
        -------
        x : Tensor
            Input placeholder to the network
        z : Tensor
            Inner-most latent representation
        y : Tensor
            Output reconstruction of the input
        cost : Tensor
            Overall cost to use for training

        """
        with tf.variable_scope('ConvolutionalAutoencoder'):
            self.transfer_function = transfer_function
            # input to the network
            self.x = tf.placeholder(tf.float32, input_shape, name='x')

            # ensure 2-d is converted to square tensor.
            if len(self.x.get_shape()) == 2:
                x_dim = np.sqrt(self.x.get_shape().as_list()[1])
                if x_dim != int(x_dim):
                    raise ValueError('Unsupported input dimensions')
                x_dim = int(x_dim)
                x_tensor = tf.reshape(self.x, [-1, x_dim, x_dim, n_filters[0]])
            elif len(self.x.get_shape()) == 3:
                x_tensor = tf.expand_dims(self.x, -1)
            elif len(self.x.get_shape()) == 4:
                x_tensor = self.x
            else:
                raise ValueError('Unsupported input dimensions')
            current_input = x_tensor

            # Optionally apply denoising autoencoder
            if corruption:
                current_input = corrupt(current_input)

            # Build the encoder
            encoder = []
            shapes = []
            for layer_i, n_output in enumerate(n_filters[1:]):
                n_input = current_input.get_shape().as_list()[3]
                shapes.append(current_input.get_shape().as_list())
                W = tf.Variable(
                    tf.random_uniform([
                        filter_sizes[layer_i], filter_sizes[layer_i], n_input,
                        n_output
                    ], -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
                b = tf.Variable(tf.zeros([n_output]))
                encoder.append(W)
                output = self.transfer_function(
                    tf.add(
                        tf.nn.conv2d(
                            current_input,
                            W,
                            strides=[1, 2, 2, 1],
                            padding='SAME'), b))
                current_input = output

            # store the latent representation
            self.hidden = current_input
            encoder.reverse()
            shapes.reverse()

            # Build the decoder using the same weights
            for layer_i, shape in enumerate(shapes):
                W = encoder[layer_i]
                b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
                output = lrelu(
                    tf.add(
                        tf.nn.conv2d_transpose(
                            current_input,
                            W,
                            tf.stack([
                                tf.shape(self.x)[0], shape[1], shape[2],
                                shape[3]
                            ]),
                            strides=[1, 2, 2, 1],
                            padding='SAME'), b))
                current_input = output

            # now have the reconstruction through the network
            self.reconstruction = current_input
            # cost function measures pixel-wise difference
            self.cost = tf.reduce_sum(
                tf.square(self.reconstruction - x_tensor))
            self.optimizer = optimizer.minimize(self.cost)

            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def partial_fit(self, X):
        """Train for one epoch."""
        cost, _ = self.sess.run((self.cost, self.optimizer),
                                feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X):
        """Calculate the cost on example set X."""
        return self.sess.run(self.cost, feed_dict={self.x: X})

    def transform(self, X):
        """Encode the example set X."""
        return self.sess.run(self.hidden, feed_dict={self.x: X})

    def reconstruct(self, X):
        """Encode and decode the example string X."""
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})
