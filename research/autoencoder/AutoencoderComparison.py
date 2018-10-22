#!/usr/bin/env python
"""
Runs several autoencoder models at once for easy comparison of results.

AUTHOR: Benjamin Mundt (bmundt6@gmail.com)
The purpose of this program is to compare the qualitative differences
between several types of autoencoder neural network.
Experiment by commenting or uncommenting the different model includes
and constructors.

NB: The images in the test/train folders come from the VIRAT dataset:

"A Large-scale Benchmark Dataset for Event Recognition in Surveillance Video"
by Sangmin Oh, Anthony Hoogs, Amitha Perera, Naresh Cuntoor, Chia-Chih Chen,
Jong Taek Lee, Saurajit Mukherjee, J.K. Aggarwal, Hyungtae Lee, Larry Davis,
Eran Swears, Xiaoyang Wang, Qiang Ji, Kishore Reddy, Mubarak Shah, Carl
Vondrick, Hamed Pirsiavash, Deva Ramanan, Jenny Yuen, Antonio Torralba, Bi
Song, Anesco Fong, Amit Roy-Chowdhury, and Mita Desai,
in Proceedings of IEEE Comptuer Vision and Pattern Recognition (CVPR), 2011.

"""

from __future__ import print_function, division

import numpy as np
import cv2
import os
import random
from glob import glob
import tensorflow as tf

imshape = None
color = True


def resize_concat(data, im):
    """Resize data or im to fit the other and concatenate them."""
    h, w = (max(data.shape[1], im.shape[1]), max(data.shape[2], im.shape[2]))
    if data.shape[1] < h or data.shape[2] < w:
        data = np.array([cv2.resize(x, (w, h)) for x in data])
    if im.shape[1] < h or im.shape[2] < w:
        im = cv2.resize(im[0, :], (w, h))[np.newaxis]
    return np.concatenate((data, im), axis=0)


def load_images(dirname, scale=0.25, n_examples=0):
    """Concatenate all images in a directory to one big array."""
    global imshape
    files = glob(os.path.join(dirname, '*'))
    if n_examples > 0:
        files = random.sample(files, n_examples)
    data = None
    print('Loading images from {}...'.format(dirname), end='')
    for f in files:
        if color:
            im = cv2.imread(f)
        else:
            im = cv2.imread(f, 0)
        if imshape is None:
            im = cv2.resize(
                    im, (0, 0), fx=scale, fy=scale
                    )[np.newaxis].astype(np.float64)
        else:
            im = cv2.resize(
                    im, (imshape[1], imshape[0])
                    )[np.newaxis].astype(np.float64)
        if data is None:
            data = im
        else:
            data = resize_concat(data, im)
    imshape = data.shape[1:]
    print('done.')
    return data


def standard_scale(X_train_in, X_test_in):
    """Produce data with zero mean and unit standard deviation."""
    mean = X_train_in.mean()
    std = X_train_in.std()
    X_train = (X_train_in - mean) / std
    X_test = (X_test_in - mean) / std
    return X_train, X_test


def min_max_scale(X_train_in, X_test_in):
    """Produce data in [0,1)."""
    X_min = X_train_in.min()
    X_max = X_train_in.max()
    X_train = (X_train_in - X_min) / (X_max - X_min)
    X_test = (X_test_in - X_min) / (X_max - X_min)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    """Randomly select batch_size number of contiguous samples."""
    start_index = random.randint(0, data.shape[0] - batch_size)
    return data[start_index:(start_index + batch_size)]


def draw_str(dst, pos, s):
    """Overlay a string on an image."""
    x, y = pos
    cv2.putText(
            dst, s, (x+1, y+1),
            cv2.FONT_HERSHEY_PLAIN,
            1.0, (0, 0, 0),
            thickness=2, lineType=cv2.LINE_AA
            )
    cv2.putText(
            dst, s, (x, y),
            cv2.FONT_HERSHEY_PLAIN,
            1.0, (255, 255, 255),
            lineType=cv2.LINE_AA
            )


class Test(object):
    """Class for displaying the output of a trained autoencoder."""

    def __init__(self, shape, autoencoders, scale=4):
        """Constructor."""
        self.shape = shape
        if color:
            h, w, c = shape
        else:
            h, w = shape
        self.autoencoders = autoencoders
        self.n_encoders = len(autoencoders)
        self.scale = scale
        self.names = [
            type(autoencoder).__name__ for autoencoder in self.autoencoders
            ]
        X_t = [
            (X_train_ss, X_train_mm)['Variational' in n or 'Contractive' in n]
            for n in self.names
            ]
        X_t = [
            (X, X_train)['Convolutional' in n]
            for (X, n) in zip(X_t, self.names)
            ]
        self.mean_img = [np.mean(X, axis=0) for X in X_t]
        self.X_test = [
            (X_test_ss, X_test_mm)['Variational' in n or 'Contractive' in n]
            for n in self.names
            ]
        self.X_test = [
            (X, X_test)['Convolutional' in n]
            for (X, n) in zip(self.X_test, self.names)
            ]
        self.n_examples = self.X_test[0].shape[0]
        if color:
            dispshape = [h*(len(autoencoders)+1), w*self.n_examples, c]
        else:
            dispshape = [h*(len(autoencoders)+1), w*self.n_examples]
        self.im = np.zeros(dispshape)
        for i in range(self.n_examples):
            if color:
                self.im[0:h, w*i:w*(i+1), :] = np.reshape(
                    X_test_mm[i, :], self.shape
                    )
            else:
                self.im[0:h, w*i:w*(i+1)] = np.reshape(
                    X_test_mm[i], self.shape
                    )

    def test_all(self):
        """Run the tests."""
        h, w = self.shape[:2]
        recon = [
            autoencoder.reconstruct(X)
            for autoencoder, X in zip(self.autoencoders, self.X_test)
            ]
        for j in range(self.n_encoders):
            for i in range(self.n_examples):
                if color:
                    self.im[(j+1)*h:(j+2)*h, w*i:w*(i+1), :] = np.reshape(
                        [recon[j][i, :] + self.mean_img[j]], self.shape
                        )
                else:
                    self.im[(j+1)*h:(j+2)*h, w*i:w*(i+1)] = np.reshape(
                        [np.squeeze(recon[j][i, :]) + self.mean_img[j]],
                        self.shape
                        )
        vis = cv2.resize(self.im, (0, 0), fx=self.scale, fy=self.scale)
        draw_str(vis, (15, 15), 'Original Image')
        for i, n in enumerate(self.names):
            draw_str(
                vis, (15, int(15+self.scale*h*(i+1))),
                '{} Reconstruction'.format(n)
                )
        cv2.imshow('out', vis)
        cv2.waitKey(1)


global_scale = .25  # WARNING: may crash if this is too high
repr_scale = 0.25
X_train = load_images('train', scale=global_scale)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
n_samples = X_train.shape[0]
imsize = X_train_flat.shape[1]
X_test = load_images('test', scale=global_scale, n_examples=8)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
training_epochs = 500
batch_size = n_samples // 5
total_batch = n_samples // batch_size
display_step = 1


def train(autoencoder, X):
    """Train an autoencoder on the training data."""
    cost = 0.
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X, batch_size)
        # Fit training using batch data
        cost += autoencoder.partial_fit(batch_xs)
    # Compute average loss
    return cost / (n_samples)


print('Scaling images...', end='')
X_train_ss, X_test_ss = standard_scale(X_train_flat, X_test_flat)
X_train_mm, X_test_mm = min_max_scale(X_train_flat, X_test_flat)
X_train, X_test = standard_scale(X_train, X_test)
print('done.')

if color:
    h, w, c = imshape
else:
    h, w = imshape
autoencoders = []

# Notes: Standard Autoencoder overfits too early --> low accuracy
# print('Initializing Autoencoder...', end='')
# from autoencoder_models.Autoencoder import Autoencoder # noqa: E402
# autoencoders.append(
#     Autoencoder(
#         n_input = imsize,
#         n_hidden = int(repr_scale * imsize),
#         transfer_function = tf.nn.softplus,
#         optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
#         )
#     )
# print('done.')

# Notes: Contractive Autoencoder does not allow batching since Jacobian
# requires dot product with hidden layer weights --> trains too slowly
#   Does, however, learn very good color representations
# print('Initializing Contractive Autoencoder...', end='')
# from autoencoder_models.ContractiveAutoencoder import (
#         ContractiveAutoencoder
#         ) # noqa: E402
# autoencoders.append(
#     ContractiveAutoencoder(
#         n_input=imsize,
#         n_hidden=int(repr_scale*imsize),
#         optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
#         dropout_probability=1.,
#         density=0.01,
#         contraction_level=0.5
#         )
#     )
# print('done.')

# Notes: Convolutional Autoencoder learns quickly, but needs greater
# depth for scale-invariant feature recognition
print('Initializing Convolutional Autoencoder...', end='')
from autoencoder_models.ConvolutionalAutoencoder import (
        ConvolutionalAutoencoder
        )  # noqa: E402
if color:
    autoencoders.append(
        ConvolutionalAutoencoder(
            input_shape=[None, h, w, c]
            )
        )
else:
    autoencoders.append(
        ConvolutionalAutoencoder(
            input_shape=[None, h, w]
            )
        )
print('done.')

# Notes: Variational Autoencoder is great for specialized applications (e.g.
# facial recognition), but requires too many
# latent variables for general recall
print('Initializing Variational Autoencoder...', end='')
from autoencoder_models.VariationalAutoencoder import (
    VariationalAutoencoder
    )  # noqa: E402
autoencoders.append(
    VariationalAutoencoder(
        n_input=imsize,
        n_hidden=int(repr_scale * imsize),
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001)
        )
    )
print('done.')

# Notes: Masking Autoencoder learns robust features quickly
print('Initializing Masking Noise Autoencoder...', end='')
from autoencoder_models.DenoisingAutoencoder import (
    MaskingNoiseAutoencoder
    )  # noqa: E402
autoencoders.append(
    MaskingNoiseAutoencoder(
        n_input=imsize,
        n_hidden=int(repr_scale * imsize),
        transfer_function=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
        dropout_probability=0.95
        )
    )
print('done.')

# Notes: Additive Gaussian Noise Autoencoder does not
# learn features as robust as Masking Autoencoder
# print('Initializing Additive Gaussian Noise Autoencoder...', end='')
# from autoencoder_models.DenoisingAutoencoder import (
#       AdditiveGaussianNoiseAutoencoder
#       ) # noqa: E402
# autoencoders.append(
#     AdditiveGaussianNoiseAutoencoder(
#         n_input = imsize,
#         n_hidden = int(repr_scale * imsize),
#         transfer_function = tf.nn.softplus,
#         optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
#         scale = 0.01
#         )
#     )
# print('done.')

tester = Test(imshape, autoencoders, scale=1./global_scale)
print('Training models...')
for epoch in range(training_epochs):
    for autoencoder in autoencoders:
        name = type(autoencoder).__name__
        if 'Variational' in name or 'Contractive' in name:
            X = X_train_mm
        elif 'Convolutional' in name:
            X = X_train
        else:
            X = X_train_ss
        avg_cost = train(autoencoder, X)
        # Display logs per epoch step
        if epoch % display_step == 0:
            print(
                "{} Epoch: {:>04d} Cost: {:>.9f}".format(name, epoch, avg_cost)
                )
    if epoch % display_step == 0:
        tester.test_all()
