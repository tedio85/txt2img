# ---------------
# Author: Tu, Tao
# Reference: https://github.com/zsdonghao/text-to-image/blob/master/model.py
# ---------------

import Layer
import tensorflow as tf


def imageEncoder(x, out_dim=128, df_dim=64, is_train=True, reuse=tf.AUTO_REUSE):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.02)
    with tf.variable_scope('ImageEncoder', reuse=reuse):
        h0 = Layer.conv2d(x, act=tf.nn.leaky_relu, filter_shape=[4, 4, x.get_shape()[-1], df_dim], strides=[1, 2, 2, 1],
                          padding='SAME', W_init=w_init, b_init=None, name='h0/conv2d')
        # 1
        h1 = Layer.conv2d(h0, act=tf.identity, filter_shape=[4, 4, df_dim, df_dim * 2], strides=[1, 2, 2, 1],
                          padding='SAME', W_init=w_init, b_init=None, name='h1/conv2d')
        h1 = Layer.batch_norm(h1, act=tf.nn.leaky_relu, is_train=is_train,
                              gamma_init=gamma_init, name='h1/batch_norm')
        # 2
        h2 = Layer.conv2d(h1, act=tf.identity, filter_shape=[4, 4, df_dim * 2, df_dim * 4], strides=[1, 2, 2, 1],
                          padding='SAME', W_init=w_init, b_init=None, name='h2/conv2d')
        h2 = Layer.batch_norm(h2, act=tf.nn.leaky_relu, is_train=is_train,
                              gamma_init=gamma_init, name='h2/batch_norm')
        # 3
        h3 = Layer.conv2d(h2, act=tf.identity, filter_shape=[4, 4, df_dim * 4, df_dim * 8], strides=[1, 2, 2, 1],
                          padding='SAME', W_init=w_init, b_init=None, name='h3/conv2d')
        h3 = Layer.batch_norm(h3, act=tf.nn.leaky_relu, is_train=is_train,
                              gamma_init=gamma_init, name='h3/batch_norm')
        h3_flat = Layer.flatten(h3, name='h3/flatten')
        # 4
        h4 = Layer.dense(h3_flat, output_dim=out_dim, W_init=w_init, b_init=None, name='h4/dense')
    return h4


