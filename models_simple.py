# ---------------
# Author: Tu, Tao
# Reference: https://github.com/zsdonghao/text-to-image/blob/master/model.py
# ---------------

import Layer
import tensorflow as tf

"""
This file contains following models
  - SentenceEncoder: transforms skip thought encodings into dimension of sent_dim
  - generator:     to generate images given noises and condition texts
  - discriminator: to discriminate real and fake images given condition texts

"""

def SentenceEncoder(skip_thought_encodings, batch_size, skip_thought_dim=2400,
                    sent_dim=256, reuse=tf.AUTO_REUSE):
    """A 3-layer FC layer that reduces sent_dim=2400 to sent_dim=256"""
    with tf.variable_scope('SentenceEncoder', reuse=reuse):
        skip_thought_encodings.set_shape([batch_size, skip_thought_dim])
        fc1 = tf.layers.dense(skip_thought_encodings,
                              skip_thought_dim//2,
                              name='fc1')
        fc2 = tf.layers.dense(fc1,
                              skip_thought_dim // 4,
                              name='fc2')
        fc3 = tf.layers.dense(fc2,
                              sent_dim,
                              name='fc3')
    return fc3

def generator(z, txt, img_height, img_width, img_depth=3, s_dim=128, gf_dim=128, is_train=True, reuse=tf.AUTO_REUSE):
    """ Generate image given a code [z, txt]. """
    H, W, D = img_height, img_width, img_depth
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.02)
    H2, H4, H8, H16 = int(H / 2), int(H / 4), int(H / 8), int(H / 16)
    W2, W4, W8, W16 = int(W / 2), int(W / 4), int(W / 8), int(W / 16)
    with tf.variable_scope('Generator', reuse=reuse):
        txt = Layer.dense(txt, s_dim, act=tf.nn.leaky_relu,
                          W_init=w_init, name='txt/dense')
        code = tf.concat([z, txt], axis=1, name='code')
        h0 = Layer.dense(code, gf_dim * 8 * H16 * W16, act=tf.identity,
                         W_init=w_init, b_init=None, name='h0/dense')
        h0 = tf.reshape(
            h0, shape=[-1, H16, W16, gf_dim * 8], name='h0/reshape')
        h0 = Layer.batch_norm(h0, act=tf.nn.relu, is_train=is_train,
                              gamma_init=gamma_init, name='h0/batch_norm')
        # 1
        h1 = Layer.deconv2d(h0, act=tf.identity, filter_shape=[4, 4, gf_dim * 4, gf_dim * 8],
                            output_shape=[tf.shape(h0)[0], H8, W8, gf_dim * 4], strides=[1, 2, 2, 1], padding='SAME',
                            W_init=w_init, name='h1/deconv2d')
        h1 = Layer.batch_norm(h1, act=tf.nn.relu, is_train=is_train,
                              gamma_init=gamma_init, name='h1/batch_norm')
        # 2
        h2 = Layer.deconv2d(h1, act=tf.identity, filter_shape=[4, 4, gf_dim * 2, gf_dim * 4],
                            output_shape=[tf.shape(h1)[0], H4, W4, gf_dim * 2], strides=[1, 2, 2, 1], padding='SAME',
                            W_init=w_init, name='h2/deconv2d')
        h2 = Layer.batch_norm(h2, act=tf.nn.relu, is_train=is_train,
                              gamma_init=gamma_init, name='h2/batch_norm')
        # 3
        h3 = Layer.deconv2d(h2, act=tf.identity, filter_shape=[4, 4, gf_dim, gf_dim * 2],
                            output_shape=[tf.shape(h1)[0], H2, W2, gf_dim], strides=[1, 2, 2, 1], padding='SAME',
                            W_init=w_init, name='h3/deconv2d')
        h3 = Layer.batch_norm(h3, act=tf.nn.relu, is_train=is_train,
                              gamma_init=gamma_init, name='h3/batch_norm')
        # output
        h4 = Layer.deconv2d(h3, act=tf.identity, filter_shape=[4, 4, D, gf_dim],
                            output_shape=[tf.shape(h3)[0], H, W, D], strides=[1, 2, 2, 1], padding='SAME',
                            W_init=w_init, name='h10/deconv2d')
        logits = h4
        outputs = tf.div(tf.nn.tanh(logits) + 1, 2)
    return outputs


def discriminator(x, txt, img_height, img_width, img_depth=3, s_dim=128, df_dim=64, is_train=True, reuse=tf.AUTO_REUSE):
    """ Determine if an image x condition on txt is real or fake. """
    H, W, D = img_height, img_width, img_depth
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.02)
    # H2, H4, H8, H16 = int(H / 2), int(H / 4), int(H / 8), int(H / 16)
    # W2, W4, W8, W16 = int(W / 2), int(W / 4), int(W / 8), int(W / 16)
    with tf.variable_scope('Discriminator', reuse=reuse):
        h0 = Layer.conv2d(x, act=tf.nn.leaky_relu, filter_shape=[4, 4, D, df_dim], strides=[1, 2, 2, 1],
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

        # txt: [batch_size, s_dim]
        # h6_out: [batch_size, _, _, df_dim*8]
        txt = Layer.dense(txt, s_dim, act=tf.nn.leaky_relu,
                          W_init=w_init, name='txt/dense')
        txt_expand = tf.expand_dims(txt, axis=1, name='txt_expand_1')
        txt_expand = tf.expand_dims(txt_expand, axis=1, name='txt_expand_2')
        txt_expand = tf.tile(txt_expand, multiples=[
                             1, h3.get_shape()[1], h3.get_shape()[2], 1], name='txt_tile')
        h_txt = tf.concat([h3, txt_expand], axis=3)
        # output
        h4 = Layer.conv2d(h_txt, act=tf.identity, filter_shape=[1, 1, h_txt.get_shape()[-1], df_dim * 8],
                          strides=[1, 1, 1, 1], padding='VALID', W_init=w_init, b_init=None, name='h4/conv2d')
        h4 = Layer.batch_norm(h4, act=tf.nn.leaky_relu, is_train=is_train,
                              gamma_init=gamma_init, name='h4/batch_norm')
        h_flat = Layer.flatten(h4, name='h4/flat')
        logits = Layer.dense(h_flat, output_dim=1)
        outputs = tf.nn.sigmoid(logits)
    return outputs, logits
