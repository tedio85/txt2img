# ---------------
# Author: Tu, Tao
# Reference: https://github.com/zsdonghao/text-to-image/blob/master/model.py
# ---------------
import Layer
import tensorflow as tf


def discriminator(x, txt, img_height, img_width, img_depth=3, df_dim=64, is_train=True, reuse=tf.AUTO_REUSE):
    """ Determine if an image x condition on txt is real or fake. """
    H, W, D = img_height, img_width, img_depth
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.02)
    H2, H4, H8, H16 = int(H / 2), int(H / 4), int(H / 8), int(H / 16)
    W2, W4, W8, W16 = int(W / 2), int(W / 4), int(W / 8), int(W / 16)
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
        h3 = Layer.batch_norm(h3, act=tf.identity, is_train=is_train,
                              gamma_init=gamma_init, name='h3/batch_norm')
        # 4
        h4 = Layer.conv2d(h3, act=tf.identity, filter_shape=[1, 1, df_dim * 8, df_dim * 2], strides=[1, 1, 1, 1],
                          padding='VALID', W_init=w_init, b_init=None, name='h4/conv2d')
        h4 = Layer.batch_norm(h4, act=tf.nn.leaky_relu, is_train=is_train,
                              gamma_init=gamma_init, name='h4/batch_norm')
        # 5
        h5 = Layer.conv2d(h4, act=tf.identity, filter_shape=[3, 3, df_dim * 2, df_dim * 2], strides=[1, 1, 1, 1],
                          padding='SAME', W_init=w_init, b_init=None, name='h5/conv2d')
        h5 = Layer.batch_norm(h5, act=tf.nn.leaky_relu, is_train=is_train,
                              gamma_init=gamma_init, name='h5/batch_norm')
        # 6
        h6 = Layer.conv2d(h5, act=tf.identity, filter_shape=[3, 3, df_dim * 2, df_dim * 8], strides=[1, 1, 1, 1],
                          padding='SAME', W_init=w_init, b_init=None, name='h6/conv2d')
        h6 = Layer.batch_norm(h6, is_train=is_train,
                              gamma_init=gamma_init, name='h6/batch_norm')
        # residual
        h6_res = tf.add(h6, h3, name='h6/residue_add')
        h6_out = tf.nn.leaky_relu(h6_res)
        # txt: [batch_size, s_dim]
        # h6_out: [batch_size, _, _, df_dim*8]
        txt_expand = tf.expand_dims(txt, axis=1, name='expand_1')
        txt_expand = tf.expand_dims(txt_expand, axis=1, name='expand_2')
        txt_expand = tf.tile(txt_expand, multiples=[
                             1, h6_out.get_shape()[1], h6_out.get_shape()[2], 1])
        h_txt = tf.concat([h6_out, txt_expand], axis=3)
        # 7
        h7 = Layer.conv2d(h_txt, act=tf.identity, filter_shape=[1, 1, h_txt.get_shape()[-1], df_dim * 8],
                          strides=[1, 1, 1, 1], padding='VALID', W_init=w_init, b_init=None, name='h7/conv2d')
        h7 = Layer.batch_norm(h7, act=tf.nn.leaky_relu, is_train=is_train,
                              gamma_init=gamma_init, name='h7/batch_norm')
        h_flat = Layer.flatten(h7, name='h7/flat')
        logits = Layer.dense(h_flat, output_dim=1)
        outputs = tf.nn.sigmoid(logits)
    return outputs, logits
