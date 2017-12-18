# ---------------
# Author: Tu, Tao
# Reference: https://github.com/zsdonghao/text-to-image/blob/master/model.py
# ---------------
import Layer
import tensorflow as tf

def generator(z, txt, img_height, img_width, img_depth=3, gf_dim=128, is_train=True, reuse=tf.AUTO_REUSE):
    H, W, D = img_height, img_width, img_depth
    # z_dim = z.get_shape()[-1]
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.02)
    H2, H4, H8, H16 = int(H / 2), int(H / 4), int(H / 8), int(H / 16)
    W2, W4, W8, W16 = int(W / 2), int(W / 4), int(W / 8), int(W / 16)
    with tf.variable_scope('Generator', reuse=reuse):
        code = tf.concat([z, txt], axis=1, name='code')
        # layer 0
        h0 = Layer.dense(code, gf_dim * 8 * H16 * W16, act=tf.identity,
                         W_init=w_init, b_init=None, name='h0/dense')
        h0 = Layer.batch_norm(h0, is_train=is_train, gamma_init=gamma_init,
                              name='h0/batch_norm')
        h0_out = tf.reshape(
            h0, shape=[-1, H16, W16, gf_dim * 8], name='h0/reshape')
        # layer 1
        h1 = Layer.conv2d(h0_out, act=tf.identity, filter_shape=[1, 1, gf_dim * 8, gf_dim * 2],
                          strides=[1, 1, 1, 1], padding='VALID', W_init=w_init, b_init=None, name='h1/conv2d_1')
        h1 = Layer.batch_norm(h1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                              name='h1/batch_norm_1')
        h1 = Layer.conv2d(h1, act=tf.identity, filter_shape=[3, 3, gf_dim * 2, gf_dim * 2],
                          strides=[1, 1, 1, 1], padding='SAME', W_init=w_init, b_init=None, name='h1/conv2d_2')
        h1 = Layer.batch_norm(h1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                              name='h1/batch_norm_2')
        h1 = Layer.conv2d(h1, act=tf.identity, filter_shape=[3, 3, gf_dim * 2, gf_dim * 8],
                          strides=[1, 1, 1, 1], padding='SAME', W_init=w_init, b_init=None, name='h1/conv2d_3')
        h1 = Layer.batch_norm(h1, is_train=is_train,
                              gamma_init=gamma_init, name='h1/batch_norm_3')
        h1_res = tf.add(h1, h0_out, name='h1/residue_add')
        h1_out = tf.nn.relu(h1_res)
        # layer 2
        h2 = Layer.deconv2d(h1_out, act=tf.identity, filter_shape=[4, 4, gf_dim * 4, gf_dim * 8],
                            output_shape=[tf.shape(h1_out)[0], H8, W8, gf_dim * 4], strides=[1, 2, 2, 1], padding='SAME',
                            W_init=w_init, name='h2/deconv2d')
        h2_out = Layer.batch_norm(h2, act=tf.identity, is_train=is_train, gamma_init=gamma_init,
                                  name='h2/batch_norm')
        # layer 3
        h3 = Layer.conv2d(h2_out, act=tf.identity, filter_shape=[1, 1, gf_dim * 4, gf_dim],
                          strides=[1, 1, 1, 1], padding='VALID', W_init=w_init, b_init=None, name='h3/conv2d_1')
        h3 = Layer.batch_norm(h3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                              name='h3/batch_norm_1')
        h3 = Layer.conv2d(h3, act=tf.identity, filter_shape=[3, 3, gf_dim, gf_dim],
                          strides=[1, 1, 1, 1], padding='SAME', W_init=w_init, b_init=None, name='h3/conv2d_2')
        h3 = Layer.batch_norm(h3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                              name='h3/batch_norm_2')
        h3 = Layer.conv2d(h3, act=tf.identity, filter_shape=[3, 3, gf_dim, gf_dim * 4],
                          strides=[1, 1, 1, 1], padding='SAME', W_init=w_init, b_init=None, name='h3/conv2d_3')
        h3 = Layer.batch_norm(h3, is_train=is_train,
                              gamma_init=gamma_init, name='h3/batch_norm_3')
        h3_res = tf.add(h3, h2_out, name='h3/residue_add')
        h3_out = tf.nn.relu(h3_res)
        # layer 4
        h4 = Layer.deconv2d(h3_out, act=tf.identity, filter_shape=[4, 4, gf_dim * 2, gf_dim * 4],
                            output_shape=[tf.shape(h3_out)[0], H4, W4, gf_dim * 2], strides=[1, 2, 2, 1], padding='SAME',
                            W_init=w_init, name='h4/deconv2d')
        h4_out = Layer.batch_norm(h4, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                                  name='h4/batch_norm')
        # layer 5
        h5 = Layer.deconv2d(h4_out, act=tf.identity, filter_shape=[4, 4, gf_dim, gf_dim * 2],
                            output_shape=[tf.shape(h4_out)[0], H2, W2, gf_dim], strides=[1, 2, 2, 1], padding='SAME',
                            W_init=w_init, name='h5/deconv2d')
        h5_out = Layer.batch_norm(h5, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                                  name='h5/batch_norm')
        # layer 6
        h6 = Layer.deconv2d(h5_out, act=tf.identity, filter_shape=[4, 4, D, gf_dim],
                            output_shape=[tf.shape(h5_out)[0], H, W, D], strides=[1, 2, 2, 1], padding='SAME',
                            W_init=w_init, name='h6/deconv2d')
        logits = h6
        outputs = tf.nn.tanh(h6)
    return outputs, logits
