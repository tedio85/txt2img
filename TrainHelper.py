# ---------------
# Author: Tu, Tao
# Reference: https://github.com/zsdonghao/text-to-image/blob/master/train_txt2im.py
# ---------------
from Layer import cosine_similarity as cos_loss
import tensorflow as tf
from imageEncoder import imageEncoder
from textEncoder import textEncoder
from generator_resnet import generator as G
from discriminator_resnet import discriminator as D
import os
import random
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---- Args ----
# G: z, txt, img_height, img_width, img_depth=3, gf_dim=128, is_train=True, reuse=tf.AUTO_REUSE
# D: x, txt, img_height, img_width, img_depth=3, df_dim=64, is_train=True, reuse=tf.AUTO_REUSE
# textEncoder: txt, vocab_size, with_matrix=False, reuse=tf.AUTO_REUSE, pad_token=0,
#              bidirectional=False, word_dim=256, sent_dim=128
# imageEncoder: x, out_dim=128, df_dim=64, is_train=True, reuse=tf.AUTO_REUSE
# --------------


def hparas():
    return {
        'lr': 2e-4,
        'lr_decay': 0.5,
        'decay_every': 100,
        'beta1': 0.5,
        'beta2': 0.9,
        'n_critic': 5
    }


class TrainHelper(object):
    """ For convenience. """

    def __init__(self, hps, imH, imW, imD, z_dim, s_dim, w_dim, gf_dim, df_dim, max_seq_len, voc_size, pad_token):
        self.hps = hps
        self.imH = imH
        self.imW = imW
        self.imD = imD
        self.max_seq_len = max_seq_len
        self.z_dim = z_dim
        self.s_dim = s_dim
        self.w_dim = w_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.voc_size = voc_size
        self.pad_token = pad_token

        # z_dim = 512 # dim of noise
        # s_dim = 128 # dim of sentense embedding
        # w_dim = 256 # dim of word embedding
        # voc_size = 6375
        # pad_token = 6372

    def build():
        with tf.variable_scope('inputs'):
            # BOSS TED you need to change these bullshit inputs XD
            self.img = tf.placeholder(
                tf.float32, shape=[None, self.imH, self.imW, self.imD], name='read_image')
            self.img_w = tf.placeholder(
                tf.float32, shape=[None, self.imH, self.imW, self.imD], name='wrong_image')
            self.cap = tf.placeholder(
                tf.int64, shape=[None, self.max_seq_len], name='real_caption')
            self.cap_w = tf.placeholder(
                tf.int64, shape=[None, self.max_seq_len], name='wrong_caption')
            self.z = tf.placeholder(
                tf.float32, shape=[None, self.z_dim], name='noise')

        with tf.variable_scope('models'):
            # x: image embedding
            # v: text embedding
            self.x = imageEncoder(self.img, out_dim=self.s_dim, is_train=True)
            self.x_w = imageEncoder(
                self.img_w, out_dim=self.s_dim, is_train=True, reuse=True)
            self.v = textEncoder(self.cap, self.voc_size, pad_token=self.pad_token,
                                 word_dim=self.w_dim, sent_dim=self.s_dim)
            self.v_w = textEncoder(self.cap_w, self.voc_size, pad_token=self.pad_token,
                                   word_dim=self.w_dim, sent_dim=self.s_dim, reuse=True)
            self.x_fake = G(self.z, self.v, img_height=self.imH, img_width=self.imW,
                            img_depth=self.imD, gf_dim=self.gf_dim, is_train=True)
            # real data
            self.d_real, self.logits_real = D(self.x, self.v, img_height=self.imH, img_width=self.imW,
                                              img_depth=self.imD, df_dim=self.df_dim, is_train=True)
            # fake data from generator
            _, self.logits_fake = D(self.x_fake, self.v, img_height=self.imH, img_width=self.imW,
                                    img_depth=self.imD, df_dim=self.df_dim, is_train=True, reuse=True)
            # mismatched data
            _, self.logits_mis = D(self.x, self.v_w, img_height=self.imH, img_width=self.imW,
                                   img_depth=self.imD, df_dim=self.df_dim, is_train=True, reuse=True)

        with tf.variable_scope('losses'):
            alpha = 0.2
            # loss of encoders (txt & img)
            self.enr_loss = tf.reduce_mean(tf.maximum(0.0, alpha - cos_loss(self.x, self.v) + cos_loss(self.x, self.v_w))) + \
                tf.reduce_mean(tf.maximum(
                    0.0, alpha - cos_loss(self.x, self.v) + cos_loss(self.x_w, self.v)))
            # loss of generator
            self.g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits_fake, labels=tf.ones_like(self.logits_fake), name='d_loss_fake')
            # loss of discriminator
            self.d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits_real, labels=tf.ones_like(self.logits_real), name='d_loss_real')
            self.d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits_fake, labels=tf.zeros_like(self.logits_fake), name='d_loss_fake')
            self.d_loss_mis = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits_mis, labels=tf.zeros_like(self.logits_mis), name='d_loss_mismatch')
            self.d_loss = self.d_loss_real + 0.5 * \
                (self.d_loss_fake + self.d_loss_mis)

        # the power of name scope
        self.cnn_vars = [var for var in tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='models/ImageEncoder')]
        self.rnn_vars = [var for var in tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='models/TextEncoder')]
        self.g_vars = [var for var in tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='models/Generator')]
        self.d_vars = [var for var in tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='models/Discriminator')]

        with tf.variable_scope('optimizers'):
            self.lr_op = tf.constant(hps['lr'])
            # optimizer for generator
            self.g_opt = tf.train.AdamOptimizer(self.lr_op, beta1=hps['beta1'],
                                                beta2=hps['beta2']).minimize(self.g_loss, var_list=self.g_vars)
            # optimizer for discriminator
            self.d_opt = tf.train.AdamOptimizer(self.lr_op, beta1=hps['beta1'],
                                                beta2=hps['beta2']).minimize(self.d_loss, var_list=self.d_vars)
            # gradient clip to avoid explosion
            grads, _ = tf.clip_by_global_norm(tf.gradients(
                self.enr_loss, self.rnn_vars + self.cnn_vars), 10)
            # encoders optimizer
            self.enr_opt = tf.train.AdamOptimizer(self.lr_op, beta1=hps['beta1'],
                                                  beta2=hps['beta2']).apply_gradients(zip(grads, self.rnn_vars + self.cnn_vars))
        self.saver = tf.train.Saver()

    def train(num_train_example, batch_size=64, epoch=1, gpu_ratio=1):
        num_batch_per_epoch = num_train_example // batch_size
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_ratio)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())
        # train
        for ep in range(epoch):
            if ep != 0 and (epoch % hps['decay_every'] == 0):
                _lr_decay = self.hps[
                    'lr_decay'] ** (ep // self.hps['decay_every'])
                sess.run(tf.assign(self.lr_op, hps['lr'] * _lr_decay))
                print('New learning rate: %f' % hps['lr'] * _lr_decay)
            for step in range(num_batch_per_epoch):
                # get mismatched caption
                # get matched caption
                # get real image
                # get wrong image
                # get noise (normal(mean=0, stddev=1.0))
                # for ep < 50: train rnn, cnn
                # update D
                # update G after D have been updated n_critic times
                # Done

        sess.close()

    def save(self, ckpt_dir='ckpt/', idx=0):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        self.saver.save(self.sess, os.path.join(
            ckpt_dir, 'model-%d.ckpt' % idx))

    def restore(self, ckpt_dir='ckpt/', idx=0):
        self.saver.restore(self.sess, os.path.join(
            ckpt_dir, 'model-%d.ckpt' % idx))

if __name__ == '__main__':
    hps = hparas()
    imH, imW, imD = (64, 64, 3)
    max_seq_len = 20
    z_dim = 512  # dim of noise
    s_dim = 128  # dim of sentense embedding
    w_dim = 256  # dim of word embedding
    gf_dim = 128
    df_dim = 64
    voc_size = 6375
    pad_token = 6372

    helper = TrainHelper(hps, imH, imW, imD, z_dim, s_dim,
                         w_dim, gf_dim, df_dim, max_seq_len, voc_size, pad_token)
    helper.build()
    helper.train(batch_size=64, epoch=1, gpu_ratio=0.333)
