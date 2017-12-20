# ---------------
# Author: Tu, Tao
# Reference: https://github.com/zsdonghao/text-to-image/blob/master/train_txt2im.py
# ---------------
from Layer import cosine_similarity as cos_loss
import tensorflow as tf
import os
import logging
import time
import random
import numpy as np
from models import imageEncoder, textEncoder, generator as G, discriminator as D

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO)

# ---- Args ----
# G: z, txt, img_height, img_width, img_depth=3, gf_dim=128, is_train=True, reuse=tf.AUTO_REUSE
# D: x, txt, img_height, img_width, img_depth=3, df_dim=64, is_train=True, reuse=tf.AUTO_REUSE
# textEncoder: txt, vocab_size, with_matrix=False, reuse=tf.AUTO_REUSE, pad_token=0,
#              bidirectional=False, word_dim=256, sent_dim=128
# imageEncoder: x, out_dim=128, df_dim=64, is_train=True, reuse=tf.AUTO_REUSE
# --------------

hps_list = {
    'lr': 2e-4,
    'lr_decay': 0.5,
    'decay_every': 100,
    'beta1': 0.5,
    'beta2': 0.9,
    'clip_norm': 10.0,
    'n_critic': 1,
    'imH': 64,
    'imW': 64,
    'imD': 3,
    'max_seq_len': 20,
    'z_dim': 512,
    's_dim': 128,
    'w_dim': 256,
    'gf_dim': 128,
    'df_dim': 64,
    'voc_size': 6375,
    'pad_token': 6372
}


def hparas(hps_list):
    class Hparas(object):
        pass
    hps = Hparas()
    for hp in hps_list:
        hps.hp = hps_list[hp]
    return hps


class TrainHelper(object):
    """ For convenience. """

    def __init__(self, sess, hps):
        self.hps = hps
        self.sess = sess
        self.saver = None

    def build():
        hps = self.hps
        with tf.variable_scope('inputs'):
            # BOSS TED you need to change these bullshit inputs XD
            self.img = tf.placeholder(
                tf.float32, shape=[None, hps.imH, hps.imW, hps.imD], name='read_image')
            self.img_w = tf.placeholder(
                tf.float32, shape=[None, hps.imH, hps.imW, hps.imD], name='wrong_image')
            self.cap = tf.placeholder(
                tf.int64, shape=[None, hps.max_seq_len], name='real_caption')
            self.cap_w = tf.placeholder(
                tf.int64, shape=[None, hps.max_seq_len], name='wrong_caption')
            self.z = tf.placeholder(
                tf.float32, shape=[None, hps.z_dim], name='noise')

        with tf.variable_scope('models'):
            # x: image embedding
            # v: text embedding
            self.x = imageEncoder(self.img, out_dim=hps.s_dim, is_train=True)
            self.x_w = imageEncoder(
                self.img_w, out_dim=hps.s_dim, is_train=True, reuse=True)
            self.v = textEncoder(self.cap, hps.voc_size, pad_token=hps.pad_token,
                                 word_dim=hps.w_dim, sent_dim=hps.s_dim)
            self.v_w = textEncoder(self.cap_w, hps.voc_size, pad_token=hps.pad_token,
                                   word_dim=hps.w_dim, sent_dim=hps.s_dim, reuse=True)
            self.x_fake = G(self.z, self.v, img_height=hps.imH, img_width=hps.imW,
                            img_depth=hps.imD, gf_dim=hps.gf_dim, is_train=True)
            # real data
            self.d_real, self.logits_real = D(self.x, self.v, img_height=hps.imH, img_width=hps.imW,
                                              img_depth=hps.imD, df_dim=hps.df_dim, is_train=True)
            # fake data from generator
            _, self.logits_fake = D(self.x_fake, self.v, img_height=hps.imH, img_width=hps.imW,
                                    img_depth=hps.imD, df_dim=hps.df_dim, is_train=True, reuse=True)
            # mismatched data
            _, self.logits_mis = D(self.x, self.v_w, img_height=hps.imH, img_width=hps.imW,
                                   img_depth=hps.imD, df_dim=hps.df_dim, is_train=True, reuse=True)

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
            self.lr_op = tf.constant(hps.lr)
            # optimizer for generator
            self.g_opt = tf.train.AdamOptimizer(self.lr_op, beta1=hps.beta1,
                                                beta2=hps.beta2).minimize(self.g_loss, var_list=self.g_vars)
            # optimizer for discriminator
            self.d_opt = tf.train.AdamOptimizer(self.lr_op, beta1=hps.beta1,
                                                beta2=hps.beta2).minimize(self.d_loss, var_list=self.d_vars)
            # gradient clip to avoid explosion
            grads, _ = tf.clip_by_global_norm(tf.gradients(
                self.enr_loss, self.rnn_vars + self.cnn_vars), hps.clip_norm)
            # encoders optimizer
            self.enr_opt = tf.train.AdamOptimizer(self.lr_op, beta1=hps.beta1,
                                                  beta2=hps.beta2).apply_gradients(zip(grads, self.rnn_vars + self.cnn_vars))

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=5)

    def train(self, batch_size=64, epoch=1, ckpt_dir='./ckpt_model', log=True):
        hps = self.hps
        num_train_example = ??
        num_batch_per_epoch = num_train_example // batch_size
        sample_size = batch_size

        # train
        count = 1
        for ep in range(epoch):
            self.restore(ckpt_dir)
            if ep != 0 and (epoch % hps.decay_every == 0):
                _lr_decay = hps.lr_decay ** (ep // hps.decay_every)
                self.sess.run(tf.assign(self.lr_op, hps.lr * _lr_decay))
                print('New learning rate: %f' % hps.lr * _lr_decay)
            for step in range(num_batch_per_epoch):
                # get matched caption
                cap = ??
                # get mismatched caption
                cap_w = ??
                # get real image
                img = ??
                # get wrong image
                img_w = ??

                # get noise (normal(mean=0, stddev=1.0))
                b_z = np.random.normal(loc=0.0, scale=1.0, size=(
                    sample_size, hps.z_dim)).astype(np.float32)

                # for ep < 50: train rnn, cnn
                if ep < 50:
                    errEnr, _ = self.sess.run([enr_loss, enr_opt], feed_dict={
                        self.img: img,
                        self.img_w: img_w,
                        self.cap: cap,
                        self.cap_w: cap_w
                    })
                else:
                    errEnr = 0.0

                # update D
                errD, _ = self.sess.run([d_loss, d_opt], feed_dict={
                    self.img: img,
                    self.img_w: img_w,
                    self.cap: cap,
                    self.cap_w: cap_w,
                    self.z: b_z
                })
                # update G after D have been updated n_critic times
                if count % hps.n_critic == 0:
                    errG, _ = self.sess.run([g_loss, g_opt], feed_dict={
                        self.img: img,
                        self.cap: cap,
                        self.z: b_z
                    })
                count += 1

                if log:
                    logging.info(
                        'Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, g_loss: %.8f, d_loss: %.8f, encoder_loss: %.8f'
                        % (ep, epoch, step, num_batch_per_epoch, time.time() - step_time, errG, errD, errEnr))

            self.save(ckpt_dir=ckpt_dir, idx=ep)

        sess.close()

    def save(self, ckpt_dir='ckpt/', idx=0):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        self.saver.save(self.sess, os.path.join(
            ckpt_dir, 'model-%d.ckpt' % idx))

    def restore(self, ckpt_dir='ckpt/', idx=None):
        if idx:
            self.saver.restore(self.sess, os.path.join(
                ckpt_dir, 'model-%d.ckpt' % idx))
            return True
        else:
            latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
            if latest_ckpt:
                self.saver.restore(latest_ckpt)
                return True
        return False


if __name__ == '__main__':
    hps = hparas(hps_list)
    gpu_ratio = 0.333
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_ratio)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    helper = TrainHelper(sess, hps)
    helper.build()
    helper.train(batch_size=64, epoch=1)
    helper.save(idx=666)

    sess.close()
