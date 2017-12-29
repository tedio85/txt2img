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
from models_simple import SentenceEncoder, generator as G, discriminator as D

import tracemalloc
#from pympler.tracker import SummaryTracker
#from util import sent2IdList, save_images

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO)

# ---- Args ----
# G: z, txt, img_height, img_width, img_depth=3, gf_dim=128, is_train=True, reuse=tf.AUTO_REUSE
# D: x, txt, img_height, img_width, img_depth=3, df_dim=64, is_train=True, reuse=tf.AUTO_REUSE
# textEncoder: txt, vocab_size, with_matrix=False, reuse=tf.AUTO_REUSE, pad_token=0,
#              bidirectional=False, word_dim=256, sent_dim=128
# imageEncoder: x, out_dim=128, df_dim=64, is_train=True, reuse=tf.AUTO_REUSE
# --------------


DIR_RECORD = '/tmp/md/ted_tmp/comp4/tfrecord/train/'
TEST_RECORD = '/tmp/md/ted_tmp/comp4/tfrecord/test/test.tfrecord'
VIS_ENCODE = '/tmp/md/ted_tmp/comp4/tfrecord/vis/vis_sentences.npy'
SUMMARY_DIR = '/tmp/md/ted_tmp/comp4/tao_summary'

hps_list = {
    'lr': 2e-4,
    'beta1': 0.5,
    'beta2': 0.9,
    'clip_norm': 1e-1,
    'n_critic': 1,
    'imH': 64,
    'imW': 64,
    'imD': 3,
    'max_seq_len': 20,
    'z_dim': 128,
    's_dim': 256,
    'w_dim': 256,
    'gf_dim': 128,
    'df_dim': 64,
    'voc_size': 6375,
    'pad_token': 6372
}

# get the number of records in training files
def get_num_records(files):
  count = 0
  for fn in files:
    for record in tf.python_io.tf_record_iterator(fn):
      count += 1
  return count

def hparas(hps_list):
    class Hparas(object):
        pass
    hps = Hparas()
    for hp in hps_list:
        setattr(hps, hp, hps_list[hp])
    return hps


class ModelWrapper(object):
    """ For convenience. """

    def __init__(self, sess, hps, train_files, batch_size, use_bn=True, is_train=True):
        self.hps = hps
        self.sess = sess
        self.saver = None
        self.files_tr = train_files
        self.batch_size = batch_size
        self.use_bn = use_bn
        self.is_train = is_train
        
        sample_encodings = np.load(VIS_ENCODE)
        vis = tf.constant(sample_encodings, shape=[8, 2400], dtype=tf.float32, name='vis_encode')
        vis = tf.tile(vis, [1, self.batch_size // 8])
        self.vis = tf.reshape(vis, [-1, 2400])
        
        self.writer = tf.summary.FileWriter(SUMMARY_DIR, self.sess.graph)

    
    def _build_dataset(self):
        def training_parser(record):
            ''' parse record from .tfrecord file and create training record

                :args
                    record - each record extracted from .tfrecord

                :return
                    a dictionary contains {
                        'img': image array extracted from vgg16 (256-dim) (Tensor),
                        'input_seq': a list of word id
                                  which describes input caption sequence (Tensor),
                        'output_seq': a list of word id
                                  which describes output caption sequence (Tensor),
                        'mask': a list of one which describe
                                  the length of input caption sequence (Tensor)
                    }
            '''

            keys_to_features = {
                "sentence": tf.VarLenFeature(dtype=tf.string),
                "encoding": tf.VarLenFeature(dtype=tf.float32),
                "fake_sentence": tf.VarLenFeature(dtype=tf.string),
                "fake_encoding": tf.VarLenFeature(dtype=tf.float32),
                "image": tf.VarLenFeature(dtype=tf.float32)
            }

            features = tf.parse_single_example(
                record, features=keys_to_features)

            sentence = features['sentence'].values
            encoding = features['encoding'].values
            fake_sentence = features['fake_sentence'].values
            fake_encoding = features['fake_encoding'].values
            image = features['image'].values

            records = {
                'sentence': sentence,
                'encoding': encoding,
                "fake_sentence": fake_sentence,
                "fake_encoding": fake_encoding,
                "image": image
            }
            return records

        def testing_parser(record):
            ''' parse record from .tfrecord file and create training record

              :args
                  record - each record extracted from .tfrecord

              :return
                  a dictionary contains {
                      'img': image array extracted from vgg16 (256-dim) (Tensor),
                      'input_seq': a list of word id
                                which describes input caption sequence (Tensor),
                      'output_seq': a list of word id
                                which describes output caption sequence (Tensor),
                      'mask': a list of one which describe
                                the length of input caption sequence (Tensor)
                  }
            '''
            keys_to_features = {
                  "sentence": tf.VarLenFeature(dtype=tf.string),
                  "encoding": tf.VarLenFeature(dtype=tf.float32)
            }

            # features contains - 'img', 'caption'
            features = tf.parse_single_example(
                record, features=keys_to_features)

            sentence = features['sentence'].values
            encoding = features['encoding'].values

            records = {
                  'sentence': sentence,
                  'encoding': encoding
            }
            return records

        def training_tfrecord_iterator(filenames, batch_size, record_parser):
            ''' create iterator to eat tfrecord dataset

            :args
                filenames     - a list of filenames (string)
                batch_size    - batch size (positive int)
                record_parser - a parser that read tfrecord
                                and create example record (function)

            :return
                iterator      - an Iterator providing a way
                                to extract elements from the created dataset.
                output_types  - the output types of the created dataset.
                output_shapes - the output shapes of the created dataset.
            '''
            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.map(record_parser, num_parallel_calls=16)

            dataset = dataset.repeat()             # repeat dataset infinitely

            # padded into equal length in each batch
            dataset = dataset.padded_batch(
                batch_size=batch_size,
                padded_shapes={
                  'sentence': [None],
                  'encoding': [None],
                  'fake_sentence': [None],
                  'fake_encoding': [None],
                  'image': [None]
                },
                padding_values={
                  'sentence': ' ',
                  'encoding': 0.0,
                  'fake_sentence': ' ',
                  'fake_encoding': 0.0,
                  'image': 1.0
                })

            dataset = dataset.shuffle(batch_size*3)  # shuffle the dataset

            iterator = dataset.make_initializable_iterator()
            output_types = dataset.output_types
            output_shapes = dataset.output_shapes

            return iterator, output_types, output_shapes

        def testing_tfrecord_iterator(filenames, batch_size, record_parser):
            ''' create iterator to eat tfrecord dataset

                :args
                    filenames     - a list of filenames (string)
                    batch_size    - batch size (positive int)
                    record_parser - a parser that read tfrecord
                                    and create example record (function)

                :return
                    iterator      - an Iterator providing a way
                                    to extract elements from the created dataset.
                    output_types  - the output types of the created dataset.
                    output_shapes - the output shapes of the created dataset.
            '''
            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.map(record_parser, num_parallel_calls=16)

            dataset = dataset.repeat()             # repeat dataset infinitely

            # padded into equal length in each batch
            dataset = dataset.padded_batch(
                        batch_size=batch_size,
                        padded_shapes={
                            'sentence': [None],
                            'encoding': [None],
                        },
                        padding_values={
                            'sentence': ' ',
                            'encoding': 0.0,
                        })

            dataset = dataset.shuffle(batch_size*2)  # shuffle the dataset

            iterator = dataset.make_initializable_iterator()
            output_types = dataset.output_types
            output_shapes = dataset.output_shapes

            return iterator, output_types, output_shapes


        if self.is_train:
            training_iterator, types, shapes = training_tfrecord_iterator(
                                                    self.files_tr,
                                                    batch_size=self.batch_size,
                                                    record_parser=training_parser)

            self.sess.run(training_iterator.initializer)
            next_batch = training_iterator.get_next()

        else:
            testing_iterator, types, shapes = testing_tfrecord_iterator(
                                                    TEST_RECORD,
                                                    batch_size=self.batch_size,
                                                    record_parser=testing_parser)
            next_batch = testing_iterator.get_next()

        return next_batch

    
    def build(self):
        hps = self.hps
        with tf.variable_scope('inputs'):
            next_batch = self._build_dataset()
            self.z = tf.random_normal(
                        shape=[self.batch_size, hps.z_dim],
                        mean=0.0,
                        stddev=1.0,
                        dtype=tf.float32,
                        name='noise')

            if self.is_train:
                self.cap = next_batch['encoding']
                self.cap_w = next_batch['fake_encoding']
                self.img = tf.reshape(next_batch['image'], shape=[self.batch_size, 64, 64, 3])
            else:
                self.cap = next_batch['encoding']

        with tf.variable_scope('models'):
            # x: image embedding
            # v: text embedding
            if self.is_train:
                # self.x = imageEncoder(self.img, out_dim=hps.s_dim, df_dim=hps.df_dim, is_train=True)
                # self.x_w = imageEncoder(self.img_w, out_dim=hps.s_dim, df_dim=hps.df_dim, is_train=True, reuse=True)
                self.v = SentenceEncoder(self.cap, self.batch_size, sent_dim=hps.s_dim)
                self.v_w = SentenceEncoder(self.cap_w, self.batch_size, sent_dim=hps.s_dim, reuse=True)
                # interpolation
                self.v_i = 0.85 * self.v + 0.15 * self.v_w

                self.img_fake = G(self.z, self.v, img_height=hps.imH, img_width=hps.imW,
                                  img_depth=hps.imD, gf_dim=hps.gf_dim, is_train=self.use_bn)
                self.img_fake_i = G(self.z, self.v_i, img_height=hps.imH, img_width=hps.imW,
                                    img_depth=hps.imD, gf_dim=hps.gf_dim, is_train=self.use_bn)
                
                # for visualization
                self.vis_batch = SentenceEncoder(self.vis, self.batch_size, sent_dim=hps.s_dim, reuse=True)
                sample_img = G(self.z, self.vis_batch, img_height=hps.imH, img_width=hps.imW,
                                      img_depth=hps.imD, gf_dim=hps.gf_dim, is_train=self.use_bn, reuse=True)
                self.sample_img = sample_img
                
                # real data
                self.d_real, self.logits_real = D(self.img, self.v, img_height=hps.imH, img_width=hps.imW,
                                                  img_depth=hps.imD, df_dim=hps.df_dim, is_train=self.use_bn)
                # fake data from generator
                _, self.logits_fake = D(self.img_fake, self.v, img_height=hps.imH, img_width=hps.imW,
                                        img_depth=hps.imD, df_dim=hps.df_dim, is_train=self.use_bn, reuse=True)
                _, self.logits_fake_i = D(self.img_fake_i, self.v, img_height=hps.imH, img_width=hps.imW,
                                          img_depth=hps.imD, df_dim=hps.df_dim, is_train=self.use_bn, reuse=True)
                # mismatched data
                _, self.logits_mis_v = D(self.img, self.v_w, img_height=hps.imH, img_width=hps.imW,
                                         img_depth=hps.imD, df_dim=hps.df_dim, is_train=self.use_bn, reuse=True)
            else:
                self.v = SentenceEncoder(self.cap, self.batch_size, sent_dim=hps.s_dim)
                self.img_fake = G(self.z, self.v, img_height=hps.imH, img_width=hps.imW,
                                  img_depth=hps.imD, gf_dim=hps.gf_dim, is_train=self.use_bn)

        with tf.variable_scope('losses'):
            if self.is_train:
                # encoder loss
                # alpha = 0.2 # margin alpha
                # self.enr_loss = tf.reduce_mean(alpha - cos_loss(self.x, self.v) + cos_loss(self.x, self.v_w)) + \
                #    tf.reduce_mean(alpha - cos_loss(self.x, self.v) + cos_loss(self.x_w, self.v))
                # loss of generator
                self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_fake, labels=tf.ones_like(self.logits_fake), name='d_loss_fake'))
                self.g_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_fake_i, labels=tf.ones_like(self.logits_fake_i), name='d_loss_fake_i'))
                # loss of discriminator
                self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_real, labels=tf.ones_like(self.logits_real), name='d_loss_real'))
                self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_fake, labels=tf.zeros_like(self.logits_fake), name='d_loss_fake'))
                self.d_loss_mis_v = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_mis_v, labels=tf.zeros_like(self.logits_mis_v), name='d_loss_mismatch_text'))

                self.d_loss = self.d_loss_real + 0.5 * (self.d_loss_fake + self.d_loss_mis_v)

        # the power of name scope
        # self.cnn_vars = [var for var in tf.get_collection(
        # tf.GraphKeys.GLOBAL_VARIABLES, scope='wrapper/models/ImageEncoder')]
        self.rnn_vars = [var for var in tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='wrapper/models/SentenceEncoder')]
        self.g_vars = [var for var in tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='wrapper/models/Generator')]
        self.d_vars = [var for var in tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='wrapper/models/Discriminator')]

        with tf.variable_scope('optimizers'):
            if self.is_train:
                self.lr_op = tf.Variable(hps.lr, trainable=False)
                # optimizer for textEncoder
                self.enr_opt = tf.train.AdamOptimizer(
                    self.lr_op, beta1=hps.beta1, beta2=hps.beta2)
                # def clipIfNotNone(grad, norm):
                #    if grad is None:
                #        return grad
                #    return tf.clip_by_norm(grad, norm)
                grads_and_vars = self.enr_opt.compute_gradients(
                    self.g_loss + self.d_loss, self.rnn_vars)
                clipped_grads_and_vars = [
                    (tf.clip_by_norm(gv[0], hps.clip_norm), gv[1]) for gv in grads_and_vars]
                # apply gradient and variables to optimizer
                self.enr_opt = self.enr_opt.apply_gradients(
                    clipped_grads_and_vars)

                # optimizer for generator
                self.g_opt = tf.train.AdamOptimizer(self.lr_op, beta1=hps.beta1,
                                                    beta2=hps.beta2).minimize(self.g_loss, var_list=self.g_vars)
                # optimizer for discriminator
                self.d_opt = tf.train.AdamOptimizer(self.lr_op, beta1=hps.beta1,
                                                    beta2=hps.beta2).minimize(self.d_loss, var_list=self.d_vars)

        
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(
            var_list=self.rnn_vars + self.g_vars + self.d_vars, max_to_keep=20)
    
    
    def _build_summary(self):
        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('d_loss', self.d_loss)
        tf.summary.image('sample_img', self.sample_img, max_outputs=self.batch_size)
        summary = tf.summary.merge_all()
        self.summary = summary

    
    def train(self, num_train_example, ep, ckpt_dir='/tmp/md/ted_tmp/comp4/tao_ckpt', log=True, load_idx=None):
        hps = self.hps
        num_batch_per_epoch = num_train_example // self.batch_size
        sample_size = self.batch_size
        
        # train
        if load_idx:
            self.restore(ckpt_dir, idx=load_idx)
        else:
            self.restore(ckpt_dir)
        
        for step in range(num_batch_per_epoch):
        #for step in range(100):
            step_time = time.time()
            # update D
            errD, _ = self.sess.run([self.d_loss, self.d_opt])

            # update G after D have been updated n_critic times
            if step % hps.n_critic == 0:
                errG, _ = self.sess.run([self.g_loss, self.g_opt])

            # update SentenceEncoder
            self.sess.run(self.enr_opt)
            
            if log and step % hps.n_critic == 0:
                logging.info(
                    'Epoch: %2d [%4d/%4d] time: %4.4fs, g_loss: %2.6f, d_loss: %2.6f'
                    % (ep, step, num_batch_per_epoch, time.time() - step_time, errG, errD))
        
            if step % 1000 == 0:
                summary, _ = self.sess.run([self.summary, self.sample_img])
                self.writer.add_summary(summary, global_step=ep*num_batch_per_epoch+step)
                
        self.save(ckpt_dir=ckpt_dir, idx=ep)
      
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
                self.saver.restore(self.sess, latest_ckpt)
                return True
        return False


if __name__ == '__main__':
    tracemalloc.start()
    
    tf.reset_default_graph()
    filenames = [DIR_RECORD + 'train-%d.tfrecord' % i for i in range(1, 15)]
    hps = hparas(hps_list)
    epoch = 10000
    gpu_ratio = 0.8
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_ratio)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with tf.variable_scope('wrapper'):
        model_tr = ModelWrapper(sess, hps, filenames,
                                batch_size=64, is_train=True)
        model_tr.build()
        model_tr._build_summary()
    #with tf.variable_scope('wrapper', reuse=True):
        #model_vis = ModelWrapper(
            #sess, hps, None, batch_size=10, is_train=False)
        #model_vis.build()

    num_train_example = get_num_records(filenames)
    print('num_train_example:', num_train_example)
    for ep in range(epoch):
        model_tr.train(num_train_example=num_train_example, ep=ep,
                       ckpt_dir='/tmp/md/ted_tmp/comp4/tao_ckpt')
        
        #model_vis._test(epoch=ep, save_path='result_simple/')
    sess.close()
