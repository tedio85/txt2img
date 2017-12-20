import os
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import time

DIR_DICT = './dictionary/'
DIR_DATA = './dataset/'


def _int64_feature(value, pass_list=False):
    if not pass_list:
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(X, name):
    imgs = X['image']
    caps = X['caption']

    filename = os.path.join(DIR_DATA, name + '.tfrecords')
    with tf.python_io.TFRecordWriter(filename) as writer:
        for idx in range(len(X)):
            image_raw = imgs[idx].tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image': _bytes_feature(tf.compat.as_bytes(image_raw)),
                        'caption': _int64_feature(caps[idx], pass_list=True)
                    }))
            writer.write(example.SerializeToString())

if __name__ == '__main__':
    df = pd.read_pickle(DIR_DATA + 'text2ImgData.pkl')
    vocab = np.load(DIR_DICT + 'vocab.npy')
    enc_map = dict(np.load(DIR_DICT + 'word2Id.npy'))
    dec_map = dict(np.load(DIR_DICT + 'id2Word.npy'))

    examples = []
    st_time = time.time()
    cnt = 0
    for _, item in df.iterrows():
        caps = item['Captions']
        im_path = item['ImagePath']
        im = Image.open(DIR_DATA + im_path).resize((500, 500))
        # dtype: uint8
        im = np.asarray(im)
        for cap in caps:
            cap = [int(x) for x in cap]
            examples.append([im, cap])
            cnt += 1
            if cnt % 10000 == 0:
                print('Finished: %d, time: %4.4fs' %
                      (cnt, time.time() - st_time))

    X = pd.DataFrame(examples, columns=['image', 'caption'])

    convert_to(X, 'img_cap_pairs')
    print('Create %s, total Time: %4.4fs' %
          (DIR_DATA + 'img_cap_pairs', time.time() - st_time))
