# ---------------
# Author: Tu, Tao
# ---------------

import Layer
import tensorflow as tf

def textEncoder(txt, vocab_size, batch_size, with_matrix=False, reuse=tf.AUTO_REUSE, pad_token=0, bidirectional=False, word_dim=256, sent_dim=128):
    with tf.variable_scope('TextEncoder', reuse=reuse):
        if with_matrix:
            w_embed_seq, w_matrix = _word_embedding(txt, word_dim, with_matrix=True)
        else:
            w_embed_seq = _word_embedding(txt, word_dim, vocab_size, with_matrix=False)
        w_seq_len = Layer.retrieve_seq_length(txt, pad_val=pad_token)
        s_embed = _sent_embedding(w_embed_seq, sent_dim, batch_size, w_seq_len, bidirectional)
        # according to the phi function of paper "Generative Adversarial Text to Image Synthesis"
        code = Layer.dense(s_embed[:, -1, :], output_dim=sent_dim, act=tf.nn.leaky_relu, name='code')
    return (code, w_matrix) if with_matrix else code

def _word_embedding(txt, w_dim, vocab_size, with_matrix):
    with tf.variable_scope('word_embedding'):
        embed_matrix = tf.get_variable('word_embed_matrix', 
                                            shape=(vocab_size, w_dim), 
                                            initializer=tf.random_normal_initializer(stddev=0.02), 
                                            dtype=tf.float32)
        w_embeds = tf.nn.embedding_lookup(embed_matrix, txt)
    return (w_embeds, embed_matrix) if with_matrix else w_embeds

def _sent_embedding(w_embed_seq, s_dim, batch_size, w_seq_len, bidirectional=False):
    with tf.variable_scope('sent_embedding') as scope:
        if bidirectional:
            cell_fw = tf.contrib.rnn.BasicLSTMCell(s_dim//2)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(s_dim//2)
            init_state_fw = cell_fw.zero_state(batch_size, dtype=tf.float32)
            init_state_bw = cell_bw.zero_state(batch_size, dtype=tf.float32)
            outputs_tuple, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=w_embed_seq, 
                sequence_length=w_seq_len,
                initial_state_fw=init_state_fw, 
                initial_state_bw=init_state_bw,
                dtype=tf.float32, 
                time_major=False, 
                scope=scope)
            output_fw, output_bw = outputs_tuple
            outputs = tf.concat([output_fw, output_bw], axis=2)
        else:
            cell_fw = tf.contrib.rnn.BasicLSTMCell(s_dim)
            init_state_fw = cell_fw.zero_state(batch_size, dtype=tf.float32)
            outputs, _ = tf.nn.dynamic_rnn(
                cell=cell_fw,
                inputs=w_embed_seq,
                sequence_length=w_seq_len,
                initial_state=init_state_fw,
                dtype=np.float32,
                time_major=False,
                scope=scope)
    return outputs
