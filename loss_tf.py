import tensorflow as tf
import numpy as np 

def compute_distance(ot, sig_dict):
    result = 0.0
    for i in sig_dict.keys():
        result += tf.reduce_sum(tf.square(ot - sig_dict[i]), 1)
    return result

def quantization_loss(config, hash_sig, hash_code):
    result = 0.0
    for m in config['modals'].keys():
        result += tf.reduce_sum(tf.square(hash_sig[m] - hash_code[m]), 1)
    return result * float(config['train']['quantization'])

def triplet_loss(config, hash_sig, hash_sig_neg, m):
    
    pos_distance = compute_distance(hash_sig[m], hash_sig)
    neg_distance = compute_distance(hash_sig_neg[m], hash_sig_neg)

    return tf.reduce_mean(tf.maximum(0.0, float(config['train']['beta']) + pos_distance - neg_distance))

def Adamstep(config, loss, params, lr_step):
    
    optimizer = tf.train.AdamOptimizer(lr_step)
    updates = optimizer.minimize(loss, var_list = params)

    return updates

if __name__ == '__main__':
    config = {'train':{'lr':-1, 'momemtum':0.75}}
