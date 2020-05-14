import tensorflow as tf
import numpy as np 

def compute_distance(ot, sig_dict):
    result = 0.0
    for i in sig_dict.keys():
        result += tf.reduce_sum(tf.square(ot - sig_dict[i]), 1)
    return result

def loss_dis(sig_dict, value):
    result = 0.0
    for i in sig_dict.values():
        result += tf.reduce_sum(tf.square(i-value))
    return result

def quantization_loss(config, hash_sig, hash_code):
    result = 0.0
    for m in config['modals'].keys():
        result += tf.reduce_sum(tf.square(hash_sig[m] - hash_code[m]))
    return result * float(config['train']['quantization'])

def triplet_loss(config, hash_sig, hash_sig_neg, m):
    
    pos_distance = compute_distance(hash_sig[m], hash_sig)
    neg_distance = compute_distance(hash_sig_neg[m], hash_sig_neg)

    return tf.reduce_mean(tf.maximum(0.0, float(config['train']['beta']) + pos_distance - neg_distance))

def Adamstep(config, loss, params, lr_step, beta = 0.9):
    
    optimizer = tf.train.AdamOptimizer(lr_step, beta1=beta)
    updates = optimizer.minimize(loss, var_list = params)

    return updates

def GANloss(config, reallabel, fakelabel, source_logits_real, hash_logits_real, source_logits_fake,
         hash_logits_fake):

    g_loss = 0.0
    d_loss = 0.0
    for m in source_logits_real.keys():
        ones = tf.ones_like(source_logits_real[m])
        zeros = tf.zeros_like(source_logits_real[m])

    for m in source_logits_real.keys():
        source_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=source_logits_real[m], labels=ones))

        source_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=source_logits_fake[m[0:3]], labels=zeros))

        hash_loss_gen = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=source_logits_fake[m[0:3]], labels=ones))

        hash_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=hash_logits_real[m],
                                                    labels=reallabel))
        hash_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=hash_logits_fake[m[0:3]],
                                                    labels=fakelabel))

        d_loss += float(config['parameters']['dim_out'])*(source_loss_real + source_loss_fake) + hash_loss_real + hash_loss_fake

        g_loss += float(config['parameters']['dim_out'])*hash_loss_gen + hash_loss_real + hash_loss_fake

    return d_loss, g_loss

if __name__ == '__main__':
    config = {'train':{'lr':-1, 'momemtum':0.75}}
