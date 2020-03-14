import tensorflow as tf 
import argparse
import configparser
from map_argv import *
import model_tf
import loss_tf
import dataset_tf
import utils_tf
import os
import numpy as np

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', type=str, required=True, help='path to the required configure file')
    parser.add_argument('-gpuid', type=int, default=-1, help='given gpu to train on')
    parser.add_argument('-gpu', type=bool, default=False, help='whether to use a gpu')
    parser.add_argument('-save', type=str, default='./checkpoints/try/', help='place to save')
    args = parser.parse_args()
    
    
    checkpoint_path = args.save	
    checkpoint_path = os.path.join(checkpoint_path, '{name}-{net}.pth')

    if args.gpuid >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

    config = configparser.ConfigParser()
    config.read(args.cfg)

    return args, config, checkpoint_path

def init_network(config, lr_step):
    
    posdata = {}
    negdata = {}

    for m, d in config['modals'].items():
        posdata[m] = tf.placeholder(tf.float32, shape=[None, int(d)], name=m+"_data")
        negdata[m] = tf.placeholder(tf.float32, shape=[None, int(d)], name=m+"_data")
    
    hash_sig_pos, hash_code_pos = model_tf.Encoder(config, posdata, reuse=False)
    hash_sig_neg, hash_code_neg = model_tf.Encoder(config, negdata, reuse=True)
    
    loss={}
    for m in config['modals'].keys():
        loss[m] = loss_tf.triplet_loss(config, hash_sig_pos, hash_sig_neg, m) 
    
    varlist = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    
    update = {}
    for m in config['modals'].keys():
        update[m] = loss_tf.Adamstep(config, loss[m] + model_tf.compute_regulation(float(config['train']['weight_decay']), 'encoder'), varlist, lr_step, 0.9)
        # update[m] = loss_tf.Adamstep(config, loss[m], varlist, lr_step)
    
    return posdata, negdata, loss, update, hash_code_pos, hash_code_neg

def train_encoder(sess, config, posdata, negdata, weight, loss, update, global_step, lr_step, train_list, global_steps, tag):
    train_size = int(config['dataset']['train_size'])
    index = 0
    BATCH_SIZE = int(config['dataset']['batch_size'])
    
    while index < train_size:
        feed = {}
        if index + BATCH_SIZE <= train_size:
            for qx in train_list.keys():
                qua = train_list[qx][index:index+BATCH_SIZE]
                if '_neg' in qx:
                    feed[negdata[qx[:-4]]] = np.asarray(qua)
                else:
                    feed[posdata[qx]] = np.asarray(qua)
        else:
            for qx in train_list.keys():
                qua = train_list[qx][index:]
                if '_neg' in qx:
                    feed[negdata[qx[:-4]]] = np.asarray(qua)
                else:
                    feed[posdata[qx]] = np.asarray(qua)
        

        index += BATCH_SIZE
        feed[global_step] = global_steps
        # print(feed)
        _, lossE, wei, lr = sess.run([update[tag], loss[tag], weight, lr_step], feed_dict = feed)

    print('E_Loss_%s: %.4f %.4f %.8f' % (tag, lossE + wei, lossE, lr))

if __name__ == '__main__':

    args, config, checkpoint_path = args_parse()
    
    
    global_step = tf.Variable(0, trainable=False)
    lr_step = tf.train.exponential_decay(float(config['train']['lr']), global_step, int(config['train']['step']), float(config['train']['gamma']), staircase=True)
    
    posdata, negdata, loss, update, hash_code_pos, hash_code_neg = init_network(config, lr_step)
    weight = model_tf.compute_regulation(float(config['train']['weight_decay']), 'encoder')
    test_feature,database_feature,test_label,database_label,train_feature,knn_idx = dataset_tf.xmedia(config)

    cf = tf.ConfigProto(allow_soft_placement=True)
    cf.gpu_options.allow_growth = True
    sess = tf.Session(config=cf)
    init = tf.global_variables_initializer()
    sess.run(init)

    varlist = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    saver = tf.train.Saver(var_list=varlist)

    map_best_val = 0.0

    for epoch in range(int(config['train']['epoch'])):
        print("EPOCH ", epoch)
        for d_epoch in range(int(config['train']['d_epoch'])):
            print('d_epoch: ', d_epoch)
            if d_epoch == 0:
                print("negerating samples")
                train_list = {}
                for m in config['modals'].keys():
                    train_list[m] = utils_tf.generate_samples(config, m, train_feature, knn_idx)

            for m in config['modals'].keys():
                train_encoder(sess, config, posdata, negdata, weight, loss, update, global_step, 
                    lr_step, train_list[m], epoch*int(config['train']['d_epoch']) + d_epoch, m)

        if (epoch + 1) % int(config['train']['print']) == 0:
            test_map = MAP_ARGV(sess, config, posdata, hash_code_pos, test_feature, database_feature, test_label, database_label)
            print(test_map)
            if test_map > map_best_val:
                map_best_val = test_map
                saver.save(sess, checkpoint_path.format(net='encoder', name='best'))

        if (epoch + 1) % int(config['train']['save_epoch']) == 0:
            saver.save(sess, checkpoint_path.format(net='encoder', name=str(epoch+1)))
    
    sess.close()
