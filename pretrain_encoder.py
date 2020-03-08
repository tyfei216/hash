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
    
    encoder = model_tf.Encoder(config)
    loss={}
    for m in config['modals'].keys():
        loss[m] = loss_tf.triplet_loss(config, encoder.hash_sig, encoder.hash_sig_neg, m) 
    varlist = [var for var in tf.trainable_variables()]
    
    update = {}
    for m in config['modals'].keys():
        update[m] = loss_tf.Adamstep(config, loss[m] + encoder.compute_regulation(float(config['train']['weight_decay'])), varlist, lr_step)
    
    return encoder, loss, update

def train_encoder(sess, config, encoder, loss, update, global_step, lr_step, train_list, global_steps, tag):
    train_size = int(config['dataset']['train_size'])
    index = 0
    BATCH_SIZE = int(config['dataset']['batch_size'])
    
    while index < train_size:
        feed = {}
        if index + BATCH_SIZE <= train_size:
            for qx in train_list.keys():
                qua = train_list[qx][index:index+BATCH_SIZE]
                if '_neg' in qx:
                    feed[encoder.negdata[qx[:-4]]] = np.asarray(qua)
                else:
                    feed[encoder.posdata[qx]] = np.asarray(qua)
        else:
            for qx in train_list.keys():
                qua = train_list[qx][index:]
                if '_neg' in qx:
                    feed[encoder.negdata[qx[:-4]]] = np.asarray(qua)
                else:
                    feed[encoder.posdata[qx]] = np.asarray(qua)
        

        index += BATCH_SIZE
        feed[global_step] = global_steps
        # print(feed)
        _, lossE, weight, lr = sess.run([update[tag], loss[tag], encoder.regulation, lr_step], feed_dict = feed)

    print('E_Loss_%s: %.4f %.4f %.8f' % (tag, lossE + weight, lossE, lr))

if __name__ == '__main__':

    args, config, checkpoint_path = args_parse()
    
    
    global_step = tf.Variable(0, trainable=False)
    lr_step = tf.train.exponential_decay(float(config['train']['lr']), global_step, int(config['train']['step']), float(config['train']['gamma']), staircase=True)
    
    encoder, loss, update = init_network(config, lr_step)
    test_feature,database_feature,test_label,database_label,train_feature,knn_idx = dataset_tf.xmedia(config)

    cf = tf.ConfigProto(allow_soft_placement=True)
    cf.gpu_options.allow_growth = True
    sess = tf.Session(config=cf)
    init = tf.global_variables_initializer()
    sess.run(init)

    varlist = [var for var in tf.trainable_variables()]
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
                train_encoder(sess, config, encoder, loss, update, global_step, 
                    lr_step, train_list[m], epoch*int(config['train']['d_epoch']) + d_epoch, m)

        if (epoch + 1) % int(config['train']['print']) == 0:
            test_map = MAP_ARGV(sess, config, encoder, test_feature, database_feature, test_label, database_label)
            if test_map > map_best_val:
                map_best_val = test_map
                saver.save(sess, checkpoint_path.format(net='encoder', name='best'))

        if (epoch + 1) % int(config['train']['save_epoch']) == 0:
            saver.save(sess, checkpoint_path.format(net='encoder', name=str(epoch+1)))
    
    sess.close()
