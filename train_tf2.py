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
    parser.add_argument('-weights', type=str, required=True, help='the pretrained model')
    parser.add_argument('-label', type=str, default='default', help='label for the model')
    args = parser.parse_args()
    
    
    checkpoint_path = args.save	
    checkpoint_path = os.path.join(checkpoint_path, '{name}-{net}.pth')

    if args.gpuid >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

    config = configparser.ConfigParser()
    config.read(args.cfg)

    return args, config, checkpoint_path

def build_net(config):
    
    posdata = {}
    negdata = {}

    for m, d in config['modals'].items():
        posdata[m] = tf.placeholder(tf.float32, shape=[None, int(d)], name=m+"_data")
        negdata[m] = tf.placeholder(tf.float32, shape=[None, int(d)], name=m+"_data")
        posdata[m+"1"] = tf.placeholder(tf.float32, shape=[None, int(d)], name=m+"_data1")
        #negdata[m+"1"] = tf.placeholder(tf.float32, shape=[None, int(d)], name=m+"_data")
        posdata[m+"2"] = tf.placeholder(tf.float32, shape=[None, int(d)], name=m+"_data2")
        #negdata[m+"2"] = tf.placeholder(tf.float32, shape=[None, int(d)], name=m+"_data")

    hashcode_pos = tf.placeholder(tf.float32, shape=[None, int(config['parameters']['dim_out'])])
    hashcode_neg = tf.placeholder(tf.float32, shape=[None, int(config['parameters']['dim_out'])])

    inputs = {}
    inputs['pos'] = posdata
    inputs['neg'] = negdata
    inputs['poscode'] = hashcode_pos
    inputs['negcode'] = hashcode_neg

    hash_sig_pos, hash_code_pos = model_tf.Encoder_new(config, posdata, reuse=False)
    print('first')
    hash_sig_neg, hash_code_neg = model_tf.Encoder_new(config, negdata, reuse=True)
    print('second')

    encoder = {}
    encoder['sig_pos'] = hash_sig_pos
    encoder['code_pos'] = hash_code_pos
    encoder['sig_neg'] = hash_sig_neg
    encoder['code_neg'] = hash_code_neg

    z = tf.placeholder(tf.float32, shape=[None, int(config['parameters']['dim_ran'])])

    genfpos = model_tf.Generator(config, hashcode_neg, z, reuse=False)

    generator = {}
    generator['neg'] = hash_code_neg
    generator['negf'] = genfpos
    generator['z'] = z

    slt, hlt = model_tf.Discriminator_new(config, posdata, reuse=False)
    slf, hlf = model_tf.Discriminator_new(config, genfpos, reuse=True)

    discriminator = {}
    discriminator['slt'] = slt
    discriminator['hlt'] = hlt
    discriminator['slf'] = slf
    discriminator['hlf'] = hlf

    return inputs, encoder, generator, discriminator


'''
def init_network(config, lr_step):
    
    posdata = {}
    negdata = {}

    for m, d in config['modals'].items():
        posdata[m] = tf.placeholder(tf.float32, shape=[None, int(d)], name=m+"_data")
        negdata[m] = tf.placeholder(tf.float32, shape=[None, int(d)], name=m+"_data")
    
    hash_sig_pos, hash_code_pos = model_tf.Encoder_new(config, posdata, reuse=False)
    hash_sig_neg, hash_code_neg = model_tf.Encoder_new(config, negdata, reuse=True)
    
    loss={}
    for m in config['modals'].keys():
        loss[m] = loss_tf.triplet_loss(config, hash_sig_pos, hash_sig_neg, m) 
    
    varlist = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    
    update = {}
    for m in config['modals'].keys():
        update[m] = loss_tf.Adamstep(config, loss[m] + model_tf.compute_regulation(float(config['train']['weight_decay']), 'encoder'), varlist, lr_step)
        # update[m] = loss_tf.Adamstep(config, loss[m], varlist, lr_step)
    
    return posdata, negdata, loss, update, hash_code_pos, hash_code_neg
'''
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
    
    inputs, encoder, generator, discriminator = build_net(config)

    global_step = tf.Variable(0, trainable=False)
    lr_step = tf.train.exponential_decay(float(config['train']['lr']), global_step, int(config['train']['step']), float(config['train']['gamma']), staircase=True)
    lr_stepE = tf.train.exponential_decay(float(config['train']['lr']), global_step+60, int(config['train']['step']), float(config['train']['gamma']), staircase=True)

    lossE = loss_tf.loss_dis(encoder['sig_pos'], inputs['poscode']) + \
        loss_tf.loss_dis(encoder['sig_neg'], inputs['negcode'])
    weightE = model_tf.compute_regulation(float(config['train']['weight_decay']), 'encoder')
    varlistE = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    updateE = loss_tf.Adamstep(config, lossE + weightE, varlistE, lr_stepE, 0.5)

    d_loss, g_loss = loss_tf.GANloss(config, inputs['poscode'], inputs['negcode'], discriminator['slt'], discriminator['hlt'], 
    discriminator['slf'], discriminator['hlf'])

    varlistG = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    updateG = loss_tf.Adamstep(config, g_loss, varlistG, lr_step, 0.5)

    varlistD = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    updateD = loss_tf.Adamstep(config, d_loss, varlistD, lr_step, 0.5)
    
    
    test_feature,database_feature,test_label,database_label,train_feature,knn_idx = dataset_tf.xmedia(config)

    cf = tf.ConfigProto(allow_soft_placement=True)
    cf.gpu_options.allow_growth = True
    sess = tf.Session(config=cf)
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(var_list=varlistE)
    saver.restore(sess, args.weights)

    map_best_val = 0.0
    train_size = int(config['dataset']['train_size'])

    batch_size = int(config['dataset']['batch_size'])
    for epoch in range(int(config['train']['warmup'])):
        print("EPOCH ", epoch)
        for d_epoch in range(int(config['train']['d_epoch'])):
            print('d_epoch: ', d_epoch)
            if d_epoch == 0:
                print("generating samples")
                train_list = {}
                for m in config['modals'].keys():
                    train_list[m] = utils_tf.generate_samples2(config, m, train_feature, knn_idx)
                print("finished generating samples")

            for m in config['modals'].keys():
                index = 0
                while index < train_size:
                    feed = {}
                    if index + batch_size <= train_size:
                        for qx in train_list[m].keys():
                            qua = train_list[m][qx][index:index+batch_size]
                            feed[inputs['pos'][qx]] = np.asarray(qua)
                    index += batch_size
                    # print(index)
                    feed[global_step] = epoch*int(config['train']['d_epoch']) + d_epoch
                    data = sess.run(encoder['code_pos'], feed_dict=feed)
                    data = utils_tf.get_hash_code(data)
                    # print(data)
                    random_z = np.random.uniform(-1, 1, size=(batch_size, int(config['parameters']['dim_ran']))).astype(np.float32)
                                
                    feed[generator['z']] = random_z
                    feed[inputs['poscode']] = data
                    feed[inputs['negcode']] = data
                    _up, lossd1 = sess.run([updateD, d_loss], feed_dict = feed)
                    _up, lossg = sess.run([updateG, g_loss], feed_dict = feed)
                    _up, lossg1 = sess.run([updateG, g_loss], feed_dict = feed)
                    rand = np.random.uniform(0, 1.999, size=(batch_size, int(config['parameters']['dim_out']))).astype(np.int32).astype(np.float32)
                    feed[inputs['negcode']] = rand
                    _up, lossd2 = sess.run([updateD, d_loss], feed_dict = feed)
                    _up, lossg = sess.run([updateG, g_loss], feed_dict = feed)
                    _up, lossg2, lr = sess.run([updateG, g_loss, lr_step], feed_dict = feed)
                
                print('lossg1', lossg1, 'lossd1', lossd1, 'lr', lr)
                print('lossg2', lossg2, 'lossd2', lossd2, 'lr', lr)
    print('finish warm up')
    test_map = MAP_ARGV(sess, config, inputs['neg'], encoder['code_neg'], test_feature, database_feature, test_label, database_label, args.label)
    print(test_map)
    
    for epoch in range(int(config['train']['epoch'])):
        print("EPOCH", epoch)
        for d_epoch in range(int(config['train']['d_epoch'])):
            print('d_epoch ', d_epoch)
            if d_epoch == 0:
                print("generating samples")
                train_list = {}
                for m in config['modals'].keys():
                    train_list[m] = utils_tf.generate_samples2(config, m, train_feature, knn_idx)
                print("finished generating samples")
            
            for m in config['modals'].keys():
                index = 0
                while index < train_size:
                    feed = {}
                    if index + batch_size <= train_size:
                        for qx in train_list[m].keys():
                            qua = train_list[m][qx][index:index+batch_size]
                            feed[inputs['pos'][qx]] = np.asarray(qua)
                    index += batch_size
                    feed[global_step] = epoch*int(config['train']['d_epoch']) + d_epoch
                    data = sess.run(encoder['code_pos'], feed_dict=feed)
                    data = utils_tf.get_hash_code(data)
                    random_z = np.random.uniform(-1, 1, size=(batch_size, int(config['parameters']['dim_ran']))).astype(np.float32)
                    feed[generator['z']] = random_z
                    feed[inputs['poscode']] = data
                    feed[inputs['negcode']] = data
                    _up, lossd1 = sess.run([updateD, d_loss], feed_dict = feed)
                    _up, lossg = sess.run([updateG, g_loss], feed_dict = feed)
                    _up, lossg1 = sess.run([updateG, g_loss], feed_dict = feed)
                    rand = np.random.uniform(0, 1.99999, size=(batch_size, int(config['parameters']['dim_out']))).astype(np.int32).astype(np.float32)
                    feed[inputs['negcode']] = rand
                    _up, lossd2 = sess.run([updateD, d_loss], feed_dict = feed)
                    _up, lossg = sess.run([updateG, g_loss], feed_dict = feed)
                    _up, lossg2 = sess.run([updateG, g_loss], feed_dict = feed)

                    feed[inputs['poscode']] = data
                    feed[inputs['negcode']] = rand
                    genf = sess.run(generator['negf'], feed_dict=feed)
                    for mm in config['modals'].keys():
                        feed[inputs['neg'][mm]] = genf[mm]
                    _up, losse = sess.run([updateE, lossE], feed_dict = feed)

                print('lossg1', lossg1, 'lossd1', lossd1, 'losse', losse)
                print('lossg2', lossg2, 'lossd2', lossd2, 'losse', losse)

        if (epoch + 1) % 1 == 0:#int(config['train']['print']) == 0:
            test_map = MAP_ARGV(sess, config, inputs['neg'], encoder['code_neg'], test_feature, database_feature, test_label, database_label, args.label)
            print(test_map)
            if test_map > map_best_val:
                map_best_val = test_map
                saver.save(sess, checkpoint_path.format(net='all', name=args.label))

        if (epoch + 1) % int(config['train']['save_epoch']) == 0:
            saver.save(sess, checkpoint_path.format(net='all', name=str(epoch+1)))
    
    sess.close()
