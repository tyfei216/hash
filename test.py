import torch
import numpy as np
import configparser
import argparse

import model
import dataset
import os

from utils import *

def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)

def average_precision(r):
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def count_map(test,data,test_lab,data_lab):
    qlen = len(test)
    dlen = len(data)

    dist = np.zeros(dlen)
    res = np.zeros(qlen)
    
    for i in range(qlen):
		#print i
        for j in range(dlen):
            # print(i)
			# pdb.set_trace()
            # print()
            dist[j] = sum(test[i]^data[j])
        idx = np.argsort(dist)
        ton = 0
        for k in range(dlen):
            if sum(data_lab[idx[k]]^test_lab[i])==0:
                # dist[j] = sum(test[i]^data[j])
                # print(k)
                ton = ton+1
                res[i] += ton/(k+1.0)   
        res[i] = res[i]/ton
    return np.mean(res)

def test(encoder, gpu, testset, label):
    encoder.eval()
    test = totensor(testset.test_feature)
    if gpu:
        for m, i in test.items():
            test[m] = test[m].cuda()
    test = encoder(test)

    database = totensor(testset.database_feature)
    if gpu:
        for m, i in test.items():
            database[m] = database[m].cuda()
    database = encoder(database)

    test = toBinary(test, gpu)
    database = toBinary(database, gpu)

    # test = torch.cat(test.values(), 0).numpy()
    database_f = []
    database_label = []

    database_l = totensor(testset.database_label)

    for m in database.keys():
        database_f.append(database[m])
        database_label.append(database_l[m])

    database_f = torch.cat(database_f, 0).cpu().numpy()
    database_label = torch.cat(database_label, 0).cpu().numpy()

    print(database_label.shape)
    print(database_f.shape)

    res = {}
    for m, d in test.items():
        print('testnum', test[m].size())
        res[m] = count_map(test[m].to(torch.int32).cpu().numpy(), database_f.astype(np.int32), 
        np.array(testset.test_label[m]).astype('int32'), database_label.astype(np.int32))
        # print(count_map(database_label.astype(np.int32),database_label.astype(np.int32),database_label.astype(np.int32),database_label.astype(np.int32)))

    print(res)
    fp = open("./result.txt","a")
    fp.write(label+str(res)+"\n")
    fp.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', type=str, required=True, help='path to the required configure file')
    parser.add_argument('-gpuid', type=str, default=-1, help='given gpu to train on')
    parser.add_argument('-gpu', type=bool, default=False, help='whether to use a gpu')
    parser.add_argument('-save', type=str, default='./checkpoints/try/', help='place to save')
    parser.add_argument('-weights', type=str, required=True, help='the weights')
    args = parser.parse_args()
    
    if args.gpuid != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    config = configparser.ConfigParser()
    config.read(args.cfg)

    encoder = model.Encoder(config)
    # generator = model.Generator(config) 
    # discriminator = model.Discriminator(config)
    if args.gpu:
        encoder = encoder.cuda()
        # generator = generator.cuda()
        # discriminator = discriminator.cuda()

    encoder.load_state_dict(torch.load(args.weights), False)
    # generator = generator.load_state_dict(args.weights+'-generator.pth', args.gpu)
    # discriminator = discriminator.load_state_dict(args.weights+'-discriminator.pth', args.gpu)

    testset = dataset.xmedia_test(config)
    test = totensor(testset.test_feature)
    test = encoder(test)
    database = encoder(totensor(testset.database_feature))

    

    test = toBinary(test, args.gpu)
    database = toBinary(database, args.gpu)

    # test = torch.cat(test.values(), 0).numpy()
    database_f = torch.cat(tuple(database.values()), 0).numpy()
    database_label = torch.cat(tuple(totensor(testset.database_label).values()), 0).numpy()

    print(database_label.size)
    print(database_f.size)

    res = {}
    for m, d in test.items():
        print('testnum', test[m].size())
        print(test[m].to(torch.int32).numpy()[0])
        print(database_f.astype(np.int32)[0])
        print(database_label.astype(np.int32)[0])
        print(testset.test_label[m][0])
        res[m] = count_map(test[m].to(torch.int32).numpy(), database_f.astype(np.int32), 
        np.array(testset.test_label[m]).astype('int32'), database_label.astype(np.int32))
        # print(count_map(database_label.astype(np.int32),database_label.astype(np.int32),database_label.astype(np.int32),database_label.astype(np.int32)))

    print(res)
    
