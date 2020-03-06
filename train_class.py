import configparser
import argparse
import os
import model

import torch
import torch.optim as optim
import torch.nn as nn
import dataset
import numpy as np

from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', type=str, required=True, help='path to the required configure file')
    parser.add_argument('-gpuid', type=str, default='-1', help='given gpu to train on')
    parser.add_argument('-gpu', type=bool, default=True, help='whether to use a gpu')
    parser.add_argument('-save', type=str, default='./checkpoints/try/', help='place to save')
    args = parser.parse_args()

    checkpoint_path = args.save

    if args.gpuid != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    config = configparser.ConfigParser()
    config.read(args.cfg)

    encoder = model.Encoder_class(config)
    # generator = model.Generator(config) 
    # discriminator = model.Discriminator(config)
    if args.gpu:
        encoder = encoder.cuda()
        # generator = generator.cuda()
        # discriminator = discriminator.cuda()
    encoder = encoder.train()
    # generator = generator.train()
    # discriminator = discriminator.train()

    optimizer = optim.Adam(encoder.parameters(), lr=float(config['train']['lr'])) 
    

    scheduler = optim.lr_scheduler.StepLR(optimizer, int(config['train']['step_size']), 
                                gamma=float(config['train']['gamma']))

    trainset = dataset.xmedia_class(config)

    rounds = trainset.trainsize // trainset.batch_size

    cri = nn.CrossEntropyLoss()
    activ = nn.LeakyReLU(negative_slope=0.1)

    checkpoint_path = os.path.join(checkpoint_path, '{epoch}-{net}.pth')

    L_DF = float(config['train']['l_df'])
    L_DT = float(config['train']['l_dt'])
    L_DI = float(config['train']['l_di'])

    L_GF = float(config['train']['l_gf'])
    # L_GT = float(config['train']['l_gt'])
    L_GI = float(config['train']['l_gi'])
    
    L_ED = float(config['train']['l_ed'])
    L_EDI = float(config['train']['l_edi'])
    L_EQ = float(config['train']['l_eq'])
    L_EQI = float(config['train']['l_eqi'])
    beta = float(config['train']['beta'])


    for i in range(int(config['train']['epoch'])):
        
       
        scheduler.step()
        # for b in scheduler_G.values():
        #     b.step()
        # for c in scheduler_D.values():
        #     c.step()
        trainset.reset()
        correct = {}
        loss_cum = {}
        for m in config['modals'].keys():
            correct[m] = 0
            loss_cum[m] = 0.0
        
        for j in range(rounds):
            d, l = trainset.sample(args.gpu)
            out = encoder(d)

            
            loss = 0
            for m, v in out.items():
                _, preds = v.max(1)
                # print('l', l[m])
                lab = torch.topk(l[m], 1)[1].squeeze(1)
                # print('lab', lab)
                # print('preds', preds)
                correct[m] += preds.eq(lab).sum()
                L = cri(v, lab.cuda())
                loss_cum[m] += L
                loss += L
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            

            if j % int(config['train']['print']) == 0:
                print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\t {modal} Loss_E: {:0.4f}\t'.format(
                    loss,
                    modal = 'all',
                    epoch=i,
                    trained_samples=int(config['dataset']['batch_size'])*j,
                    total_samples=int(config['dataset']['train_size'])
                ))


            trainset.update()
        
        for m in config['modals'].keys():
            print('--Training Results: {epoch} [{trained_samples}/{total_samples}]\t {modal} Loss_E: {:0.4f}\t'.format(
                loss_cum[m],
                modal = m,
                epoch=i,
                trained_samples=int(correct[m]),
                total_samples=int(config['dataset']['train_size'])
            ))


        trainset.reset()
        correct = {}
        loss_cum = {}
        for m in config['modals'].keys():
            correct[m] = 0
            loss_cum[m] = 0.0
        
        for j in range(1000//20):
            d, l = trainset.sampletest(args.gpu)
            out = encoder(d)

            
            loss = 0
            for m, v in out.items():
                _, preds = v.max(1)
                # print('l', l[m])
                lab = torch.topk(l[m], 1)[1].squeeze(1)
                # print('lab', lab)
                # print('preds', preds)
                correct[m] += preds.eq(lab).sum()
                L = cri(v, lab.cuda())
                loss_cum[m] += L
                loss += L
                

            trainset.cnt += 20
        
        for m in config['modals'].keys():
            print('TEST Epoch: {epoch} [{trained_samples}/{total_samples}]\t {modal} Loss_E: {:0.4f}\t'.format(
                loss_cum[m],
                modal = m,
                epoch=i,
                trained_samples=int(correct[m]),
                total_samples=1000
            ))
        L_DF *= L_DI
        L_DT *= L_DI
        L_GF *= L_GI

        L_ED *= L_EDI
        L_EQ *= L_EQI
        if i % int(config['train']['save_epoch']) == 0:
            torch.save(encoder.state_dict(), checkpoint_path.format(net='encoder', epoch=i))
            # torch.save(generator.state_dict(), checkpoint_path.format(net='generator', epoch=i))
            # torch.save(discriminator.state_dict(), checkpoint_path.format(net='discriminator',epoch=i))

                
                

                    









                



                
