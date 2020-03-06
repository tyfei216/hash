import configparser
import argparse
import os
import model

import torch
import torch.optim as optim
import torch.nn as nn
import test
import dataset

from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', type=str, required=True, help='path to the required configure file')
    parser.add_argument('-gpuid', type=str, default='-1', help='given gpu to train on')
    parser.add_argument('-gpu', type=bool, default=True, help='whether to use a gpu')
    parser.add_argument('-label', type=str, default="default", help='label for result')
    parser.add_argument('-save', type=str, default='./checkpoints/try/', help='place to save')
    args = parser.parse_args()

    checkpoint_path = args.save

    if args.gpuid != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    config = configparser.ConfigParser()
    config.read(args.cfg)

    encoder = model.Encoder(config)
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
    optimizers_E = {}
    # optimizers_G = {}
    # optimizers_D = {}
    params = {}
    for m in config['modals']:
        params[m] = [d for n, d in encoder.named_parameters() if m in str(n)]
        optimizers_E[m] = optim.Adam(params[m], lr=float(config['train']['lr']))
        # params = [d for n, d in generator.named_parameters() if m in str(n)] 
        # optimizers_G[m] = optim.Adam(params, lr=float(config['train']['lr']))
        # params = [d for n, d in discriminator.named_parameters() if m in str(n)]
        # optimizers_D[m] = optim.Adam(params, lr=float(config['train']['lr']))

    scheduler = optim.lr_scheduler.StepLR(optimizer, int(config['train']['step_size']), 
                                gamma=float(config['train']['gamma']))
    scheduler_E = {}
    # scheduler_G = {}
    # scheduler_D = {}
    for m in config['modals']:
        scheduler_E[m] = optim.lr_scheduler.StepLR(optimizers_E[m], int(config['train']['step_size']), 
                                gamma=float(config['train']['gamma']))
        # scheduler_G[m] = optim.lr_scheduler.StepLR(optimizers_G[m], int(config['train']['step_size']), 
        #                         gamma=float(config['train']['gamma']))                                
        # scheduler_D[m] = optim.lr_scheduler.StepLR(optimizers_D[m], int(config['train']['step_size']), 
        #                         gamma=float(config['train']['gamma']))

    trainset = dataset.xmedia(config)
    testset = dataset.xmedia_test(config)

    rounds = trainset.trainsize // trainset.batch_size

    cri = nn.MSELoss()
    activ = nn.ReLU()

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
        # for a in scheduler_E.values():
        #    a.step()
        # for b in scheduler_G.values():
        #     b.step()
        # for c in scheduler_D.values():
        #     c.step()
        trainset.reset()

        for times in range(int(config['train']['d_epoch'])):
            encoder.train()
            trainset.cnt = 0

            for j in range(rounds):
                
                for m in config['modals'].keys():

                    pos, neg = trainset.sample(m, args.gpu)
                    outputpos = encoder(pos)
                    outputneg = encoder(neg)

                    hashcodepos = toBinary(outputpos, args.gpu)
                    hashcodeneg = toBinary(outputneg, args.gpu)

                    # neg = toneg(hashcodepos)
                    # pos_sample = generator(hashcodepos, args.gpu)
                    # neg_sample = generator(hashcodeneg, args.gpu)

                    dist = distance(outputpos, m) - distance(outputneg, m)
                    dist = activ(dist+beta)
                    
                    # dist = nn.ReLU(dist + beta)

                    # p = encoder(todetach(pos_sample))  
                    # n = encoder(todetach(neg_sample))
                    # R = 0.0
                    # for p in params[m]:
                    #     R += (p*p).sum()*0.0005
                    q = 0.0
                    for mm in config['modals'].keys():
                        q += ((outputpos[mm]-hashcodepos[mm])**2 + (outputneg[mm]-hashcodeneg[mm])**2).mean()
                    r = 0.0
                    for t in encoder.parameters():
                        r += (t*t).sum()
                    loss_E = dist.mean() + r * float(config['train']['weight_decay'])

                    if i > 5:
                        q = 0.0
                        for mm in config['modals'].keys():
                            q += ((outputpos[mm]-hashcodepos[mm])**2 + (outputneg[mm]-hashcodeneg[mm])**2).mean()
                        loss_E += q * L_ED
                    # L_ED * ((p[m]-n[m])**2).mean() + \
                        
                    optimizer.zero_grad()
                    loss_E.backward()
                    optimizer.step()

                    if j % int(config['train']['print']) == 0:
                        print('Training Epoch: {epoch} {round} [{trained_samples}/{total_samples}]\t {modal} Loss_E: {:0.4f} {:0.4f}\t'.format(
                            loss_E,
                            r,
                            modal = m,
                            epoch=i,
                            round=times,
                            trained_samples=int(config['dataset']['batch_size'])*j,
                            total_samples=int(config['dataset']['train_size'])
                        ))
            
                trainset.update()
            
            if times % int(config['train']['save_epoch']) == 0:
                torch.save(encoder.state_dict(), checkpoint_path.format(net='encoder', epoch=i*int(config['train']['d_epoch'])+times))
                test.test(encoder, args.gpu, testset, args.label)

        

        L_DF *= L_DI
        L_DT *= L_DI
        L_GF *= L_GI

        # L_ED *= 1.1
        L_EQ *= L_EQI
        # if i % int(config['train']['save_epoch']) == 0:
        #    torch.save(encoder.state_dict(), checkpoint_path.format(net='encoder', epoch=i))
            # torch.save(generator.state_dict(), checkpoint_path.format(net='generator', epoch=i))
            # torch.save(discriminator.state_dict(), checkpoint_path.format(net='discriminator',epoch=i))

                
                

                    









                



                
