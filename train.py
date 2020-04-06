import configparser
import argparse
import os
import model

import torch
import torch.optim as optim
import torch.nn as nn
import dataset

from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', type=str, required=True, help='path to the required configure file')
    parser.add_argument('-gpuid', type=str, default=-1, help='given gpu to train on')
    parser.add_argument('-gpu', type=bool, default=True, help='whether to use a gpu')
    parser.add_argument('-save', type=str, default='./checkpoints/try/', help='place to save')
    args = parser.parse_args()

    checkpoint_path = args.save

    if int(args.gpuid) >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    config = configparser.ConfigParser()
    config.read(args.cfg)

    encoder = model.Encoder(config)
    generator = model.VideoGenerator3D(384, 1) 
    discriminator = model.Discriminator(config)
    if args.gpu:
        encoder = encoder.cuda()
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    encoder = encoder.train()
    generator = generator.train()
    discriminator = discriminator.train()

    # optimizers_E = {}
    optimizerE = optim.Adam(encoder.parameters(), lr=float(config['train']['lr']))
    optimizers_G = {}
    optimizers_D = {}
    
    for m in config['modals']:
        # params = [d for n, d in encoder.named_parameters() if m in str(n)]
        # optimizers_E[m] = optim.SGD(params, lr=float(config['train']['lr']))
        params = [d for n, d in generator.named_parameters() if m in str(n)] 
        optimizers_G[m] = optim.Adam(params, lr=float(config['train']['lr']))
        params = [d for n, d in discriminator.named_parameters() if m in str(n)]
        optimizers_D[m] = optim.Adam(params, lr=float(config['train']['lr']))

    # scheduler_E = {}
    schedulerE = optim.lr_scheduler.StepLR(optimizerE, int(config['train']['step_size']), 
                                gamma=float(config['train']['gamma']))
    scheduler_G = {}
    scheduler_D = {}
    for m in config['modals']:
        # scheduler_E[m] = optim.lr_scheduler.StepLR(optimizers_E[m], int(config['train']['step_size']), 
        #                        gamma=float(config['train']['gamma']))
        scheduler_G[m] = optim.lr_scheduler.StepLR(optimizers_G[m], int(config['train']['step_size']), 
                                gamma=float(config['train']['gamma']))                                
        scheduler_D[m] = optim.lr_scheduler.StepLR(optimizers_D[m], int(config['train']['step_size']), 
                                gamma=float(config['train']['gamma']))

    trainset = dataset.xmedia(config)

    rounds = trainset.trainsize // trainset.batch_size

    cri = nn.MSELoss()
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
        
        schedulerE.step()
        # for a in scheduler_E.values():
        #    a.step()
        for b in scheduler_G.values():
            b.step()
        for c in scheduler_D.values():
            c.step()
        trainset.reset()

        cp = float(config['train']['clip'])
        
        for j in range(rounds):

            for m in config['modals'].keys():

                pos, neg = trainset.sample(m, args.gpu)
                # print(pos['img'])
                outputpos = encoder(pos)
                outputneg = encoder(neg)

                hashcodepos = toBinary(outputpos, args.gpu)
                hashcodeneg = toBinary(outputneg, args.gpu)

                neg = toneg(hashcodepos)

                pos_sample = generator(hashcodepos, args.gpu)
                neg_sample = generator(hashcodeneg, args.gpu)

                fpos_w, fpos_ac = discriminator(todetach(pos_sample))
                fneg_w, fneg_ac = discriminator(todetach(neg_sample))
                tpos_w, tpos_ac = discriminator(pos)

                for p in discriminator.parameters():
                    p.data.clamp_(-cp, cp)
                
                for mm in config['modals'].keys():
                    loss_D = fpos_w[mm].mean() + fneg_w[mm].mean() - tpos_w[mm].mean() + \
                        L_DF * ((fpos_ac[mm]*(hashcodeneg[mm]*2-1)).mean() + (fneg_ac[mm] * (hashcodepos[mm]*2-1)).mean()) + \
                            L_DT * (tpos_ac[mm]*(hashcodeneg[mm]*2-1)).mean()

                    optimizers_D[mm].zero_grad()
                    loss_D.backward()
                    optimizers_D[mm].step()

                    fpos_wg, fpos_acg = discriminator({mm:pos_sample[mm]})
                    fneg_wg, fneg_acg = discriminator({mm:neg_sample[mm]})

                    loss_G = - fpos_wg[mm].mean() - fneg_wg[mm].mean() + \
                        L_GF * ((fpos_acg[mm]*(hashcodeneg[mm]*2-1)).mean() + (fneg_acg[mm] * (hashcodepos[mm]*2-1)).mean())
                    loss_G.backward()
                    optimizers_G[mm].step()

                    if m == mm and j % int(config['train']['print']) == 0:
                        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\t {modal} Loss_D: {:0.4f}\t'.format(
                            loss_D,
                            modal = m,
                            epoch=i,
                            trained_samples=int(config['dataset']['batch_size'])*j,
                            total_samples=int(config['dataset']['train_size'])
                        ))
                        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\t {modal} Loss_G: {:0.4f}\t'.format(
                            loss_G,
                            modal = m,
                            epoch=i,
                            trained_samples=int(config['dataset']['batch_size'])*j,
                            total_samples=int(config['dataset']['train_size'])
                        ))



                pos_sample = generator(hashcodepos, args.gpu)
                neg_sample = generator(hashcodeneg, args.gpu)

                dist = distance(outputpos, m) - distance(outputneg, m)
                # dist = activ(dist + beta)
                r = 0.0
                for mm in config['modals'].keys():
                    r += ((outputpos[mm]-hashcodepos[mm])**2 + (outputneg[mm]-hashcodeneg[mm])**2).mean()

                p = encoder(todetach(pos_sample))  
                n = encoder(todetach(neg_sample))
                loss_E = dist.mean() + L_ED * ((p[m]-n[m])**2).mean() + \
                    L_EQ * r
                optimizerE.zero_grad()
                loss_E.backward()
                optimizerE.step()

                if j % int(config['train']['print']) == 0:
                    # for p in discriminator.parameters():
                    #    print('parameters')
                    #    print(p)
                    print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\t {modal} Loss_E: {:0.4f}\t'.format(
                        loss_E,
                        modal = m,
                        epoch=i,
                        trained_samples=int(config['dataset']['batch_size'])*j,
                        total_samples=int(config['dataset']['train_size'])
                    ))


            trainset.update()

        L_DF *= L_DI
        L_DT *= L_DI
        L_GF *= L_GI

        L_ED *= L_EDI
        L_EQ *= L_EQI
        if i % int(config['train']['save_epoch']) == 0:
            torch.save(encoder.state_dict(), checkpoint_path.format(net='encoder', epoch=i))
            # torch.save(generator.state_dict(), checkpoint_path.format(net='generator', epoch=i))
            # torch.save(discriminator.state_dict(), checkpoint_path.format(net='discriminator',epoch=i))

                
                

                    









                



                
