import torch
import torch.nn as nn

class SimgleEncoder(nn.Module):
    def __init__(self, name, dim_In, dim_Hid, dim_Out):
        super(SimgleEncoder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_In, dim_Hid),
            nn.Tanh(),
            nn.Linear(dim_Hid, dim_Out),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.net(input)


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()

        self.networks = nn.ModuleDict()

        for m, d in cfg['modals'].items():
            self.networks.update({m: self.buildNet(int(d), int(cfg['parameters']['dim_hid']), int(cfg['parameters']['dim_out']))})
    
    def buildNet(self, dim_In, dim_Hid, dim_Out):
        return nn.Sequential(
            nn.Linear(dim_In, dim_Hid),
            nn.Tanh(),
            nn.Linear(dim_Hid, dim_Out),
            nn.Sigmoid(),
        )

    def forward(self, input):
        ret = {}
        for m, d in input.items():
            ret[m] = self.networks[m](d)

        return ret

class Encoder_class(nn.Module):
    def __init__(self, cfg):
        super(Encoder_class, self).__init__()

        self.networks = nn.ModuleDict()

        for m, d in cfg['modals'].items():
            self.networks.update({m: self.buildNet(int(d), int(cfg['parameters']['dim_hid']), int(cfg['parameters']['dim_out']))})


    def buildNet(self, dim_In, dim_Hid, dim_Out):
        return nn.Sequential(
            nn.Linear(dim_In, dim_Hid),
            nn.ReLU(),
            nn.Linear(dim_Hid, dim_Out),
        )

    def forward(self, input):
        ret = {}
        for m, d in input.items():
            ret[m] = self.networks[m](d)

        return ret

class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()

        self.networks = nn.ModuleDict()
        self.dim_ran = int(cfg['parameters']['dim_ran'])

        for m, d in cfg['modals'].items():
            self.networks.update({m: self.buildNet(int(cfg['parameters']['dim_out'])+int(cfg['parameters']['dim_ran']), 
            int(cfg['parameters']['dim_hid']), int(d))})


    def buildNet(self, dim_In, dim_Hid, dim_Out):
        return nn.Sequential(
            nn.Linear(dim_In, dim_In*2), 
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(dim_In*2, dim_Hid),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(dim_Hid, dim_Out),
            nn.Tanh(),
        )

    def forward(self, input, gpu):
        ret = {}
        for m, d in input.items():
            batchSize = d.shape[0]
            rand = torch.rand((batchSize, self.dim_ran))
            if gpu:
                rand = rand.cuda()
            d = torch.cat((d, rand), 1)
            ret[m] = self.networks[m](d)
        return ret

class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()

        self.firstLayers = nn.ModuleDict()
        for m, d in cfg['modals'].items():
            self.firstLayers.update({m: self.buildFirstLayer(int(d), int(cfg['parameters']['dim_hid']))})

        self.TF = nn.ModuleDict()
        for m in cfg['modals'].keys():
            self.TF.update({m: self.buildTF(int(cfg['parameters']['dim_hid']))})

        self.AC = nn.ModuleDict()
        for m in cfg['modals'].keys():
            self.AC.update({m: self.buildAC(int(cfg['parameters']['dim_hid']), int(cfg['parameters']['dim_out']))})

    def buildFirstLayer(self, dim_In, dim_Hid):
        return nn.Sequential(
            nn.Linear(dim_In, dim_Hid),
            nn.ReLU(),
            nn.Linear(dim_Hid, dim_Hid),
            nn.ReLU(),
        )

    def buildTF(self, dim_Hid):
        return nn.Linear(dim_Hid, 1)

    def buildAC(self, dim_Hid, dim_Out):
        return nn.Linear(dim_Hid, dim_Out)

    def forward(self, input):
        retTF = {}
        retAC = {}
        for m, d in input.items():
            output = self.firstLayers[m](d)
            retTF[m] = self.TF[m](output)
            retAC[m] = self.TF[m](output)
        return retTF, retAC

if __name__ == '__main__':
    import configparser
    import torch.optim as optim

    def toBinary(v):
        ret = {}
        for m, d in v.items():
            dd = d.detach()
            dd = (dd+0.5).to(torch.int32).to(torch.float32)
            ret[m] = dd
        return ret

    def todetach(v):
        ret = {}
        for m, d in v.items():
            ret[m] = d.detach()
        return ret

    def toneg(v):
        ret = {}
        for m, d in v.items():
            ret[m] = 1 - d
        return ret

    def distance(v, m):
        ret = 0.0
        for mm, d in v.items():
            if mm != m:
                ret += (d.detach()-v[m])*(d.detach()-v[m])
        return ret

    config = configparser.ConfigParser()
    config.read('try.ini')
    encoder = Encoder(config)
    generator = Generator(config) 
    discriminator = Discriminator(config)

    optimizers_E = {}
    optimizers_G = {}
    optimizers_D = {}
    for m in config['modals']:
        params = [d for n, d in encoder.named_parameters() if m in str(n)]
        optimizers_E[m] = optim.Adam(params, lr=float(config['train']['lr']))
        params = [d for n, d in generator.named_parameters() if m in str(n)] 
        optimizers_G[m] = optim.Adam(params, lr=float(config['train']['lr']))
        params = [d for n, d in discriminator.named_parameters() if m in str(n)]
        optimizers_D[m] = optim.Adam(params, lr=float(config['train']['lr']))
    
    pos = {'img':torch.randn(3, 4096), 'txt':torch.randn(3, 3000)}
    neg = {'img':torch.randn(3, 4096), 'txt':torch.randn(3, 3000)}
    
    outputpos = encoder(pos)
    outputneg = encoder(neg)

    hashcodepos = toBinary(outputpos)
    hashcodeneg = toBinary(outputneg)

    neg = toneg(hashcodepos)

    pos_sample = generator(hashcodepos)
    neg_sample = generator(hashcodeneg)

    fpos_w, fpos_ac = discriminator(todetach(pos_sample))
    fneg_w, fneg_ac = discriminator(todetach(neg_sample))
    tpos_w, tpos_ac = discriminator(pos)

    for mm in config['modals'].keys():
        print(mm, ' dis')
        loss_D = fpos_w[mm].mean() + fneg_w[mm].mean() - tpos_w[mm].mean() + \
             ((fpos_ac[mm]*(hashcodeneg[mm]*2-1)).mean() + (fneg_ac[mm] * (hashcodepos[mm]*2-1)).mean()) + \
                 (tpos_ac[mm]*(hashcodeneg[mm]*2-1)).mean()

        optimizers_D[mm].zero_grad()
        loss_D.backward()
        optimizers_D[mm].step()

        fpos_wg, fpos_acg = discriminator({mm:pos_sample[mm]})
        fneg_wg, fneg_acg = discriminator({mm:neg_sample[mm]})
        print(mm, ' gen')
        loss_G = - fpos_wg[mm].mean() - fneg_wg[mm].mean() + \
             ((fpos_acg[mm]*(hashcodeneg[mm]*2-1)).mean() + (fneg_acg[mm] * (hashcodepos[mm]*2-1)).mean())
        loss_G.backward()
        optimizers_G[mm].step()
    pos_sample = generator(hashcodepos)
    neg_sample = generator(hashcodeneg)

    dist = distance(outputpos, m) - distance(outputneg, m)
   
    p = encoder(todetach(pos_sample))  
    n = encoder(todetach(neg_sample))
    loss_E = dist.mean() + ((p[m]-n[m])**2).mean() + \
    (((outputpos[m]-hashcodepos[m])**2).mean() + ((outputneg[m]-hashcodeneg[m])**2).mean())
    optimizers_E[m].zero_grad()
    loss_E.backward()
    optimizers_E[m].step()

    '''
    b = {'img':I, 'txt':T}
    generator = Generator(config)
    c = generator(b)

    print(c)

    discriminator = Discriminator(config)
    print(discriminator(c))

    print(type(discriminator.named_parameters()))
    print(type(discriminator.parameters()))
    params = [d for n, d in encoder.named_parameters() if 'img' in str(n)]
    print(params)

    # for a, b in discriminator.named_parameters():
    #    print(a, b)
    '''
