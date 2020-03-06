import torch
import numpy as np
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.9)
        # nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)
    
# trasform and computing
def toBinary(v, gpu):
    ret = {}
    for m, d in v.items():
        dd = d.detach()
        if gpu:
            dd = (dd+0.5).to(torch.int32).to(torch.float32).cuda()
        else:
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
            ret += (d-v[m])*(d-v[m])
    return ret

def totensor(v):
    ret = {}
    for m, d in v.items():
        ret[m] = torch.from_numpy(np.array(d)).to(torch.float32)
    return ret
    

    