import configparser

config = configparser.ConfigParser()

config['modals'] = {'img':4096, 'txt':3000, 'aud':78, '3d':4700, 'vid':4096}

config['parameters'] = {'dim_hid':4096, 'dim_out':32, 'dim_ran':128}

config['dataset'] = {'test_size':1000, 'train_size':4000, 'batch_size':64, 'kx':10}

config['train'] = {'epoch':31, 'd_epoch':120, 'lr':0.0005, 'step_size':1, 'gamma': 0.1, 'train_encoder': 2, 'train_generator': 2,
                    'weight_decay':0.0005,
                    'beta':1.0, 'save_epoch':100, 'print': 20, 
                    'l_df':0.1, 'l_dt':0.4, 'l_di':1.2, 'l_gf':0.2, 'l_gi':1.2, 
                    'l_ed':0.01, 'l_edi':1.2, 'l_eq': 0.01, 'l_eqi':1.1, 'clip':0.05}

with open('all.ini', 'w') as f:
    config.write(f)
'''
config2 = configparser.ConfigParser()
config2.read('try.ini')

print(config2.sections())
for a in config2['modals']:
    print(a)
for a in config2['parameters']:
    print(a)
for a in config2['dataset'].values():
    print(a)
'''