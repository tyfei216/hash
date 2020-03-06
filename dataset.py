import configparser
import numpy as np

import random
import torch

# loading dataset
def load_feature(modals, path):
    ret = {}
    for m in modals:
        feature = open(path + m + '.txt', 'r').read().split('\n')
        feature_list = []
        for i in range(len(feature) - 1):
            feature_string = feature[i].split()
            feature_float = []
            for j in range(len(feature_string)):
                feature_float.append(float(feature_string[j]))
            
            feature_float = np.asarray(feature_float)
            feature_list.append(feature_float)

        ret[m] = feature_list

    return ret

def load_label(modals, path):
    ret = {}
    for m in modals:
        feature = open(path + m + '_label.txt', 'r').read().split('\n')
        feature_list = []
        for i in range(len(feature) - 1):
            feature_string = feature[i].split()
            feature_float = []
            for j in range(len(feature_string)):
                feature_float.append(float(feature_string[j]))
            
            feature_float = np.asarray(feature_float)
            feature_list.append(feature_float)

        ret[m] = feature_list
    
    return ret

def totensor(v):
    ret = {}
    for m, d in v.items():
        ret[m] = torch.from_numpy(d)
    return ret

class xmedia_test():
	def __init__(self, cfg):
		self.modals = [x for x in cfg['modals'].keys()]

		self.test_feature = load_feature(self.modals, './feature_znorm/' + 'test_')
		self.test_label = load_label(self.modals, './list/' + 'test_')
		self.test_size = {}
		for m,d in self.test_label.items():
			self.test_size[m] = len(d)

		self.database_feature = load_feature(self.modals, './feature_znorm/' + 'database_')
		self.database_label = load_label(self.modals, './list/'+'database_')
		self.database_size = {}
		for m, d in self.database_label.items():
			self.database_size[m] = len(d)

	def load_feature(self, path):
		ret = {}
		for m in self.modals:
			feature = open(path + m + '.txt', 'r').read().split('\n')
			feature_list = []
			for i in range(len(feature) - 1):
				feature_string = feature[i].split()
				feature_float = []
				for j in range(len(feature_string)):
					feature_float.append(float(feature_string[j]))
				
				feature_float = np.asarray(feature_float)
				feature_list.append(feature_float)

			ret[m] = feature_list

		return ret

	def load_label(self, path):
		ret = {}
		for m in self.modals:
			feature = open(path + m + '_label.txt', 'r').read().split('\n')
			feature_list = []
			for i in range(len(feature) - 1):
				feature_string = feature[i].split()
				feature_float = []
				for j in range(len(feature_string)):
					feature_float.append(float(feature_string[j]))
				
				feature_float = np.asarray(feature_float)
				feature_list.append(feature_float)

			ret[m] = feature_list
		
		return ret

#used for test only
class xmedia_class():
	def __init__(self, cfg):
		
		self.modals = [x for x in cfg['modals'].keys()]

		self.trainsize = int(cfg['dataset']['train_size'])
		self.testsize = int(cfg['dataset']['test_size'])

		self.test_feature = load_feature(self.modals, './feature_znorm/' + 'test_')
		self.test_label = load_label(self.modals, './list/' + 'test_')
		self.test_size = {}
		for m,d in self.test_label.items():
			self.test_size[m] = len(d)
			print('test size', m, self.test_size[m])

		self.train_feature = load_feature(self.modals, './feature_znorm/' + 'train_')
		self.train_label = load_label(self.modals, './list/' + 'train_')
		self.train_size = {}
		for m,d in self.train_label.items():
			self.train_size[m] = len(d)
			print('train size', m, self.train_size[m])

		# self.knn = self.load_knn('./knn/')

		self.batch_size = int(cfg['dataset']['batch_size'])
		self.data_size = int(cfg['dataset']['train_size'])
		self.rand = [i for i in range(self.data_size)]
		self.randt = [i for i in range(1000)]
		'''
		self.cnt = 0
		self.badp = 0
		self.badn = 0
		self.goodp = 0
		self.goodn = 0
		'''
		self.kx = int(cfg['dataset']['kx'])
	
	def reset(self):
		random.shuffle(self.rand)
		self.cnt = 0

	def update(self):
		self.cnt += self.batch_size

	def sample(self, gpu):
		d = {}
		l = {}
		for m in self.modals:
			d[m] = []
			l[m] = []

		for i in range(self.batch_size):
			index = self.rand[i+self.cnt]

			for j in self.modals:
				d[j].append(self.train_feature[j][index])
				l[j].append(self.train_label[j][index])

		retd = {}
		retl = {}

		for m, v in d.items():
			temp = torch.from_numpy(np.array(v)).to(torch.float32)
			if gpu:
				temp = temp.cuda()
			retd[m] =  temp

		for m, v in l.items():
			temp = torch.from_numpy(np.array(v)).to(torch.float32)
			if gpu:
				temp = temp.cuda()
			retl[m] = temp

		return retd, retl

	def sampletest(self, gpu):
		d = {}
		l = {}
		for m in self.modals:
			d[m] = []
			l[m] = []

		for i in range(20):
			index = self.randt[i+self.cnt]

			for j in self.modals:
				d[j].append(self.test_feature[j][index % self.test_size[j]])
				l[j].append(self.test_label[j][index % self.test_size[j]])

		retd = {}
		retl = {}

		for m, v in d.items():
			temp = torch.from_numpy(np.array(v)).to(torch.float32)
			if gpu:
				temp = temp.cuda()
			retd[m] =  temp

		for m, v in l.items():
			temp = torch.from_numpy(np.array(v)).to(torch.float32)
			if gpu:
				temp = temp.cuda()
			retl[m] = temp

		return retd, retl
	
#used for training
class xmedia():
	def __init__(self, cfg):
		
		self.modals = [x for x in cfg['modals'].keys()]

		self.trainsize = int(cfg['dataset']['train_size'])
		self.testsize = int(cfg['dataset']['test_size'])

		self.test_feature = load_feature(self.modals, './feature_znorm/' + 'test_')
		self.test_label = load_label(self.modals, './list/' + 'test_')
		self.test_size = {}
		for m,d in self.test_label.items():
			self.test_size[m] = len(d)

		self.train_feature = load_feature(self.modals, './feature_znorm/' + 'train_')
		self.train_label = load_label(self.modals, './list/' + 'train_')
		self.train_size = {}
		for m,d in self.train_label.items():
			self.train_size[m] = len(d)

		self.knn = self.load_knn('./knn/')

		self.batch_size = int(cfg['dataset']['batch_size'])
		self.data_size = int(cfg['dataset']['train_size'])
		self.rand = [i for i in range(self.data_size)]
		'''
		self.cnt = 0
		self.badp = 0
		self.badn = 0
		self.goodp = 0
		self.goodn = 0
		'''
		self.kx = int(cfg['dataset']['kx'])


	def reset(self):
		random.shuffle(self.rand)
		self.cnt = 0

	def update(self):
		self.cnt += self.batch_size

	def checklabel(self, a, b):
		# print(a, b)
		a = sum(a^b)
		if a == 0:
			return False
		else:
			return True

	def sample(self, m, gpu):
		pos = {}
		neg = {}

		for m in self.modals:
			pos[m] = []
			neg[m] = []

		for i in range(self.batch_size):
			index = self.rand[i+self.cnt]
			for j in self.modals:
				if j==m:
					#print j,i
					pos[j].append(self.train_feature[j][index])
				else:
					t_idx = random.randint(0,self.kx-1)
					'''
					if self.checklabel(self.train_label[m][index].astype(np.int32), self.train_label[j][self.knn[j][index][t_idx]].astype(np.int32)):
						self.badp += 1
						if self.badp % 2000 == 0:
							print('badp', self.badp, self.goodp)
					else:
						self.goodp += 1
					'''
					pos[j].append(self.train_feature[j][self.knn[j][index][t_idx]])
					

			for j in self.modals:
				if j==m:
					neg[j].append(self.train_feature[j][index])
				else:
					k = random.randint(0, self.trainsize-1)
					while k in self.knn[j][index]:
						k = random.randint(0,self.trainsize-1)
					'''
					if not self.checklabel(self.train_label[m][index].astype(np.int32), self.train_label[j][k].astype(np.int32)):
						self.badn += 1
						if self.badn % 500 == 0:
							print('badn', self.badn, self.goodn)
					else:
						self.goodn += 1
					'''
					neg[j].append(self.train_feature[j][k])
					
		
		negret = {} 
		for m, d in neg.items():
			temp = torch.from_numpy(np.array(d)).to(torch.float32)
			if gpu:
				temp = temp.cuda()
			negret[m] =  temp

		posret = {}
		for m, d in pos.items():
			temp = torch.from_numpy(np.array(d)).to(torch.float32)
			if gpu:
				temp = temp.cuda()
			posret[m] = temp

		return posret, negret
			

	def load_knn(self, path):
		knn_result = {}
		for m in self.modals:
			filename = path + 'KNN_' + m + '.npy'
			knn = np.load(filename)
			knn_result[m] = knn.astype(int)
		return knn_result
		
class xmedia_new():
	def __init__(self, cfg):
		
		self.modals = [x for x in cfg['modals'].keys()]

		self.trainsize = int(cfg['dataset']['train_size'])
		self.testsize = int(cfg['dataset']['test_size'])

		self.test_feature = load_feature(self.modals, './feature_znorm/' + 'test_')
		self.test_label = load_label(self.modals, './list/' + 'test_')
		self.test_size = {}
		for m,d in self.test_label.items():
			self.test_size[m] = len(d)

		self.train_feature = load_feature(self.modals, './feature_znorm/' + 'train_')
		self.train_label = load_label(self.modals, './list/' + 'train_')
		self.train_size = {}
		for m,d in self.train_label.items():
			self.train_size[m] = len(d)

		self.knn = self.load_knn('./knn/')

		self.batch_size = int(cfg['dataset']['batch_size'])
		self.data_size = int(cfg['dataset']['train_size'])
		self.rand = [i for i in range(self.data_size)]

		self.trainpos = {}
		self.trainneg = {}
		'''
		self.cnt = 0
		self.badp = 0
		self.badn = 0
		self.goodp = 0
		self.goodn = 0
		'''
		self.kx = int(cfg['dataset']['kx'])


	def reset(self):
		random.shuffle(self.rand)
		self.trainpos = {}
		self.trainneg = {}
		self.cnt = 0

		for m in self.modals:
			self.trainpos[m] = {}
			self.trainneg[m] = {}
			for k in self.modals:
				self.trainpos[m][k] = []
				self.trainneg[m][k] = []
		
		for m in self.modals:
			for i in range(self.trainsize):
				index = self.rand[i+self.cnt]
				for j in self.modals:
					if j==m:
						#print j,i
						self.trainpos[m][j].append(self.train_feature[j][index])
					else:
						t_idx = random.randint(0,self.kx-1)
						self.trainpos[m][j].append(self.train_feature[j][self.knn[j][index][t_idx]])
						

				for j in self.modals:
					if j==m:
						self.trainneg[m][j].append(self.train_feature[j][index])
					else:
						k = random.randint(0, self.trainsize-1)
						while k in self.knn[j][index]:
							k = random.randint(0,self.trainsize-1)
						self.trainneg[m][j].append(self.train_feature[j][k])
			

	def update(self):
		self.cnt += self.batch_size

	def checklabel(self, a, b):
		# print(a, b)
		a = sum(a^b)
		if a == 0:
			return False
		else:
			return True

	def sample(self, m, gpu):
		pos = {}
		neg = {}

		for mm in self.modals:
			pos[mm] = self.trainpos[m][mm][self.cnt:self.cnt+self.batch_size]
			neg[mm] = self.trainneg[m][mm][self.cnt:self.cnt+self.batch_size]
		
		negret = {} 
		for m, d in neg.items():
			temp = torch.from_numpy(np.array(d)).to(torch.float32)
			if gpu:
				temp = temp.cuda()
			negret[m] =  temp

		posret = {}
		for m, d in pos.items():
			temp = torch.from_numpy(np.array(d)).to(torch.float32)
			if gpu:
				temp = temp.cuda()
			posret[m] = temp

		return posret, negret
			

	def load_knn(self, path):
		knn_result = {}
		for m in self.modals:
			filename = path + 'KNN_' + m + '.npy'
			knn = np.load(filename)
			knn_result[m] = knn.astype(int)
		return knn_result

		


if __name__ == '__main__':
	
	config = configparser.ConfigParser()
	config.read('try.ini')

	dataset = xmedia(config)

	dataset.reset()

	for i in range(3):
		for m in config['modals']:
			pos, neg = dataset.sample(m)
			print(pos['img'].shape)
