import numpy as np
import pdb
import os

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
		for  j in range(dlen):
			#pdb.set_trace()
			dist[j] = sum(test[i]^data[j])
		idx = np.argsort(dist)
		ton = 0
		for k in range(dlen):
			if sum(data_lab[idx[k]]^test_lab[i])==0:
				ton = ton+1
				res[i] += ton/(k+1.0)
		res[i] = res[i]/ton

	return np.mean(res)

def gethashcode(sess, config, data, hash_code, test_feature, database_feature):
	feed = {}
	for m in config['modals'].keys():
		feed[data[m]] = np.asarray(test_feature[m])
	hash_test = sess.run(hash_code, feed_dict=feed)
	for m in config['modals'].keys():
		feed[data[m]] = np.asarray(database_feature[m])
	hash_dataset = sess.run(hash_code, feed_dict = feed)

	return hash_test, hash_dataset

def testmap(config, hash_test1, hash_test2, hash_dataset1, hash_dataset2, test_label, database_label, label='two'):
	dh1 = []
	dh2 = []
	dl = []
	for m in config['modals'].keys():
		dh1.append(hash_dataset1[m])
		dh2.append(hash_dataset2[m])
		dl.append(database_label[m])

	data1 = np.concatenate(tuple(dh1))
	data2 = np.concatenate(tuple(dh2))

	data_lab = np.concatenate(tuple(dl)).astype(int)

	data = np.concatenate((data1, data2), axis=1)

	res = {}
	for m in config['modals'].keys():
		ht = np.concatenate((np.asarray(hash_test1[m]), np.asarray(hash_test2[m])), axis=1)
		res[m] = count_map(ht, data, np.asarray(test_label[m]).astype(int), data_lab)

	filename = 'result/test_map_' + str(config['parameters']['dim_out']) + '.txt'
	fp = open(filename,"a")
	fp.write(label+str(res)+"\n")
	fp.close()
	
	return np.mean([a for a in res.values()])
	


def MAP_ARGV(sess, config, data, hash_code, test_feature, database_feature, test_label, database_label, label='second'):

	# test_feature = np.asarray(test_feature)
	# database_feature = np.asarray(database_feature)
	feed = {}
	for m in config['modals'].keys():
		feed[data[m]] = np.asarray(test_feature[m])
	#pdb.set_trace()

	hash_test = sess.run(hash_code, feed_dict=feed)
	for m in config['modals'].keys():
		feed[data[m]] = np.asarray(database_feature[m])

	hash_dataset = sess.run(hash_code, feed_dict = feed)

	dh = []
	dl = []
	for m in config['modals'].keys():
		dh.append(hash_dataset[m])
		dl.append(database_label[m])
	
	data = np.concatenate(tuple(dh))
	data_lab = np.concatenate(tuple(dl)).astype(int)

#	test = np.concatenate((image_hash_test['I'],image_hash_test['T'],
#		image_hash_test['A'],image_hash_test['V'],image_hash_test['D'],))
#	test_lab = np.concatenate((test_label[0],test_label[1],test_label[2],test_label[3],test_label[4])).astype(int)
	res = {}
	for m in config['modals'].keys():
		res[m] = count_map(np.asarray(hash_test[m]), data, np.asarray(test_label[m]).astype(int), data_lab)
	

	filename = 'result/test_map_' + str(config['parameters']['dim_out']) + '.txt'
	fp = open(filename,"a")
	fp.write(label+str(res)+"\n")
	fp.close()
	
	return np.mean([a for a in res.values()])
