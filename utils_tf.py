import linecache, pdb
import numpy as np
import configparser
import random

def is_same_cate(strA, strB, label_dim):
    labelA = strA.split()
    labelB = strB.split()
    
    if len(labelA) != label_dim or len(labelB) != label_dim:
        print(strA)
        print(strB)
        pdb.set_trace()
    
    for i in range(label_dim):
        if labelA[i] == '1' and labelA[i] == labelB[i]:
            return True
    return False

def push_query(query, url, dict):
    if query in dict:
        dict[query].append(url)
    else:
        dict[query] = [url]
    return dict
    
def make_train_dict(query_list, url_list, label_dim):
    query_url = {}
    query_pos = {}
    query_neg = {}
    query_num = len(query_list) - 1
    url_num = len(url_list) - 1
    
    for i in range(query_num):
        query = query_list[i]
        for j in range(url_num):
            url = url_list[j]
            if i == j:
                push_query(query, url, query_url)
                push_query(query, url, query_pos)
            else:
                push_query(query, url, query_url)
                push_query(query, url, query_neg)
            
    return query_url, query_pos, query_neg
    
def load_knn(config, path):
    knn_result = {}
    for modal in config['modals'].keys():
        file = path + 'KNN_' + modal + '.npy'
        knn = np.load(file)
        knn_result[modal] = knn.astype(int)
    return knn_result
    

def make_test_dict(query_list, url_list, query_label, url_label, label_dim):
    query_url = {}
    query_pos = {}
    query_num = len(query_list) - 1
    url_num = len(url_list) - 1
    
    for i in range(query_num):
        query = query_list[i]
        for j in range(url_num):
            url = url_list[j]				
            if is_same_cate(query_label[i], url_label[j], label_dim):
                push_query(query, url, query_url)
                push_query(query, url, query_pos)
            else:
                push_query(query, url, query_url)
    return query_url, query_pos	

def standard_all(a, b, c):
    for m in a.keys():
        aa = a[m]
        bb = b[m]
        cc = c[m]
        d = np.concatenate((aa,bb,cc), axis=0)
        s = standard2(d)
        a[m] = s[:aa.shape[0]]
        b[m] = s[aa.shape[0]:aa.shape[0]+bb.shape[0]]
        c[m] = s[aa.shape[0]+bb.shape[0]:]
        print(a[m].shape, b[m].shape, c[m].shape)
    
    return a, b, c

def standard(a):
    # return a
    print(a.size, a.shape)
    mu = np.mean(a, axis=0)
    sigma = np.std(a, axis=0)
    a = (a - mu)
    a = a/(sigma*4 + 0.00000001)
    print(len(a[a>1.0]), len(a[a<-1.0]))
    a[a>1.0] = 1.0
    a[a<-1.0] = -1.0
    return a

def standard2(a):
    print(a.size, a.shape)
    mu = np.mean(a, axis=0)
    sigma = np.std(a, axis=0)
    a = (a - mu)
    a = a/(sigma*4 + 0.00000001)
    print(len(a[a>1.0]), len(a[a<-1.0]))
    a[a>1.0] = 1.0
    a[a<-1.0] = -1.0
    return a

def load_all_query_url(config, feature_dir,list_dir, label_dim):

    test_feature = {}
    database_feature = {}
    test_label = {}
    database_label = {}
    
    for modal in config['modals'].keys():
        feature = open(feature_dir + 'test_' + modal + '.txt', 'r').read().split('\n')
        feature_list = []
        for i in range(len(feature) - 1):
            feature_string = feature[i].split()
            feature_float = []
            for j in range(len(feature_string)):
                feature_float.append(float(feature_string[j]))
            
            feature_float = np.asarray(feature_float)
            feature_list.append(feature_float)
            
        feature_list = np.asarray(feature_list)
        feature_list = standard(feature_list)
        test_feature[modal] = feature_list
        
    for modal in config['modals'].keys():
        feature = open(feature_dir + 'database_' + modal + '.txt', 'r').read().split('\n')
        feature_list = []
        for i in range(len(feature) - 1):
            feature_string = feature[i].split()
            feature_float = []
            for j in range(len(feature_string)):
                feature_float.append(float(feature_string[j]))
            
            feature_float = np.asarray(feature_float)
            feature_list.append(feature_float)
        
        feature_list = np.asarray(feature_list)
        feature_list = standard(feature_list)
        database_feature[modal] = feature_list

    for modal in config['modals'].keys():
        feature = open(list_dir + 'test_' + modal + '_label.txt', 'r').read().split('\n')
        feature_list = []
        for i in range(len(feature) - 1):
            feature_string = feature[i].split()
            feature_float = []
            for j in range(len(feature_string)):
                feature_float.append(float(feature_string[j]))
            
            feature_float = np.asarray(feature_float)
            feature_list.append(feature_float)

        test_label[modal] = feature_list
        
    for modal in config['modals'].keys():
        feature = open(list_dir + 'database_' + modal + '_label.txt', 'r').read().split('\n')
        feature_list = []
        for i in range(len(feature) - 1):
            feature_string = feature[i].split()
            feature_float = []
            for j in range(len(feature_string)):
                feature_float.append(float(feature_string[j]))
            
            feature_float = np.asarray(feature_float)
            feature_list.append(feature_float)
        
        feature_list = np.asarray(feature_list)
        # feature_list = standard(feature_list)
        database_label[modal] = feature_list

    #print(test_label) 
    return test_feature,database_feature,test_label,database_label


    train_img = open(list_dir + 'train_img.txt', 'r').read().split('\r\n')
    test_img = open(list_dir + 'test_img.txt', 'r').read().split('\r\n')
    test_img_label = open(list_dir + 'test_img_label.txt', 'r').read().split('\r\n')
    validation_img = open(list_dir + 'database_img.txt', 'r').read().split('\r\n')
    validation_img_label = open(list_dir + 'database_img_label.txt', 'r').read().split('\r\n')
    
    
    train_txt = open(list_dir + 'train_txt.txt', 'r').read().split('\r\n')
    test_txt = open(list_dir + 'test_txt.txt', 'r').read().split('\r\n')
    validation_txt = open(list_dir + 'database_txt.txt', 'r').read().split('\r\n')

    validation_img_label = open(list_dir + 'database_img_label.txt', 'r').read().split('\r\n')
    validation_txt_label = open(list_dir + 'database_txt_label.txt', 'r').read().split('\r\n')
    
    test_img_label = open(list_dir + 'test_img_label.txt', 'r').read().split('\r\n')
    test_txt_label = open(list_dir + 'test_txt_label.txt', 'r').read().split('\r\n')
    
    train_i2t, train_i2t_pos, train_i2t_neg = make_train_dict(train_img, train_txt, label_dim)
    train_t2i, train_t2i_pos, train_t2i_neg = make_train_dict(train_txt, train_img, label_dim)
    
    test_i2t, test_i2t_pos = make_test_dict(test_img, validation_txt, test_img_label, validation_txt_label, label_dim)
    test_t2i, test_t2i_pos = make_test_dict(test_txt, validation_img, test_txt_label, validation_img_label, label_dim)

    return train_i2t, train_i2t_pos, train_i2t_neg, train_t2i, train_t2i_pos, train_t2i_neg, test_i2t, test_i2t_pos, test_t2i, test_t2i_pos


def load_train_feature(config, feature_dir):
    train_feature = {}
    for modal in config['modals'].keys():
        feature = open(feature_dir + 'train_' + modal + '.txt', 'r').read().split('\n')
        
        feature_list = []
        for i in range(len(feature) - 1):
            feature_string = feature[i].split()
            feature_float = []
            for j in range(len(feature_string)):
                feature_float.append(float(feature_string[j]))
            
            feature_float = np.asarray(feature_float)
            feature_list.append(feature_float)

        feature_list = np.asarray(feature_list)
        feature_list = standard(feature_list)
        train_feature[modal] = feature_list

    return train_feature

def load_all_label(list_dir):
    label_dict = {}
    for dataset in ['database', 'test']:
        for modal in ['img', 'txt']:
            list = open(list_dir + dataset + '_' + modal + '.txt', 'r').read().split('\r\n')
            label = open(list_dir + dataset + '_' + modal + '_label.txt', 'r').read().split('\r\n')
            for i in range(len(list) - 1):
                item = list[i]
                label_string = label[i].split()
                label_float_list = []
                for j in range(len(label_string)):
                    label_float_list.append(float(label_string[j]))
                label_dict[item] = label_float_list
    return label_dict

def get_query_pos(file, semi_flag):
    query_pos = {}
    with open(file) as fin:
        for line in fin:
            cols = line.split()
            rank = float(cols[0])
            query = cols[1]
            url = cols[2]
            if rank > semi_flag:
                if query in query_pos:
                    query_pos[query].append(url)
                else:
                    query_pos[query] = [url]
    return query_pos


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


# Get batch data from training set
def get_batch_data(file, index, size):
    pos = []
    neg = []
    for i in range(index, index + size):
        line = linecache.getline(file, i)
        line = line.strip().split()
        pos.append([float(x) for x in line[0].split(',')])
        neg.append([float(x) for x in line[1].split(',')])
    return pos, neg

def get_hash_code(v):
    # print(v)
    cnt = 0.0
    ret = 0.0
    for i in v.values():
        cnt += 1
        ret += i
    #print('cnt', cnt)
    ret = ret.astype(np.float32)/cnt
    ret = np.asarray(ret + 0.5).astype(np.int32).astype(np.float32)
    #print(ret[0])
    return ret

def get_hash_code2(v):
    # print(v)
    cnt = 0.0
    ret = 0.0
    for i in v.values():
        #print(i)
        cnt += 1
        ret += i
    #print('cnt', cnt)
    ret = ret.astype(np.float32)/cnt
    #print(ret)
    minval = np.min(np.array(ret), 1)
    minval = minval.reshape(minval.shape[0], 1)
    maxval = np.max(np.array(ret), 1)
    maxval = maxval.reshape(maxval.shape[0], 1)
    #print(ret.shape, minval.shape, maxval.shape)
    ret = (ret - minval) / (maxval - minval + 0.0000001)
    #print(ret)
    ret = np.asarray(ret + 0.5).astype(np.int32).astype(np.float32)
    #print(ret[0])
    return ret

def generate_samples(config, fix, train_feature, knn_idx):
    data = {}
    TRAIN_NUM = int(config['dataset']['train_size'])
    Kx = int(config['dataset']['kx'])
    #pdb.set_trace()
    rand = [i for i in range(TRAIN_NUM)]
    random.shuffle(rand)
    for m in config['modals'].keys():
        data[m] = []
        data[m+'_neg'] = []
    for index in range(TRAIN_NUM):
        i = rand[index]
        for j in config['modals'].keys():
            if j==fix:
                #print j,i
                data[j].append(train_feature[j][i])
            else:
                t_idx = random.randint(0,Kx-1)
                data[j].append(train_feature[j][knn_idx[j][i][t_idx]])

        for j in config['modals'].keys():
            if j==fix:
                data[j+'_neg'].append(train_feature[j][i])
            else:
                k = random.randint(0, TRAIN_NUM-1)
                while k in knn_idx[j][i]:
                    k = random.randint(0,TRAIN_NUM-1)
                data[j+'_neg'].append(train_feature[j][k])

    return data

def generate_samples2(config, fix, train_feature, knn_idx):
    data = {}
    TRAIN_NUM = int(config['dataset']['train_size'])
    Kx = int(config['dataset']['kx'])
    #pdb.set_trace()
    rand = [i for i in range(TRAIN_NUM)]
    random.shuffle(rand)
    for m in config['modals'].keys():
        data[m] = []
        data[m+"1"] = []
        data[m+"2"] = []
        
    for index in range(TRAIN_NUM):
        i = rand[index]
        for j in config['modals'].keys():
            if j==fix:
                #print j,i
                data[j].append(train_feature[j][i])
                t_idx = random.randint(0,Kx-1)
                data[j+"1"].append(train_feature[j][knn_idx[j][i][t_idx]])
                t_idx = random.randint(0,Kx-1)
                data[j+"2"].append(train_feature[j][knn_idx[j][i][t_idx]])
            else:
                t_idx = random.randint(0,Kx-1)
                data[j].append(train_feature[j][knn_idx[j][i][t_idx]])
                t_idx = random.randint(0,Kx-1)
                data[j+"1"].append(train_feature[j][knn_idx[j][i][t_idx]])
                t_idx = random.randint(0,Kx-1)
                data[j+"2"].append(train_feature[j][knn_idx[j][i][t_idx]])

    return data