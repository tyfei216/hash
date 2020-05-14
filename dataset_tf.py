import tensorflow as tf 
import utils_tf as ut 
import numpy as np

def xmedia(config):
    test_feature,database_feature,test_label,database_label = ut.load_all_query_url(config, './feature_znorm/','./list/', 20)
    train_feature = ut.load_train_feature(config, './feature_znorm/')
    #test_feature, database_feature, train_feature = ut.standard_all(test_feature, database_feature, train_feature)
    knn_idx = ut.load_knn(config, './knn/')
    return test_feature, database_feature, test_label, database_label, train_feature, knn_idx

def wiki(config):
    test_feature = {}
    test_feature['img'] = ut.standard2(np.load('./wiki/I_test.npy'))
    test_feature['txt'] = ut.standard2(np.load('./wiki/T_test.npy'))
    
    database_feature = {}
    database_feature['img'] = ut.standard2(np.load('./wiki/I_train.npy'))
    database_feature['txt'] = ut.standard2(np.load('./wiki/T_train.npy'))
    
    train_feature = database_feature

    test_label = {}
    test_label['img'] = np.load('./wiki/test_label.npy')
    n_values = np.max(test_label['img']) + 1
    test_label['img'] = np.eye(n_values)[test_label['img']]
    print('shape ', test_label['img'].shape)
    test_label['txt'] = np.load('./wiki/test_label.npy')
    n_values = np.max(test_label['txt']) + 1
    test_label['txt'] = np.eye(n_values)[test_label['txt']]

    database_label = {}
    database_label['img'] = np.load('./wiki/train_label.npy')
    n_values = np.max(database_label['img']) + 1
    print(n_values)
    database_label['img'] = np.eye(n_values)[database_label['img']]
    
    database_label['txt'] = np.load('./wiki/train_label.npy')
    n_values = np.max(database_label['txt']) + 1
    database_label['txt'] = np.eye(n_values)[database_label['txt']]

    knn_idx = {}
    knn_idx['img'] = np.load('./wiki/KNN_img.npy').astype(int)
    knn_idx['txt'] = np.load('./wiki/KNN_img.npy').astype(int)

    return test_feature, database_feature, test_label, database_label, train_feature, knn_idx