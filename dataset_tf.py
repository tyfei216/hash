import tensorflow as tf 
import utils_tf as ut 

def xmedia(config):
    test_feature,database_feature,test_label,database_label = ut.load_all_query_url(config, './feature_znorm/','./list/', 20)
    train_feature = ut.load_train_feature(config, './feature_znorm/')
    #test_feature, database_feature, train_feature = ut.standard_all(test_feature, database_feature, train_feature)
    knn_idx = ut.load_knn(config, './knn/')
    return test_feature, database_feature, test_label, database_label, train_feature, knn_idx