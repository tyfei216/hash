import configparser
import tensorflow as tf

'''
class Encoder():
    def __init__(self, config):

        self.posdata = {}
        self.negdata = {}

        for m, d in config['modals'].items():
            self.posdata[m] = tf.placeholder(tf.float32, shape=[None, int(d)], name=m+"_data")
            self.negdata[m] = tf.placeholder(tf.float32, shape=[None, int(d)], name=m+"_data")

        with tf.variable_scope('encoder'):
            self.hash_sig = {}
            self.hash_code = {}
            self.hash_code_neg = {}
            self.hash_sig_neg = {}
            self.params = {}
            for m in config['modals'].keys():
                self.hash_sig[m], self.hash_code[m], self.params[m] = self.build_net(self.posdata[m], m, int(config['modals'][m]), 
                int(config['parameters']['dim_hid']), int(config['parameters']['dim_out']), None)

                self.hash_sig_neg[m], self.hash_code_neg[m] = self.build_net(self.negdata[m], m, int(config['modals'][m]), 
                int(config['parameters']['dim_hid']), int(config['parameters']['dim_out']), self.params[m])

        self.regulation = self.compute_regulation(float(config['train']['weight_decay']))

    
    def build_layer(self, input, name, shape, l_param, activ):
        W_init_args = {}
        b_init_args = {}
        if l_param == None:
            l_param = []
            W_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
            b_init = tf.constant_initializer(value=0.0)
            l_param.append(tf.get_variable(name=name+'_W', shape=shape, initializer=W_init, **W_init_args))
            l_param.append(tf.get_variable(name=name+'_b', shape=(shape[-1]), initializer=b_init, **b_init_args))
            if activ == 'tanh':
                return tf.nn.tanh(tf.nn.xw_plus_b(input, l_param[0], l_param[1])), l_param
            elif activ == 'sigmoid':
                return tf.sigmoid(tf.nn.xw_plus_b(input, l_param[0], l_param[1])), l_param
        else:
            if activ == 'tanh':
                return tf.nn.tanh(tf.nn.xw_plus_b(input, l_param[0], l_param[1]))
            elif activ == 'sigmoid':
                return tf.sigmoid(tf.nn.xw_plus_b(input, l_param[0], l_param[1]))

			
    def build_net(self, input, name, input_dim, hidden_dim, output_dim, n_param):
        if n_param == None:
            n_param = []
            l1, l1_param = self.build_layer(input, name+'_l1', (input_dim, hidden_dim), None, 'tanh')
            l2, l2_param = self.build_layer(l1, name+'_l2', (hidden_dim, output_dim), None, 'sigmoid')
            n_param.append(l1_param)
            n_param.append(l2_param)
            hash_code = tf.cast(l2 + 0.5, tf.int32)
            return l2, hash_code, n_param
        else:
            l1 = self.build_layer(input, name+'_l1', (input_dim, hidden_dim), n_param[0], 'tanh')
            l2 = self.build_layer(l1, name+'_l2', (hidden_dim, output_dim), n_param[1], 'sigmoid')
            hash_code = tf.cast(l2 + 0.5, tf.int32)
            return l2, hash_code

    def compute_regulation(self, weight_decay):
        result = 0.0
        var_list = []
        for var in self.params.values():
            var_list.append(var[0][0])
            var_list.append(var[0][1])
            var_list.append(var[1][0])
            var_list.append(var[1][1])
        for item in var_list:
            result += tf.nn.l2_loss(item)
        return weight_decay * result
'''

def build_layer(x, out_dim, scope = 'fc'):
    with tf.variable_scope(scope):
        w = tf.get_variable(
            'w', [x.get_shape()[-1], out_dim],
            initializer=tf.truncated_normal_initializer(stddev=0.1))

        biases = tf.get_variable(
            'biases', [out_dim], initializer=tf.constant_initializer(0.0))

        output = tf.nn.bias_add(tf.matmul(x, w), biases)

        return output

def compute_regulation(weight_decay, scope):
    result = 0.0
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    for item in var_list:
        result += tf.nn.l2_loss(item)
    return weight_decay * result

def Encoder(config, feature, reuse = False):
    with tf.variable_scope('encoder', reuse = reuse,):  
        # regularizer=tf.contrib.layers.l2_regularizer(0.1)):
        hash_sig = {}
        hash_code = {}
        for m in config['modals'].keys():
            l1 = tf.nn.tanh(build_layer(feature[m], int(config['parameters']['dim_hid']), 'fc1'+m))
            hash_sig[m] = tf.sigmoid(build_layer(l1, int(config['parameters']['dim_out']), 'fc2'+m))
            hash_code[m] = tf.cast(hash_sig[m] + 0.5, tf.int32)

    return hash_sig, hash_code

def Encoder_new(config, feature, reuse = False):
    with tf.variable_scope('encoder', reuse = reuse,):  
        # regularizer=tf.contrib.layers.l2_regularizer(0.1)):
        hash_sig = {}
        hash_code = {}
        for m in config['modals'].keys():
            l0 = tf.nn.relu(build_layer(feature[m], int(config['modals'][m]), 'fc0'+m))
            l1 = tf.nn.tanh(build_layer(l0, int(config['parameters']['dim_hid']), 'fc1'+m))
            hash_sig[m] = tf.sigmoid(build_layer(l1, int(config['parameters']['dim_out']), 'fc2'+m))
            hash_code[m] = tf.cast(hash_sig[m] + 0.5, tf.int32)

    return hash_sig, hash_code

def Generator(config, hash_code, z, reuse = False):
    with tf.variable_scope('generator', reuse=reuse):
        generated_feature = {}
        for m in config['modals'].keys():
            z_labels = tf.concat([hash_code, z], 1)
            first = tf.nn.relu(build_layer(z_labels, int(config['parameters']['dim_out'])*4, 'fc1'+m))
            second = tf.nn.relu(build_layer(first, int(config['parameters']['dim_hid']), 'fc2'+m))
            generated_feature[m] = tf.nn.tanh(build_layer(second, int(config['modals'][m]), 'fc3'+m))
    
    return generated_feature

def Discriminator(config, feature, reuse = False):
    with tf.variable_scope('discriminator', reuse=reuse):
        source_logits = {}
        hash_logits = {}
        for m in config['modals'].keys():
            first = tf.nn.relu(build_layer(feature[m], int(config['parameters']['dim_hid']), 'fc1'+m))
            second = tf.nn.relu(build_layer(first, int(config['parameters']['dim_hid'])//4, 'fc2'+m))
            source_logits[m] = build_layer(second, 1, 'sl'+m)
            hash_logits[m] = build_layer(second, int(config['parameters']['dim_out']), 'cl'+m)
    return source_logits, hash_logits        
