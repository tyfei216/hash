import configparser
import tensorflow as tf

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