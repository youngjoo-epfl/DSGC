from tfnntools.model import BaseNet, rprint
from utils import L2feeddict
from layers import embedding_network, histNorm_layer, graphConv_layer, projHist_layer, getReceptiveField, convertToGraph
import tensorflow as tf
import numpy as np
from IPython import embed

class GCNet(BaseNet):
    def default_params(self):
        d_params = super().default_params()
        d_params['n_vectors'] = 50
        d_params['n_classes'] = 2
        d_params['node_feat_out'] = [50,50]
        d_params['node_khop'] = [4,4]
        d_params['graph_feat_out'] = [50,50]
        d_params['graph_khop'] = [4,4]
        d_params['norm_coef'] = 1
        d_params['n_bins'] = 5
        d_params['flag_mask'] = True
        d_params['flag_norm'] = True
        d_params['flag_stats'] = 'hist'
        d_params['flag_tanh'] = True
        d_params['use_attributes'] = False
        d_params['use_sum_arch'] = False
        d_params['laplacian'] = None
        return d_params
    #def __init__(self, *args, **kwargs):
    #    super().__init__(*args, **kwargs)
    #    self._given_L = L
    
    def _build_net(self):
        rprint('--- Build GCNet ---', reuse=False)
        
        # 0) set random seed
        tf.random.set_random_seed(1)

        # 1) make the placeholders
        self.L = tf.sparse_placeholder(dtype=tf.float32, name='L')
        #self.y = tf.placeholder(tf.int32, shape=[1, self.params['n_classes']])
        self.y = tf.placeholder(tf.float32, shape=[1, self.params['n_classes']])

        #shape_x = [None, 1, 1] #[Num_sample, Nume_node, feat_in]
        shape_x = [None, 1] #[Nume_node, feat_in]
        self.x = tf.placeholder(tf.float32, shape=shape_x, name = "x")
        
        #shape_Idx = [None, None]
        #self.setIdx = tf.placeholder(tf.float32, shape=shape_Idx, name = "setIdx")
        
        #shape_attributes = [None, self.params['node_label_max']] #[num_nodes, feat_dim]
        #self.node_attributes = tf.placeholder(tf.float32, shape=shape_attributes, name="node_attributes")

        self._inputs = (self.L, self.y, self.x)
        rprint(' TFGraph inputs: ', reuse=False)
        rprint('   L : {}'.format(self.L.shape), reuse=False)
        rprint('   y : {}'.format(self.y.shape), reuse=False)
        rprint('   x : {}'.format(self.x.shape), reuse=False)
        #rprint('   node_attributes : {}'.format(self.node_attributes.shape), reuse=False)
        
    
        rprint(' Graph convolution for node embedding', reuse=False)

        scope = "layer_node_embedding"
        khop = self.params['node_khop']
        feat_out = self.params['node_feat_out']
        x = self.graph_hist_layer(self.x, khop, feat_out, scope, self.params['flag_mask'])
        ##INFO x = [num_node, feat_out]
        x = tf.reduce_sum(x, 0, keepdims=True)

        #fc-layer
        rprint(' End of convolution layers', reuse=False)            
        rprint('   x output shape: {}'.format(x.shape), reuse=False)
        rprint(' Fully connected layers with {} outputs'.format(self.params['n_classes']), reuse=False)            
        #x = tf.layers.flatten(x_feat)
        #x = x_g_lstm[-1]
        #x = tf.layers.flatten(x_g_lstm)
        rprint('   Reshape all embeddings to: {}'.format(x.shape), reuse=False)
        x = tf.layers.dense(x, 64, name='fc_1')
        #x = tf.layers.dense(x, 256, name='fc_1')
        self.logits = tf.layers.dense(x, self.params['n_classes'], name='fc_2')
        rprint('   Output shape: {}'.format(self.logits.shape), reuse=False)
        self._outputs = tf.nn.softmax(self.logits, axis=1)
        
        #Loss-define
        #self._loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.logits)
        self._loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.logits))
        rprint('   loss shape: {}'.format(self._loss.shape), reuse=False)



        
    def graph_hist_layer(self, x, khop, feat_out, scope, flag_mask):
        assert(len(khop)==len(feat_out))
        feat_in = x.shape[1]
        n_nodes = tf.shape(self.L)[0]
        for idx in range(len(khop)):
            rprint(' * Layer {} with {} graph convolution'.format(idx, feat_out[idx]), reuse=False)
            rprint('   x input shape: {}'.format(x.shape), reuse=False)
            shape = [khop[idx], feat_in, feat_out[idx]]
            x = graphConv_layer(x, self.L, shape, False, f'{scope}_{idx}')
            #x = tf.transpose(x, perm=[1,2,0]) 
            rprint('    Variable size after convolution {}: {}'.format(idx, x.shape), reuse=False)
            if idx<len(khop)-1:
                # relu 
                x = tf.nn.elu(x)
                feat_in = feat_out[idx]
        rprint(' End of convolution layers', reuse=False)            
        
        #x = tf.transpose(x, perm=[2,0,1]) 
        rprint(' Reshape of output Conv : {}'.format(x.shape), reuse=False)

        #Read-out
        if True:
            tf.summary.histogram("output/%s/gout"%scope, x)
            
        #return x_hist
        return x

    def _add_summary(self):
        super()._add_summary()        
        #traiable variables
        for trainVars in tf.trainable_variables():
            tf.summary.histogram("params/%s"%trainVars.name, trainVars, collections=["train"])

    def batch2dict(self,batch, train=True):
        #return L2feeddict(batch[0], self.params['n_vectors'], batch[1], train)
        #return L2feeddict(batch[0], self.params['n_vectors'], y=batch[1], node_attributes=batch[2], train=train)
        return L2feeddict(self.params['laplacian'], batch[0], y=batch[1], node_attributes=None, train=train)
        




# import tensorflow as tf
# import numpy as np

# from layers import *
# from IPython import embed

# class Model(object):
#     def __init__(self, config):
        
#         # MODEL 
#         self.num_layers  = config.num_layers
#         self.num_nodes   = None
#         self.feat_in     = config.feat_in
#         self.feat_out    = config.feat_out
#         self.norm_coef   = config.norm_coef 
#         self.flag_mask   = config.flag_mask 
#         self.flag_norm   = config.flag_norm
#         self.flag_stats  = config.flag_stats
#         self.flag_tanh   = config.flag_tanh
#         self.num_samples = config.num_samples
#         self.num_bins    = config.num_bins
#         self.batch_size  = config.batch_size
#         self.khop        = config.khop

#         # OPTIMIZATION
#         self.nclasses   = config.nclasses
#         self.optimizer  = config.optimizer
#         self.learning_rate = config.learning_rate
#         self.max_grad_norm = config.max_grad_norm
#         self.beta1 = config.beta1
#         self.beta2 = config.beta2

#         # DEPRECATED 
# #         self.task = config.task
# #         self.num_set    = 2 ##config.num_set
# #         self.kernel_width = config.kernel_width
# #         self.embedding_dim = config.embedding_dim
# #         self.embedding_space = config.embedding_space

#         if config.data_name == "MUTAG":
#             self.n_node_classes = 7
#         elif config.data_name == "NCI1":
#             self.n_node_classes = 37
#         elif config.data_name == "ENZYMES":
#             self.n_node_classes = 3
#         else:
#             self.n_node_classes = 86

#         self._build_placeholders()
#         self._build_model()
#         self._build_optim()
#         self._build_runs()

#     def _build_placeholders(self):
#         self.x = tf.placeholder(tf.float32, shape=[self.num_nodes,
#                                                        self.feat_in, self.num_samples], name = "x")
        
#         self.setIdx = tf.placeholder(tf.float32, shape=[self.num_samples,
#                                                        self.num_nodes], name = "setIdx")
        
#         self.num_nodes = tf.placeholder(tf.int32)
#         self.L = tf.sparse_placeholder(dtype=tf.float32, name='L')
#         self.y = tf.placeholder(tf.int32, shape=[self.batch_size, self.nclasses])
#         self.model_step = tf.Variable(0, name='model_step', trainable=False)

#     def _build_model(self, flag_reuse=None):
        
#         input_x = self.x
#         flag_mask = self.flag_mask
#         embeddings_list = []
#         feat_in = 1
        
#         rangeList = np.array([self.num_layers, self.khop, self.feat_out]).transpose()
#         for lyIdx, khop_, feat_out_ in rangeList:

#             scope = "layer_%d"%lyIdx            
#             shape = [khop_, feat_in, feat_out_]
            
#             embeddings, input_x = embedding_network(input_x, self.L, shape, scope=scope,
#                                     norm_coef    = self.norm_coef,
#                                     num_bins     = self.num_bins,
#                                     num_nodes    = self.num_nodes,
#                                     setIdx       = self.setIdx,
#                                     flag_reuse   = False, 
#                                     flag_mask    = flag_mask,
#                                     flag_summary = True, 
#                                     flag_norm    = self.flag_norm, 
#                                     flag_stats   = self.flag_stats, 
#                                     flag_tanh    = self.flag_tanh)
            
#             feat_in = feat_out_*(self.num_bins+2)
            
#             embeddings_list.append(embeddings)
            
#             # only use a mask the first time the embedding_network is used
#             flag_mask = False

#         #fc-layer
#         embeddings = tf.layers.flatten(embeddings)
#         self.dist = tf.layers.dense(embeddings, self.nclasses, name='fc')
        
#         #Loss-define
#         self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.dist)
        
#         #set-modelvars
#         self.model_vars = tf.trainable_variables()

#         #Write summary
#         self.model_summary = tf.summary.merge([tf.summary.scalar("cross_entropy", self.loss)])
        
#         #traiable variables
#         for trainVars in tf.trainable_variables():
#             tf.summary.histogram("params/%s"%trainVars.name, trainVars)
#         self.model_summary = tf.summary.merge_all()
 
#     def _build_optim(self):
#         def minimize(loss, step, var_list, learning_rate, optimizer):
#             if optimizer == "adam":
#                 optim = tf.train.AdamOptimizer(learning_rate, beta1=self.beta1, beta2=self.beta2)
#             elif optimizer == "rmsprop":
#                 optim = tf.train.RMSPropOptimizer(learning_rate)
#             else:
#                 raise Exception("[!] Unknown optimizer: {}".format(optimizer))

#             ## DO gradient clipping
#             if self.max_grad_norm is not None:
#                 grads_and_vars = optim.compute_gradients(loss)
#                 #grads_and_vars = optim.compute_gradients(loss, var_list=var_list)
#                 new_grads_and_vars = []
#                 for idx, (grad, var) in enumerate(grads_and_vars):
#                     if grad is not None and var in var_list:
#                         grad = tf.clip_by_norm(grad, self.max_grad_norm)
#                         grad = tf.check_numerics(grad, "numerical error in gradient for {}".format(var.name))
#                         new_grads_and_vars.append((grad, var))
#             else:
#                 new_grads_and_vars = optim.compute_gradients(loss, var_list=var_list)
#             return optim.apply_gradients(new_grads_and_vars, global_step=step) #after minimize, step is increased by 1.
#         # optim
#         self.model_optim = minimize(
#              self.loss,
#              self.model_step,
#              self.model_vars,
#              self.learning_rate,
#              self.optimizer)


#     def _build_runs(self):
#         def run(sess, feed_dict, fetch,
#                 summary_op, summary_writer, output_op=None, output_img=None):
#             if summary_writer is not None: #What is the diff btw summary_op and writer?
#                 fetch['summary'] = summary_op
#             if output_op is not None:
#                 fetch['loss'] = self.loss
#                 fetch['output'] = output_op
            
#             result = sess.run(fetch, feed_dict=feed_dict)
#             if "summary" in result.keys() and "step" in result.keys():
#                 summary_writer.add_summary(result['summary'], result['step'])
#                 summary_writer.flush()

#             return result

#         def train(sess, feed_dict, summary_writer=None, with_output=False):
#             fetch = {'loss': self.loss,
#                      'optim' : self.model_optim,
#                      'step': self.model_step}
#             return run(sess, feed_dict, fetch,
#                        self.model_summary, summary_writer,
#                        output_op=self.dist if with_output else None)
#         def test(sess, feed_dict, summary_writer=None, with_output=False):
#             fetch = {'loss': self.loss,
#                      'step': self.model_step}
#             return run(sess, feed_dict, fetch,
#                        self.model_summary, summary_writer,
#                        output_op=self.dist if with_output else None)


#         self.train = train
#         self.test = test














