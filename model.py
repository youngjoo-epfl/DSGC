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
        return d_params
    
    def _build_net(self):
        rprint('--- Build GCNet ---', reuse=False)
        
        # 0) set random seed
        tf.random.set_random_seed(1)

        # 1) make the placeholders
        self.L = tf.sparse_placeholder(dtype=tf.float32, name='L')
        self.y = tf.placeholder(tf.int32, shape=[1, self.params['n_classes']])

        shape_x = [None, 1, None]
        self.x = tf.placeholder(tf.float32, shape=shape_x, name = "x")
        
        shape_Idx = [None, None]
        self.setIdx = tf.placeholder(tf.float32, shape=shape_Idx, name = "setIdx")
        
        shape_attributes = [None, self.params['node_label_max']] #[num_nodes, feat_dim]
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
        x, x_e1, x_e2, x_e_lstm, x_e_state  = self.graph_hist_layer(self.x, khop, feat_out, scope, self.params['flag_mask'])
        ##INFO x = [num_sample feat_out, bin]
        ## incase of x_list = [num_sample, feat_out]

        self.embedding_node = x
        #self.embedding_node = x_e_lstm[-1]
        ## 3) Second convolution layer to go to the graph embedding
        ## lift to a tensor of num_nodes size (when performing the computation for a node subset)
        
        x = convertToGraph(x, self.setIdx)
        #x = convertToGraph(x_e_lstm[-1], self.setIdx)
        ## x = [Node, feat_out*bin, 1] --> in the graph embedding layer it will work as an input
        self.converted = x
        ##Here I need to add up attributes info into x
        if self.params['use_attributes']:
            print("I am using node attributes!!!!!")
            embed()
            #x = tf.concat([x, tf.expand_dims(self.node_attributes, 2)], 1) #This is concatenated feature with attributes
            #x = tf.expand_dims(self.node_attributes, 2) #only node attributes
        
            ##Removing node which is not used
            #used_node = tf.cast(tf.matmul(tf.transpose(self.setIdx), self.setIdx), dtype=tf.bool)
            #used_node = tf.cast(used_node, dtype=tf.float32)
            #x = tf.tensordot(used_node, x, axes=[0,0])
        

        
        rprint(' Graph convolution for graph embedding', reuse=False)

        scope = "layer_graph_embedding"
        khop = self.params['graph_khop']
        feat_out = self.params['graph_feat_out']
        x, x_g1, x_g2, x_g_lstm, x_g_state = self.graph_hist_layer(x, khop, feat_out, scope, False)
        self.embedding_graph = x
        #self.embedding_graph = x_g_lstm[-1]
        # for checkign feature
        self._x_e1 = x_e1
        self._x_e2 = x_e2
        self._x_g1 = x_g1
        self._x_g2 = x_g2


        #fc-layer
        rprint(' End of convolution layers', reuse=False)            
        rprint('   x output shape: {}'.format(x.shape), reuse=False)
        rprint(' Fully connected layers with {} outputs'.format(self.params['n_classes']), reuse=False)            
        x = tf.layers.flatten(x)
        #x = x_g_lstm[-1]
        #x = tf.layers.flatten(x_g_lstm)
        rprint('   Reshape all embeddings to: {}'.format(x.shape), reuse=False)
        x = tf.layers.dense(x, 64, name='fc_1')
        #x = tf.layers.dense(x, 256, name='fc_1')
        self.logits = tf.layers.dense(x, self.params['n_classes'], name='fc_2')
        rprint('   Output shape: {}'.format(self.logits.shape), reuse=False)
        self._outputs = tf.nn.softmax(self.logits, axis=1)
        
        #Loss-define
        self._loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.logits)




        
    def graph_hist_layer(self, x, khop, feat_out, scope, flag_mask):
        assert(len(khop)==len(feat_out))
        feat_in = x.shape[1]
        n_nodes = tf.shape(self.L)[0]
        for idx in range(len(khop)):
            rprint(' * Layer {} with {} graph convolution'.format(idx, feat_out[idx]), reuse=False)
            rprint('   x input shape: {}'.format(x.shape), reuse=False)
            shape = [khop[idx], feat_in, feat_out[idx]]
            x = graphConv_layer(x, self.L, shape, False, f'{scope}_{idx}')
            x = tf.transpose(x, perm=[1,2,0]) 
            rprint('    Variable size after convolution {}: {}'.format(idx, x.shape), reuse=False)
            if idx<len(khop)-1:
                # relu 
                x = tf.nn.elu(x)
                feat_in = feat_out[idx]
        rprint(' End of convolution layers', reuse=False)            
        
        x = tf.transpose(x, perm=[2,0,1]) 
        rprint(' Reshape to : {}'.format(x.shape), reuse=False)

        
        
        # get receptive field masks
        mask, mask_norm = getReceptiveField(x)

        rprint(' Normalization', reuse=False)
        # graph normalization layer
        x_norm, x_mean, x_std, x_sum = histNorm_layer(x,
                                          mask_norm,
                                          norm_coef=self.params['norm_coef'],
                                          flag_mask=flag_mask,
                                          flag_norm=self.params['flag_norm'])

        # activation
        if self.params['flag_tanh']: 
            rprint(' Tanh', reuse=False)
            x_norm = tf.tanh(x_norm)

        rprint(' Histogram layer', reuse=False)

        # projective histogram layer
        #x_hist, x_repeat, bin_repeat, theta = projHist_layer(x_norm, 
        x_hist, x_lstm, x_state = projHist_layer(x_norm, 
                                mask_norm, 
                                f'{scope}',
                                num_bins=self.params['n_bins'], 
                                flag_mask=self.params['flag_mask'])
        #Size : [n_sample, n_nodes, featout=fout*nbins]
        #If we do LSTM, [n_sample, n_nodes, fout or some hidden dim]

        #To check
        #self._x_repeat = x_repeat
        #self._bin_repeat = bin_repeat
        #self._x_sum  = x_sum
        
        rprint(' * Variable size after histogram: {}'.format(x_hist.shape), reuse=False)

#         # keep mean and variance? 
        if self.params['flag_stats']=='hist_stats':
            x_hist = tf.concat([x_hist, tf.transpose(x_mean, (0,2,1)), tf.transpose(x_std, (0,2,1))], 2)
            rprint(' * Variable size after adding stats: {}, aggr : {}'.format(x_hist.shape, self.params['flag_stats']), reuse=False)
        elif self.params['flag_stats']=='hist':
            x_hist = x_hist
            rprint(' * Variable size after adding stats: {}, aggr : {}'.format(x_hist.shape, self.params['flag_stats']), reuse=False)
        elif self.params['flag_stats']=='avg':
            x_hist = tf.transpose(x_mean, (0,2,1))
            rprint(' * Variable size after adding stats: {}, aggr : {}'.format(x_hist.shape, self.params['flag_stats']), reuse=False)
        elif self.params['flag_stats']=='sum':
            x_hist = tf.transpose(x_sum, (0,2,1))
            rprint(' * Variable size after adding stats: {}, aggr : {}'.format(x_hist.shape, self.params['flag_stats']), reuse=False)

        if True:
            tf.summary.histogram("output/%s/gout"%scope, x)
            tf.summary.histogram("output/%s/gout_norm"%scope, x_norm)
            tf.summary.histogram("output/%s/hist"%scope, x_hist)
            
        #return x_hist
        return x_hist, x_norm, x, x_lstm, x_state

    def _add_summary(self):
        super()._add_summary()        
        #traiable variables
        for trainVars in tf.trainable_variables():
            tf.summary.histogram("params/%s"%trainVars.name, trainVars, collections=["train"])

    def batch2dict(self, batch, train=True):
        #return L2feeddict(batch[0], self.params['n_vectors'], batch[1], train)
        #return L2feeddict(batch[0], self.params['n_vectors'], y=batch[1], node_attributes=batch[2], train=train)
        return L2feeddict(batch[0], self.params['n_vectors'], y=batch[1], node_attributes=None, train=train)
        




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














