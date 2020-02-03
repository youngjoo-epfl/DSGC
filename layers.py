import sklearn.metrics
import sklearn.neighbors
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance
import numpy as np

import tensorflow as tf
from IPython import embed

def embedding_network(X, L, shape, scope, norm_coef, num_bins, num_nodes, setIdx, flag_mask=True, flag_reuse=False, flag_summary=True, flag_norm=True, flag_stats=True, flag_tanh=True, aggregation='histogram'):
    """    
    PARAMETERS
    ----------
    X: input features TODO: shape?
    L: the graph Laplacian
    norm_coef: how many std-deviations to take into account in the normalization (default: 1)
    num_bins: number of histogram bins
    num_nodes: number of nodes 
    scope: the variable scope in tensorflow
    setIdx: the ids of the nodes for which to compute the embedding
    
    FLAGS
    -----
    flag_reuse: Do you wish to reuse the same mask? (default=True)
    flag_mask: Do you wish to mask out the values (zeros) of nodes outside the receptive field when computing the local normalization/histogram? (default=True)
    flag_summary: Enable to write summary in the output
    flag_norm: Elable histogram normalization
    flag_stats: add skip connection passing the mean/std to the histogram output
    flag_tanh: tanh normalization after histogram
    flag_aggregation: option for aggregation : histogram or sum or avg
    
    TODOs
    -----
    * isn't num_nodes unecessary, since it can be found from L?
    """
    
    # graph convolutional layer 1 
    print('    Variable size before convolution 1: {}'.format(X.shape))
    gout = graphConv_layer(X, L, shape, flag_reuse, f'{scope}_{1}')
    print('    Variable size after convolution 1: {}'.format(gout.shape))

    gout = tf.transpose(gout, perm=[1,2,0]) 
    print('    Variable size after reshape 1: {}'.format(gout.shape))

    # relu 
    gout = tf.nn.relu(gout)
    
    # graph convolutional layer 2
    shape[1] = shape[2]
    gout = graphConv_layer(gout, L, shape, flag_reuse, f'{scope}_{2}')

    print('    Variable size after convolution 2: {}'.format(X.shape))

    # get receptive field masks
    mask, mask_norm = getReceptiveField(gout)
        
    # graph normalization layer
    gout_norm, gout_mean, gout_std, gout_sum = histNorm_layer(gout, mask_norm, norm_coef=norm_coef, flag_mask=flag_mask, flag_norm=flag_norm)

    print('    Variable size after normalization: {}'.format(gout_norm.shape))

    # activation
    if flag_tanh: 
        gout_norm = tf.tanh(gout_norm)

    print('    Variable size before histogram: {}'.format(gout_norm.shape))
   
    # projective histogram layer
    hist = projHist_layer(gout_norm, mask_norm, num_nodes, num_bins=num_bins, flag_mask=flag_mask)

    print('    Variable size after histogram: {}'.format(hist.shape))

    if aggregation == 'histogram':
        embedding = hist
    elif aggregation == 'avg':
        embedding = tf.transpose(gout_mean, (0,2,1))
    elif aggregation == 'sum':
        embedding = tf.transpose(gout_sum, (0,2,1))
    else:
        raise NotImplementedError('NOT support except for the histogram, avg and sum option in aggregatation')

    #This is previous method only using hist and gout_mean std..
    ## keep mean and variance? 
    #if not flag_stats:
    #    gout_mean *= 0
    #    gout_std *= 0
    #    
    #
    #
    #embedding = tf.concat([hist, tf.transpose(gout_mean, (0,2,1)), tf.transpose(gout_std, (0,2,1))], 2)
    #print('    Variable size after embedding: {}'.format(embedding.shape))
    
    # lift to a tensor of num_nodes size (when performing the computation for a node subset)
    em_convert = convertToGraph(embedding, setIdx)
    print('    Variable size after convert to graph: {}'.format(em_convert.shape))

    if flag_summary:
        tf.summary.histogram("output/%s/gout"%scope, gout)
        tf.summary.histogram("output/%s/gout_norm"%scope, gout_norm)
        tf.summary.histogram("output/%s/hist"%scope, hist)
        
    return embedding, em_convert

def getReceptiveField(x, epsilon=1e-8):
    '''
    INPUT
    -----
    x : The output of the last GCN layer [num_sample, num_nodes, feat_out]
    
    OUTPUT
    ------
    mask: binary mask, size [num_sample, num_nodes]
    mask_norm: mask with entries normalized by 1/num_nodes of size [num_sample, num_nodes]
    '''
    mask = tf.abs(x)
    mask = tf.reduce_sum(mask, 2)
    mask = tf.cast(mask, dtype=tf.bool)          # --> [num_sample, num_nodes]

    mask_norm = tf.cast(mask, dtype=tf.float32)
    mask_norm = mask_norm / (tf.reduce_sum(mask_norm,1,keepdims=True)+np.float32(epsilon))
        
    return mask, mask_norm
    
## Define graph convolution
def graphConv_layer(x, L, shape, flag_reuse, scope, scale=0.8, epsilon=1e-8, type='chev', **kwards):
    '''
    INPUT
    -----
    x : [num_nodes, feat_in, num_sample] size tensor. num_node can be None type
    L : sparse matrix of laplacian with normalized and scaled
    gweights : [feat_in*khops, feat_out] size weight matrix
    
    OUTPUT
    -------
    out: [num_sample, num_nodes, feat_out] 
    '''
    khops, feat_in, feat_out = shape
    
    def scale_operator(L, lmax=2, scale=scale):
        r"""Scale the eigenvalues from [0, lmax] to [-scale, scale].
        
        Here lmax=2 because we use the normalized laplacian
        """
        I = tf.sparse.eye(tf.shape(L)[0], dtype=tf.float32)
        L *= 2 * scale / (lmax+np.float32(epsilon))
        L = tf.sparse.add(L, I*(-scale))
        return L

    L_scale = scale_operator(L)
    with tf.variable_scope(scope, flag_reuse) as sc:
        gweights = tf.get_variable("{}_gweights".format(scope), shape=[feat_in*khops, feat_out],
                initializer=cheby_initializer(feat_out, scale=scale, activation='tanh', order=khops))
                            # tf.glorot_normal_initializer()        
        gbias   = tf.get_variable("{}_gbias".format(scope), shape=[feat_out],
                initializer=tf.zeros_initializer())
    n_nodes = tf.shape(x)[0]
    feat_in = tf.shape(x)[1]
    n_samples = tf.shape(x)[2]
    output_list = []
    x0 = x
    x0 = tf.reshape(x0, [n_nodes, feat_in*n_samples]) # Is there anyway to merge two dim directly without using reshape?
    output_list.append(x0)
    if khops > 1:
        x1 = tf.sparse_tensor_dense_matmul(L_scale, x0)
        output_list.append(x1)

    for k in range(2, khops):
        x2 = 2 * tf.sparse_tensor_dense_matmul(L_scale, x1) - x0
        output_list.append(x2)
        x0, x1 = x1, x2

    x = tf.stack(output_list) # x is now [khops, num_nodes, feat_in*num_sample] size tensor    
    x = tf.reshape(x, [khops, n_nodes, feat_in, n_samples])
    x = tf.transpose(x, perm=[3,1,2,0]) #num_sample, num_nodes, feat_in, khops=num_masks]
    x = tf.reshape(x, [n_samples, n_nodes, feat_in*khops]) #[num_sample, num_nodes, khops]
    
    # Reshape output
    out = tf.tensordot(x, gweights, axes=[2, 0]) + gbias # out is [num_sample, num_node, feat_out] size tensor

    return out 


def histNorm_layer(x, mask_norm, norm_coef=1.0, axis=1, flag_mask=True, flag_norm=True, epsilon=1e-4):
    """
    INPUT
    -----
    x : input tensor to be normalized. ex) [num_sample, num_nodes, khops]
    mask : graph mask which has same size of x.
    norm_coef : std value
    axis : where the num_nodes located.

    x is the output of graphconv, so that.. it may have non-zero value locally.
    therefore, to get the correct normalization, we need to know the location,
    where the non-zero value exist.
    
    OUPUT
    -----
    x_norm : normalized results
    x_mean : mean value
    x_std  : standard deviation value
    """
    if flag_mask:
        num_sample, _, _ = x.get_shape()
        mask_norm  = tf.expand_dims(mask_norm, 2)
        mask       = tf.cast(mask_norm, dtype=tf.bool)
        mask_float = tf.cast(mask, dtype=tf.float32)
        
        x_mean = tf.reduce_sum(x * mask_norm, axis, keepdims=True)
        #x_mean = tf.reduce_sum(x * mask_norm, axis=1, keepdims=True)
        
        #x_mean_ = tf.reduce_sum(x_mean, axis=0, keepdims=True)
        x_sum = tf.reduce_sum(x * mask_float, axis, keepdims=True)


        x_hat  = (x - x_mean) * mask_float
        x_std = tf.sqrt(np.float32(epsilon) + tf.reduce_sum(x_hat * x_hat * mask_norm, axis, keepdims=True))
        #x_std = tf.reduce_sum(x_hat * x_hat * mask_norm, axis=1, keepdims=True)
        #x_std = tf.sqrt(np.float32(epsilon) + tf.reduce_sum(x_std, axis=0, keepdims=True))
    else:
        num_sample, _, _ = x.get_shape()
        #x_mean = tf.reduce_sum(x, axis=1, keepdims=True)
        #x_mean_ = tf.reduce_sum(x_mean, axis=0, keepdims=True)
        

        x_mean = tf.reduce_mean(x, axis, keepdims=True)

        x_sum = tf.reduce_sum(x, axis, keepdims = True)

        x_hat  = (x - x_mean)
        #x_std = tf.reduce_sum(x_hat * x_hat, axis=1, keepdims=True)
        #x_std = tf.sqrt(np.float32(epsilon) + tf.reduce_sum(x_std, axis=0, keepdims=True))
        x_std = tf.sqrt(np.float32(epsilon) + tf.reduce_mean(x_hat * x_hat, axis, keepdims=True))
    with tf.name_scope('histogram'):
        tf.summary.histogram('x_hat', x_hat)
        tf.summary.histogram('x_std', x_std)
        tf.summary.histogram('x_mean', x_mean)
        tf.summary.histogram('x_input', x)
        tf.summary.histogram('x_sum', x_sum)
    if flag_norm: 
        x = x_hat / (norm_coef * x_std )
        
    return x, x_mean, x_std, x_sum


def convertToGraph(hist, setIdx, dropout_resize=False):
    """
    mapping sample hist to graph Node.
    setIdx knows where the sample comes from.
    """
#     print('    setIdx size : {}'.format(setIdx.shape))
    if len(hist.get_shape()) > 2:
        num_sample, feat_out, bins = hist.get_shape()
        hist = tf.reshape(hist, [-1, feat_out*bins, 1])
    else:
        print('use LSTM feat')
        hist = tf.expand_dims(hist, 2)
    
#     print('    After reshape size : {}'.format(hist.shape))
    conversion = tf.tensordot(setIdx, hist, axes=[0,0])
    if dropout_resize:
        n_nodes = tf.shape(setIdx)[1]
        n_samples = tf.shape(setIdx)[0]
#         n_nodes = tf.reduce_sum(tf.constant(1, dtype=tf.float32, shape=[n_nodes]))
        n_samples = tf.reduce_sum(tf.ones(shape=[n_samples], dtype=tf.float32))
        n_nodes = tf.reduce_sum(tf.ones(shape=[n_nodes], dtype=tf.float32))
        conversion = (conversion/n_nodes)*n_samples

    return conversion




def rbf_activation(x, c, theta):
    #y = -tf.square(x - c)
    y = tf.exp(-tf.square(x - c)/theta)
    #y = x - c
    return y

def smoothRect_activation(x, c, theta, steepness_coef=3):
    return 1 / ( tf.math.pow( 2*(x-c)/theta, 2*steepness_coef) + 1)


#def rbf_activation(x, c, theta):
#     s = tf.nn.relu(np.float32(1) - (tf.abs(x-c)/(2*np.sqrt(theta))))
     #s = -(tf.abs(x-c))+np.float32(1)
#     return s


def projHist_layer(x, mask, scope, num_bins=5, flag_mask=True, epsilon=1e-8):
    '''
    Computes a projective histogram (actually pdf). 
    
    INPUT
    -----
    x: feature tensor of size [num_sample, num_nodes]
    mask_norm: receptive field mask of size [num_sample, num_nodes] normalized to take an average
    
    PARAMETERS
    ----------
    num_bins: number of bins
    
    FLAGS
    -----
    flag_mask: For each bin, aggregate over all or non-zeros? 
    
    OUTPUT
    ------
    hist: tensor of size [num_sample, feat_out, num_bins]
    '''
    
    binIdx = np.linspace(-1, 1, num_bins+1).astype(np.float32)
    #theta = ((binIdx[1] - binIdx[0])/2)**2
    #theta = ((binIdx[1] - binIdx[0])/5)
    theta = 2/(num_bins-1)
    bin_center = (binIdx[:-1] + binIdx[1:])/2
    x_repeat = tf.tile(tf.expand_dims(x,3),[1,1,1,num_bins]) #[num_sample, num_nodes, feat_out, num_bins]
    for i in range(3):
        bin_center = np.expand_dims(bin_center, 0)
    bin_center = tf.convert_to_tensor(bin_center, dtype=tf.float32)
    n_nodes = tf.shape(x)[1]
    n_sample = tf.shape(x)[0]
    bin_repeat = tf.tile(bin_center, [n_sample, n_nodes, x.shape[2], 1])
    #hist_out = rbf_activation(x_repeat, bin_repeat, theta)
    hist_out = smoothRect_activation(x_repeat, bin_repeat, theta, 10)
    #hist_out = x_repeat
    
    # PDF MODE 
    #if flag_mask:
    #    mask_float = tf.cast(mask, dtype=tf.float32)
    #    mask_float = mask_float / (tf.expand_dims(tf.reduce_sum(mask_float,1),1)+np.float32(epsilon))
    #    hist_out = hist_out * tf.expand_dims(tf.expand_dims(mask_float,2),3)
    #    hist = tf.reduce_sum(hist_out, 1) #[feat_out, num_bins]
    #else:
    #    hist = tf.reduce_mean(hist_out, 1) #[feat_out, num_bins]
    
    hist = tf.reduce_sum(hist_out, 1)
#     if False:
#         n_sample = tf.reduce_sum(tf.ones(shape=[n_sample], dtype=tf.float32))
#         hist = hist/n_sample
    
    # COUNT MODE 
    # NOT IMPLEMENTED (it tends to overfit in my experiments)
    
    # LSTM MODE
    # hist_out = [num_sample, num_node, f_out, n_bins] dimension.
    # Need to put LSTM to hist per each num_sample*num_node seperately actually...
    
    #with tf.variable_scope(scope, False):
    #    cell = tf.nn.rnn_cell.LSTMCell(128, forget_bias=1.0)
    #    rnn_input_seq = tf.unstack(hist, num_bins, 2)
    #    output, states = tf.nn.static_rnn(cell, rnn_input_seq, dtype=tf.float32)
    #    print('Use LSTM!!!')

    #return hist, x_repeat, bin_repeat, theta
    #return hist, output, states
    return hist, hist, hist


def cheby_initializer(n_filter, order, activation='relu',scale=1):
    def compute_bound(x, order):
        p = []
        p.append(x**0)
        p.append(x**1)
        for o in range(2,order+1):
            p.append(2*x*p[o-1]-p[o-2])
        return np.sum(np.array(p[0:o])**2,axis=0)
    bound = compute_bound(np.arange(-scale,scale,0.01),order)
    if activation == 'relu':
        scale = 2  # from He et al.
        bound = np.max(bound)*0.95
    elif activation in ['linear', 'sigmoid', 'tanh']:
        # sigmoid and tanh are linear around 0
        scale = 1  # from Glorot et al.
        bound = np.mean(bound)
    else:
        raise ValueError('unknown activation')

    std = np.sqrt(scale / (n_filter*bound))
    return tf.initializers.random_normal(0, std, dtype=tf.float32)
            
        
