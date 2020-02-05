import numpy as np
import os
import networkx as nx #Version 2.x
from sklearn.model_selection import KFold, train_test_split
import pygsp

import tensorflow as tf
from scipy import sparse
from IPython import embed

import pickle

np.random.seed(1)

class Dataset(object):
    """Dataset class for graph object. 
    
    Currently only support a batch size of 1.
    """
    def __init__(self, graphs, labels, node_attributes=None, shuffle=True):
        self.graphs = graphs
        self.labels = labels
        self.node_attributes = node_attributes
        self._N = len(graphs)
        self._shuffle = shuffle
    def iter(self, batch_size=1):
        return self.__iter__(batch_size)
    def __iter__(self, batch_size=1):
        if self._shuffle:
            indexes = np.random.permutation(self.N)
        else:
            indexes = np.arange(self.N)
        if self.node_attributes is not None:
            return zip(np.expand_dims(self.graphs[indexes],2), self.labels[indexes], self.node_attributes[indexes])
        else:
            return zip(np.expand_dims(self.graphs[indexes],2), self.labels[indexes])
    @property
    def N(self):
        return self._N
    
def L2feeddict(L, signal, y=None, node_attributes=None, train=True, use_all=False):
    """
    [07. 26. 2019 : Need to extend for node attributes]
    """
    feed_dict = dict()
    tfL = tf.SparseTensorValue(
    indices=np.array([L.row, L.col]).T,
    values=L.data,
    dense_shape=L.shape)
    feed_dict['L'] = tfL
    n_nodes = L.shape[0]
    #deltas, setIdx = getSamples(n_nodes, n_nodes, train)
    #deltas, setIdx = getSamples(n_nodes, n_vectors, train)
    #deltas, setIdx, node_attributes = getSamples(n_nodes, n_vectors, train)
    feed_dict['x'] = signal
    #feed_dict['setIdx'] = setIdx

    #This is for node attributes
    #if node_attributes is not None:
    #    feed_dict['node_attributes'] = node_attributes

    if y is not None:
        feed_dict['y'] = np.expand_dims(y, axis=0)
    if node_attributes is not None:
        #Here, the node_attributes should be match with the size of placeholder!!
        feed_dict['node_attributes'] = node_attributes

    return feed_dict


def make_problem(graphs, labels, n_validation=50, seed=0):
    """This function should be improved..."""
    if seed is not None:
        np.random.seed(seed)
        
    # convert label to 1 hot encoding
    y = convertToOneHot(labels)

    # Shuffle
    indexes = np.random.permutation(len(graphs))
    graphs_shuffle = np.array(graphs)[indexes]
    y_shuffle = np.array(y)[indexes]

    dataset_train = Dataset(graphs_shuffle[:-n_validation], y_shuffle[:-n_validation])
    dataset_validation = Dataset(graphs_shuffle[-n_validation:], y_shuffle[-n_validation:])
    return dataset_train, dataset_validation

def test_resume(try_resume, params):
    """ Try to load the parameters saved in `params['save_dir']+'params.pkl',`

        Not sure we should implement this function that way.
    """
    resume = False
    if try_resume:
        params_loaded = try_load_params(params['save_dir'])
        if params_loaded is None:
            print('No resume, the training will start from the beginning!')
        else:
            params = params_loaded
            print('Resume, the training will start from the last iteration!')
            resume = True
    return resume, params

def try_load_params(path):
    try:
        return load_params(path)
    except:
        return None

def load_params(path):
    with open(os.path.join(path,'params.pkl'), 'rb') as f:
        params = pickle.load(f)
    params['save_dir'] = path
    return params

def getSamples(num_nodes, num_samples, train=True):
    '''
    This is a more recent and lightweight version of the getSubSet function.
    To avoid overfitting, a new sample set should be defined every time the network is run.
    
    PARAMETERS
    ----------
    num_nodes: the size of the graph
    num_samples: how many diracs to create
    '''
#     if train:
    if True:
        if num_samples <= num_nodes:
            #randInt = np.random.choice(range(num_nodes), num_samples, replace=False)
            randInt = np.random.choice(range(num_nodes), num_nodes, replace=False) #deterministic
        else:
            randInt = np.random.choice(range(num_nodes), num_samples, replace=True)
            randInt[:num_nodes] = np.arange(num_nodes)

        setIdx = convertToOneHot(randInt, num_classes=num_nodes)        

        tvs = convertToOneHot(randInt, num_classes=num_nodes) #[num_samples, num_node]
        test_vectors = np.expand_dims(np.transpose(tvs, [1,0]),1) #[num_node, num_samples]
        return test_vectors, setIdx


def getLaplacian(nx_graphs, labels, node_label_max=2, verbose=0):
# def getLaplacian(nx_graphs, num_sim, labels, coarsening=False):
    '''
    from nx_graphs, this function extract Laplacian tensor and node informations
    Also, it contains coarsened version of graph seperately.
    the output graph_list contains
    1) Laplacian
    2) Laplacian of coarsened graph
    3) node information
    4) node information of coarsened graph
    5) test vectors
    6) test vectors of coarsened graph
    7) original graph
    8) node attribute.
    As list. We only take into account for the connected graph with node larger than 5.
    '''
    graph_list = []
    label_list = []
    attribute_list = []
    print('I am using Laplacian matrix as L!!!!!!!!!!')
    print('I am using all nodes on dirac')
    for graph, label in zip(nx_graphs, labels):
        #if graph.number_of_nodes() > 5 and nx.is_connected(graph):
            label_list.append(label)
            W = nx.adjacency_matrix(graph)
            G = pygsp.graphs.Graph(W)
            W = W.astype(float)
            d = W.sum(axis=0)
            d += np.spacing(np.array(0, W.dtype))
            d = 1.0 / np.sqrt(d)
            D = sparse.diags(d.A.squeeze(), 0)
            I = sparse.identity(d.size, dtype=W.dtype)
            L_norm = I - D*W*D
            graph_list.append(L_norm.tocoo())
    return graph_list, label_list, attribute_list

def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array. The labels should start from 0 and None 
    negative.
    Input : numpy array type list of labels

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0
    vector = np.array(vector, dtype=int)

    if num_classes is None:
        assert(np.min(vector)==0)
        num_classes = np.max(vector)+1
        assert(len(np.unique(vector))==num_classes)
    else:
        assert(num_classes > 0)
        assert(num_classes >= np.max(vector))

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)

def set_labels(labels):
    '''
    labels should be numpy array that contains index of class
    '''
    temp = np.copy(labels)
    uniq = np.unique(labels)
    str_cls = 0
    for cls in uniq:
        temp[labels==cls] = str_cls
        str_cls += 1
    return temp

def load_data(db_loc, ds_name, use_node_labels=False):
    node2graph = {}
    Gs = []
    node_label_max = 0
    with open("%s/%s/%s_graph_indicator.txt"%(db_loc,ds_name,ds_name),"r") as f:
        c = 1
        for line in f:
            node2graph[c] = int(line[:-1])
            if not node2graph[c] == len(Gs):
                Gs.append(nx.Graph())
            Gs[-1].add_node(c)
            c += 1
    with open("%s/%s/%s_A.txt"%(db_loc,ds_name,ds_name), "r") as f:
        for line in f:
            edge = line[:-1].split(",")
            edge[1] = edge[1].replace(" ", "")
            Gs[node2graph[int(edge[0])]-1].add_edge(int(edge[0]), int(edge[1]))

    if use_node_labels:
        with open("%s/%s/%s_node_labels.txt"%(db_loc,ds_name,ds_name), "r") as f:
            c = 1
            for line in f:
                node_label = int(line[:-1])
                Gs[node2graph[c]-1].node[c]['label'] = node_label
                c += 1

                if node_label_max < node_label:
                    node_label_max = node_label
                    print(node_label_max)

    labels = []
    with open("%s/%s/%s_graph_labels.txt"%(db_loc,ds_name,ds_name), "r") as f:
        for line in f:
            labels.append(int(line[:-1]))
    
    node_label_max = node_label_max + 1
    graph_list, labels, node_attributes = getLaplacian(Gs, labels, node_label_max)

    labels = np.array(labels, dtype = np.int)
    #Read labels and set class starting from 0
    labels = set_labels(labels)
#     labels = convertToOneHot(labels)

    return graph_list, labels, node_attributes, node_label_max


def load_dataset_from_name(name, *args, **kwargs):
    """Load a dataset from its name."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    datapath = os.path.join(dir_path, 'data')
    
    return load_data(datapath, name, *args, **kwargs)


def generate_fakeDB(sizes, num_sample, num_classes):
    """
    Generate fake dataset for testing
    output: graphs - list of graph signal
            labels - list of multi-class label
    """
    #Gen SBM
    num_node = np.array(sizes).sum()
    probs = [[0.25, 0.05, 0.02],
            [0.05, 0.35, 0.07],
            [0.02, 0.07, 0.40]]
    graph = nx.stochastic_block_model(sizes, probs, seed=0)
    W = nx.adjacency_matrix(graph)
    G = pygsp.graphs.Graph(W)
    W = W.astype(float)
    d = W.sum(axis=0)
    d += np.spacing(np.array(0, W.dtype))
    d = 1.0 / np.sqrt(d)
    D = sparse.diags(d.A.squeeze(), 0)
    I = sparse.identity(d.size, dtype=W.dtype)
    L_norm = I - D*W*D
    L = L_norm.tocoo()
    
    graph_signals = np.random.random([num_sample, num_node])
    #Multi-label
    labels = (np.random.random([num_sample, num_classes]) > 0.5 )*1
    #labels = np.float((np.random.random([num_sample, num_classes]) > 0.5 )*1)
    
    #Single-label
    #labels = np.random.choice(num_classes, num_sample)
    #labels = convertToOneHot(labels, num_classes)
    return graph_signals, labels, L
    
