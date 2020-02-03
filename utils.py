import numpy as np
import os
import networkx as nx #Version 2.x
from sklearn.model_selection import KFold, train_test_split
import pygsp

import tensorflow as tf
from scipy import sparse
from IPython import embed

from libraries.coarsening_utils import *
import libraries.graph_utils

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
            return zip(self.graphs[indexes], self.labels[indexes], self.node_attributes[indexes])
        else:
            return zip(self.graphs[indexes],self.labels[indexes])
    @property
    def N(self):
        return self._N
    
def L2feeddict(L, n_vectors, y=None, node_attributes=None, train=True, use_all=False):
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
    deltas, setIdx = getSamples(n_nodes, n_vectors, train)
    #deltas, setIdx, node_attributes = getSamples(n_nodes, n_vectors, train)
    feed_dict['x'] = deltas
    feed_dict['setIdx'] = setIdx

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
#     else:
#         test_vectors = np.expand_dims(np.eye(num_nodes),1)
#         setIdx = convertToOneHot(np.arange(num_nodes), num_classes=num_nodes)   
#         return test_vectors, setIdx

# def getSamples(num_nodes, num_samples, train=True):
#     '''
#     This is a more recent and lightweight version of the getSubSet function.
#     To avoid overfitting, a new sample set should be defined every time the network is run.
    
#     PARAMETERS
#     ----------
#     num_nodes: the size of the graph
#     num_samples: how many diracs to create
#     '''
# #     if train:
# #         num_samples = num_nodes
# #         if num_samples <= num_nodes:
# #             randInt = np.random.choice(range(num_nodes), num_samples, replace=False)
# #         else:
# #             randInt = np.random.choice(range(num_nodes), num_samples, replace=True)
# #             randInt[:num_nodes] = np.arange(num_nodes)

# #         setIdx = convertToOneHot(randInt, num_classes=num_nodes)        

# #         tvs = convertToOneHot(randInt, num_classes=num_nodes) #[num_samples, num_node]
# #         test_vectors = np.expand_dims(np.transpose(tvs, [1,0]),1) #[num_node, num_samples]
# #         return test_vectors, setIdx
# #     else:
#     if True:
#         test_vectors = np.expand_dims(np.eye(num_nodes),1)
#         setIdx = convertToOneHot(np.arange(num_nodes), num_classes=num_nodes)   
#         return test_vectors, setIdx

    
# def getSubSet(num_nodes, num_sim, graph=None, set_type='dirac'):
#     '''    
#     @ DEPRECATED FUNCTION
#     Generate Testvectors
#     generate [num_sim, num_node] size numpy ndarray 
#     num_sim : number of test vectors 
    
#     set_type : 'dirac'
#     each test vector has 1-hot vector on N-nodes.
    
#     set_type : 'edge'
#     select random edge that has value on two connected node

#     set_type: 'khop' [Not yet implemented]
#     select random node and take k-hop neighbors 
    
#     TODO 
#     * rename num_sim to num_samples
#     * handle case  num_sim > num_nodes better
#     '''
#     if set_type == 'dirac':
#         if num_sim <= num_nodes:
#             randInt = np.random.choice(range(num_nodes), num_sim, replace=False)
#             setIdx = convertToOneHot(randInt, num_classes=num_nodes)
#         else:
#             #want to equaly distributes the testvector on each node.
#             randInt1 = np.random.choice(range(num_nodes), num_nodes, replace=False)
#             setIdx = randInt1
#             setIdx = convertToOneHot(randInt1, num_classes=num_nodes)
#             #pad zeros
#             padding = np.zeros([num_sim-num_nodes, num_nodes])
#             setIdx = np.concatenate([setIdx, padding])
#             randInt1 = np.expand_dims(randInt1, 1)
#             randInt1 = np.repeat(randInt1, num_sim/num_nodes, 1)
#             randInt1 = np.reshape(randInt1.transpose(), [-1])
#             randInt2 = np.random.choice(range(num_nodes), num_sim%num_nodes)
#             randInt = np.concatenate([randInt1, randInt2])

#         tvs = convertToOneHot(randInt, num_classes=num_nodes) #[num_sim, num_node]
    
#     elif set_type == 'edge':
#         num_nodes = graph.number_of_nodes()
#         A = nx.adjacency_matrix(graph)
#         pygspGraph = pygsp.graphs.Graph(A)
#         v_in, v_out, e_weight = pygspGraph.get_edge_list()
#         edge_list = []
#         for i,j in zip(v_in, v_out):
#             edge = np.zeros([pygspGraph.N])
#             edge[i] = 0.5
#             edge[j] = 0.5
#             edge_list.append(edge)
#         tvs = np.array(edge_list) #[num_edges, num_node] --> sample
    
#         num_edges = nx.number_of_edges(graph)
#         if num_sim <= num_edges:
#             randInt_edge = np.random.choice(range(num_edges), num_sim, replace=False)
#         else:
#             randInt1 = np.random.choice(range(num_edges), num_edges, replace=False)
#             randInt1 = np.repeat(randInt1, num_sim/num_edges, 0)
#             randInt2 = np.random.choice(range(num_edges), num_sim%num_edges)
#             randInt_edge = np.concatenate([randInt1, randInt2])
    
#         tvs = edge_list[randInt_edge] #[num_sim, num_node]
#     else:
#         raise NotImplementedError("Use dirac or edge")
    
#     test_vectors = np.expand_dims(np.transpose(tvs, [1,0]),1) #[num_node, num_sim
#     return test_vectors, setIdx



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
            
            # Use Adjacency as L
            #L_norm = W
            

            # L is scaled iniside the network
#             L_scaled = L_norm - I
#             L = L_scaled.tocoo()
#             indices = np.column_stack((L.row, L.col))
#             L_tensor = tf.SparseTensorValue(indices, L.data, L.shape)
            graph_list.append(L_norm.tocoo())
            #Node attributes
            #node_attribute = [*nx.get_node_attributes(graph, 'label').values()]
            #Convert to onehot
            #node_attribute = convertToOneHot(np.array(node_attribute), node_label_max)
            #attribute_list.append(np.array(node_attribute))
            
            #make deltas and Set "Here we define set"
#             num_nodes = graph.number_of_nodes()
#             deltas, setIdx = getSubSet(num_nodes, num_sim, graph, set_type="dirac")

#             if coarsening :
#                 #Get Coarsened version
#                 C, Gc, Call, Gall = coarsen(G, K=5, r=0.5, method='variation_neighborhood')
#                 G_coarsen = Gall[-1]
#                 G_coarsen.compute_laplacian('normalized')
#                 L_coarsen = G_coarsen.L.tocoo()
#                 indices_c = np.column_stack((L_coarsen.row, L_coarsen.col))
#                 L_tensor_c = tf.SparseTensorValue(indices_c, L_coarsen.data, L_coarsen.shape)
#                 num_nodes_c = G_coarsen.N
#                 deltas_c, setIdx_c  = getSubSet(num_nodes_c, num_sim, Gc)
#                 graph_list.append(((L_tensor, num_nodes, deltas, setIdx, graph, node_attribute),(L_tensor_c, num_node_c, deltas_c, setIdx_c)))
#             else:
#             graph_list.append(((L_tensor, num_nodes, deltas, setIdx, graph, node_attribute),(None)))
           
        #else:
        #    if verbose:
        #        print("[!] Disconnected graph sample --> discard!")
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


# class dataLoader(object):
#     """
#     Provide data to for feed_dict.
#     graphs and labels are list type of data.
#     """
#     def __init__(self, config):
#         self.config = config
#         #Load data
#         use_node_label = False
#         graphs, labels = load_data(config.data_dir, config.data_name, config.num_samples, use_node_label)

#         #Split dataset into train and test --> further needs to be 10-CV
#         train_data, test_data, train_label, test_label = train_test_split(graphs, labels,random_state=1,
#                                                                           test_size=config.test_ratio, shuffle=True)

#         self.train_data = train_data
#         self.test_data = test_data
#         self.train_label = train_label
#         self.test_label = test_label

#         self.num_tr = len(train_data)
#         self.num_te = len(test_data)
#         self.batch_size = config.batch_size
#         self.batch_size_test = config.batch_size_test
#         self.num_batch_tr = self.num_tr/config.batch_size
#         self.num_batch_te = self.num_te/config.batch_size_test
#         self.curr_p = 0

#     def reset(self):
#         self.curr_p =0

# #     def reshuffle(self):
# #         np.random.shuffle(self.train_data)


#     def __iter__(self):
#         return self

#     def __next__(self, is_train=True):
#         if is_train:
#             next_p = self.curr_p + self.batch_size
#             if next_p > self.num_tr:
#                 self.reset()
# #                 self.reshuffle()
#                 next_p = self.curr_p + self.batch_size

#             data = self.train_data[self.curr_p:next_p]

#             label = self.train_label[self.curr_p:next_p]
#             self.curr_p += self.batch_size

#         else:
#             next_p = self.curr_p + self.batch_size_test
#             if next_p > self.num_te:
#                 self.reset()
#                 next_p = self.curr_p + self.batch_size_test

#             data = self.test_data[self.curr_p:next_p]
#             label = self.test_label[self.curr_p:next_p]
#             self.curr_p += self.batch_size_test
#         return data, label

#     next = __next__


def load_dataset_from_name(name, *args, **kwargs):
    """Load a dataset from its name."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    datapath = os.path.join(dir_path, 'data')
    
    return load_data(datapath, name, *args, **kwargs)
    
