# Go one folder back because we are in the notebook folder
import sys
#sys.path.insert(0, '../')

# Use only the first GPU of the machine
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

# Load packages
import numpy as np
from model import GCNet
from trainer import PredictionNNSystem, KFoldTrainer
from utils import test_resume, make_problem, load_dataset_from_name
from sklearn.metrics import accuracy_score, f1_score

# Dataset selection
data_name = 'NCI1'

# Load dataset
graphs, labels, node_attributes, node_label_max = load_dataset_from_name(data_name, use_node_labels=False)
from IPython import embed
embed()
raise ValueError('check')
n_classes = len(np.unique(labels))


# dataset_train, dataset_validation = make_problem(graphs, labels)


# Optimization
params = dict()
params['optimization'] = dict()
params['optimization']['learning_rate'] = 0.0001
params['optimization']['batch_size'] = 128
params['optimization']['epoch'] = 1200
params['optimization']['kwargs'] = {'beta1':0.99, 'beta2':0.999}

# Network
params['net'] = dict()
params['net']['n_classes'] = n_classes
params['net']['n_vectors'] = 50
params['net']['node_feat_out'] = [32,32,64,64,32,32]
params['net']['node_khop'] = [5,5,3,3,5,5]
params['net']['graph_feat_out'] = [32,32,16,16]
params['net']['graph_khop'] = [3,3,3,3]
params['net']['norm_coef'] = 1
params['net']['n_bins'] = 16
params['net']['flag_mask'] = True
params['net']['flag_norm'] = True
params['net']['flag_stats'] = 'hist'
params['net']['flag_tanh'] = True
params['net']['node_label_max'] = node_label_max

# EXP name
exp_name  = 'bc%d_tv%d_bin%d_%s_vis_test_1200epoch'%(params['optimization']['batch_size'], params['net']['n_vectors'], params['net']['n_bins'], params['net']['flag_stats'])
#exp_name  = 'bc%d_tv%d_bin%d_%s_correct_xhist'%(params['optimization']['batch_size'], params['net']['n_vectors'], params['net']['n_bins'], params['net']['flag_stats'])
net_name = data_name + ' ' +exp_name

# Save and summaries
params['save_dir'] = '/mnt/scratch/lts2/youngjoo/check_vis1118_NCI/'+net_name # Where to save the model
params['summary_dir'] = '/mnt/scratch/lts2/youngjoo/check_vis1118_NCI/'+net_name # Where to save the tensorboard summaries
params['summary_every'] = 2000 # tensorboard summary every ?? iterations
params['print_every'] = 2000 # print in console every ?? iterations
params['save_every'] = 10000 # save the model every ?? iterations

# save param
np.savez('result_vis/{}/params_{}'.format(data_name, net_name), **params)

# resume allow to start from the last checkpoint
try_resume = False
resume, params = test_resume(try_resume, params )

# The ValidationNNSystem will build the network
net = PredictionNNSystem(GCNet, params, debug_mode=False)

print(net.params)

# net.train(dataset_train, dataset_validation, resume=resume)


# checkpoint = None
 
# pred_train, labels_train = net.predict_dataset(dataset_train, checkpoint=checkpoint)
# pred_valid, labels_valid = net.predict_dataset(dataset_validation, checkpoint=checkpoint)

# f1_train = f1_score(labels_train, pred_train, average='weighted')
# acc_train = accuracy_score(labels_train, pred_train)

# f1_valid = f1_score(labels_valid, pred_valid, average='weighted')
# acc_valid = accuracy_score(labels_valid, pred_valid)

# loss_train = net.loss(dataset_train, checkpoint=checkpoint)
# loss_valid = net.loss(dataset_validation, checkpoint=checkpoint)

# print('Training dataset: acc {}, f1 {}, loss {}'.format(acc_train, f1_train, loss_train))
# print('Validation dataset: acc {}, f1 {}, loss {}'.format(acc_valid, f1_valid, loss_valid))

n_folds = 10
k_fold_trainer = KFoldTrainer(net, graphs, labels, node_attributes=None, name=net_name, dbname=data_name, n_folds=n_folds)

res = k_fold_trainer.train()
