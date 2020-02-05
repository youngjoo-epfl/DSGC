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
from utils import test_resume, make_problem, generate_fakeDB
from sklearn.metrics import accuracy_score, f1_score
from IPython import embed

# Dataset selection
data_name = 'default'

# Load dataset
sizes = [100, 200 ,300]
num_sample = 100000
n_classes = 2

graphs, labels, L = generate_fakeDB(sizes, num_sample, n_classes)

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
params['net']['node_feat_out'] = [32,32,32,32]
params['net']['node_khop'] = [3,3,3,3]
params['net']['graph_feat_out'] = [32,32,16,16]
params['net']['graph_khop'] = [3,3,3,3]
params['net']['norm_coef'] = 1
params['net']['n_bins'] = 16
params['net']['flag_mask'] = True
params['net']['flag_norm'] = True
params['net']['flag_stats'] = 'sum'
params['net']['flag_tanh'] = True
params['net']['laplacian'] = L
#params['net']['node_label_max'] = node_label_max

# EXP name
exp_name  = 'bc%d_bin%d_%s_vis_test_1200epoch'%(params['optimization']['batch_size'], params['net']['n_bins'], params['net']['flag_stats'])
net_name = data_name + ' ' +exp_name

# Save and summaries
params['save_dir'] = '/mnt/scratch/lts2/youngjoo/check_vis1118_NCI/'+net_name # Where to save the model
params['summary_dir'] = '/mnt/scratch/lts2/youngjoo/check_vis1118_NCI/'+net_name # Where to save the tensorboard summaries
params['summary_every'] = 2000 # tensorboard summary every ?? iterations
params['print_every'] = 2000 # print in console every ?? iterations
params['save_every'] = 10000 # save the model every ?? iterations

# save param
#np.savez('result/{}/params_{}'.format(data_name, net_name), **params)

# resume allow to start from the last checkpoint
try_resume = False
resume, params = test_resume(try_resume, params )

# The ValidationNNSystem will build the network
net = PredictionNNSystem(GCNet, params, debug_mode=False)

print(net.params)

n_folds = 10
k_fold_trainer = KFoldTrainer(net, graphs, labels, node_attributes=None, name=net_name, dbname=data_name, n_folds=n_folds)

res = k_fold_trainer.train()
