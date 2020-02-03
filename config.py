#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Description :
# configuration for setting the hyper-parameters and other params.

import argparse
parser = argparse.ArgumentParser()
arg_list = []

def str2bool(v):
    return v.lower() in ('true', '1', 'yes', 't', 'y')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_list.append(arg)
    return arg

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_dir', type=str, default='../data/')
data_arg.add_argument('--data_name', type=str, default="MUTAG")
data_arg.add_argument('--nclasses', type=int, default=2)

# Graph parameters
graph_arg = add_argument_group('Graph')
graph_arg.add_argument('--khop', type=int, nargs='+', default=[4, 4])
graph_arg.add_argument('--feat_in', type=int, default=1)
graph_arg.add_argument('--feat_out', type=int, nargs='+',  default=[16,32])

#Model params
model_arg = add_argument_group('Model')
model_arg.add_argument('--num_layers', type=int, nargs='+', default=[1,2])
model_arg.add_argument('--norm_coef', type=float, default=1.0)
model_arg.add_argument('--num_bins', type=int, default=10)
model_arg.add_argument('--num_samples', type=int, default=50) 
model_arg.add_argument('--flag_norm', type=str2bool, default=True)
model_arg.add_argument('--flag_stats', type=str2bool, default=True)
model_arg.add_argument('--flag_tanh', type=str2bool, default=True)
model_arg.add_argument('--flag_mask', type=str2bool, default=True)
model_arg.add_argument('--flag_pdf', type=str2bool, default=False)

# Train params
train_arg = add_argument_group('Train')
train_arg.add_argument('--test_ratio', type=float, default=0.25)
train_arg.add_argument('--num_epochs', type=int, default=500)
train_arg.add_argument('--batch_size', type=int, default=1) #Should be possible for batch learning!!
train_arg.add_argument('--batch_size_test', type=int, default=1) #Should be possible for batch learning!!

# train_arg.add_argument('--margin', type=float, default=1.0) # NOT USED ANY MORE 
# train_arg.add_argument('--task', type=str, default='classifier', choices=['classifier', 'siamese']) # NOT USED ANY MORE ??
# train_arg.add_argument('--embedding_space', type=int, default=50) # NOT USED ANY MORE ??
# train_arg.add_argument('--kernel_width', type=float, default=2.0) # NOT USED ANY MORE ??
# train_arg.add_argument('--embedding_dim', type=int, default=10) # NOT USED ANY MORE ??


# Optimization param
optim_arg = add_argument_group('Optim')
optim_arg.add_argument('--learning_rate', type=float, default=0.001)
optim_arg.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'rmsprop'])
optim_arg.add_argument('--beta1', type=float, default=0.9) #Momentum
optim_arg.add_argument('--beta2', type=float, default=0.999) #std"
optim_arg.add_argument('--max_grad_norm', type=float, default=3.0)

# Miscellaneous
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--use_coarsen', type=str2bool, default=False)
misc_arg.add_argument('--exp_name', type=str, default='test_exp')
misc_arg.add_argument('--checkpoint_secs', type=int, default=300)
misc_arg.add_argument('--save_step', type=int, default=20, help='save logs per each steps of epoch')
misc_arg.add_argument('--save_dir', type=str, default='../saved_results/')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed






