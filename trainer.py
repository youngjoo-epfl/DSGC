"""
Description
[Training class]

"""
import tensorflow as tf

#tf.set_random_seed(1234)

# from model import Model
# import networkx as nx
# import pygsp
# from scipy import sparse
import numpy as np

# from sklearn.metrics import confusion_matrix
# import pickle

from IPython import embed

from tfnntools.nnsystem import NNSystem
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from utils import Dataset, convertToOneHot

class PredictionNNSystem(NNSystem):
    
    def default_params(self):
        d_param = super().default_params()
        d_param['average_validation'] = 1
        return d_param

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_loss = tf.placeholder(tf.float32, name='train_loss')
        self._train_acc = tf.placeholder(tf.float32, name='train_accuracy')
        self._train_f1 = tf.placeholder(tf.float32, name='train_f1')
        tf.summary.scalar("train/loss", self._train_loss, collections=["wtf"])
        tf.summary.scalar("train/accuracy", self._train_acc, collections=["wtf"])
        tf.summary.scalar("train/f1", self._train_f1, collections=["wtf"])
        self._summaries_train = tf.summary.merge(tf.get_collection("wtf"))
        
        self._test_loss = tf.placeholder(tf.float32, name='test_loss')
        self._test_acc = tf.placeholder(tf.float32, name='test_accuracy')
        self._test_f1 = tf.placeholder(tf.float32, name='test_f1')
        tf.summary.scalar("test/loss", self._test_loss, collections=["test"])
        tf.summary.scalar("test/accuracy", self._test_acc, collections=["test"])
        tf.summary.scalar("test/f1", self._test_f1, collections=["test"])
        self._summaries_test = tf.summary.merge(tf.get_collection("test"))


        self._validation_loss = tf.placeholder(tf.float32, name='validation_loss')
        self._validation_acc = tf.placeholder(tf.float32, name='validation_accuracy')
        self._validation_f1 = tf.placeholder(tf.float32, name='validation_f1')
        tf.summary.scalar("validation/loss", self._validation_loss, collections=["validation"])
        tf.summary.scalar("validation/accuracy", self._validation_acc, collections=["validation"])
        tf.summary.scalar("validation/f1", self._validation_f1, collections=["validation"])
        self._summaries_validation = tf.summary.merge(tf.get_collection("validation"))

        #Need to add global step for lr schedule
        #self._global_step = tf.Variable(0, trainable=False)

    def train(self, dataset_train, dataset_validation, dataset_test, **kwargs):
        self._max_f1 = 0
        self._min_loss = np.inf
        self._best_loss_checkpoint = None
        self._best_f1_checkpoint = None
        self._currbatch = 0
        self._validation_dataset = dataset_validation
        self._test_dataset = dataset_test
        self._train_dataset = dataset_train
        return super().train(dataset_train, **kwargs)
        
    def predict(self, **kwargs):
        out = self.outputs(**kwargs)
        return np.argmax(out)
    
    def predict_dataset(self, dataset, sess=None, checkpoint=None, savefeat=False):
        # Start a session only once
        if sess is None:
            self._sess = tf.Session()
            # load the latest checkpoint
            self.load(sess=self._sess, checkpoint=checkpoint)

        # Compute error for the training set
        pred = []
        labels = []
        loss = []
        #for saving node, graph embeddings
        node_embedding = []
        graph_embedding = []
        graph_info = []
        pred_info = []
        gt_info = []

        for batch in dataset.iter():
#             feed_dict = self.net.batch2dict(batch, train=False)
#             output = self.predict(sess=self._sess,**feed_dict)
            p = []
            l = []
            for _ in range(self.params['average_validation']):
                feed_dict = self.net.batch2dict(batch, train=False)
#                 p.append(self.outputs(sess=self._sess,**feed_dict))
                #o = self.eval([self.net.logits, self.net.loss, self.net.node_attributes], sess=self._sess, **feed_dict)
                o = self.eval([self.net.logits, self.net.loss, 
                    self.net.embedding_node,self.net.converted, self.net._x_e1,self.net._x_e2, 
                    self.net.embedding_graph, self.net._x_g1, self.net._x_g2], sess=self._sess, **feed_dict)
                
                #o = self.eval([self.net.logits, self.net.loss], sess=self._sess, **feed_dict)
                #embed()
                #raise ValueError('stop')
                p.append(o[0])
                l.append(o[1])
#             print(np.concatenate(p, axis=0))
            output = np.argmax(np.mean(np.concatenate(p, axis=0),axis=0))
#             print(np.mean(np.concatenate(p, axis=0),axis=0))
            pred.append(output)
            loss.append(np.mean(np.array(l)))
            labels.append(np.argmax(batch[1]))
            
            if savefeat:
                #embed()
                node_embedding.append(o[3]) # save [numsam, featout, numbin]
                graph_embedding.append(o[6])
                graph_info.append(batch[0])
                pred_info.append(output)
                gt_info.append(np.argmax(batch[1]))

        if sess is None:
            self._sess.close()
        pred = np.array(pred)
        labels = np.array(labels)
        loss = np.array(loss)
        saveinfo = (node_embedding, graph_embedding, graph_info, pred_info, gt_info)
        return pred, labels, loss, saveinfo

    def _train_log(self, feed_dict=dict()):
        super()._train_log(feed_dict)
        loss = 0
        pred_class, true_class, loss, _ = self.predict_dataset(self._validation_dataset, sess=self._sess)
        accuracy = 100 * accuracy_score(true_class, pred_class)
        f1 = 100 * f1_score(true_class, pred_class, average='weighted')
        
        loss = np.mean(loss)
        if f1>self._max_f1:
            self._max_f1 = f1
            self._save()
            self._best_f1_checkpoint = self._counter
            print('Best F1 score obtained!')
        
        if loss < self._min_loss:
            self._min_loss = loss
            self._save()
            self._best_loss_checkpoint = self._counter
            print('Best loss score obtained!')


        print("Validation loss: {}".format(loss))
        print("Average accuracy: {:.4}%  / F1 score : {:.4}% ".format(accuracy,f1))
        feed_dict[self._validation_loss] = loss
        feed_dict[self._validation_acc] = accuracy
        feed_dict[self._validation_f1] = f1
        summary = self._sess.run(self._summaries_validation, feed_dict=feed_dict)
        self._summary_writer.add_summary(summary, self._counter)

        pred_class, true_class, loss, _ = self.predict_dataset(self._test_dataset, sess=self._sess)
        accuracy = 100 * accuracy_score(true_class, pred_class)
        f1 = 100 * f1_score(true_class, pred_class, average='weighted')
        
        loss = np.mean(loss)
        print("Test loss: {}".format(loss))
        print("Average accuracy: {:.4}%  / F1 score : {:.4}% ".format(accuracy,f1))
        feed_dict[self._test_loss] = loss
        feed_dict[self._test_acc] = accuracy
        feed_dict[self._test_f1] = f1
        summary = self._sess.run(self._summaries_test, feed_dict=feed_dict)
        self._summary_writer.add_summary(summary, self._counter)
        
        pred_class, true_class, loss, _ = self.predict_dataset(self._train_dataset, sess=self._sess)
        accuracy = 100 * accuracy_score(true_class, pred_class)
        f1 = 100 * f1_score(true_class, pred_class, average='weighted')
        
        loss = np.mean(loss)
        print("Train loss: {}".format(loss))
        print("Average accuracy: {:.4}%  / F1 score : {:.4}% ".format(accuracy,f1))
        feed_dict[self._train_loss] = loss
        feed_dict[self._train_acc] = accuracy
        feed_dict[self._train_f1] = f1
        summary = self._sess.run(self._summaries_train, feed_dict=feed_dict)
        self._summary_writer.add_summary(summary, self._counter)


    def _add_optimizer(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            learning_rate = self._params['optimization']['learning_rate']
            kwargs = self._params['optimization']['kwargs']
            #self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, **kwargs)
            #embed()
            #raise ValueError('optim')
            # get all trainable variables
            t_vars = tf.trainable_variables()
            # create a copy of all trainable variables with `0` as initial values
            accum_tvars = [tf.Variable(tf.zeros_like(tv.initialized_value()),trainable=False) for tv in t_vars]
            # create a op to initialize all accums vars
            self._zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_tvars]
            
            # compute gradients for a batch
            batch_grads_vars = self._optimizer.compute_gradients(self._net.loss, t_vars)
            # collect the batch gradient into accumulated vars
            self._accum_ops = [accum_tvars[i].assign_add(batch_grad_var[0]) for i, batch_grad_var in enumerate(batch_grads_vars)]
            
            new_vars = [(accum_tvars[i], batch_grad_var[1]) for i, batch_grad_var in enumerate(batch_grads_vars)]
            # apply accums gradients 
            self._optimize = self._optimizer.apply_gradients(new_vars)
            
            #tf.summary.scalar("training/loss", self._net.loss, collections=["train"])
        
        # Gradient norm summary
        with tf.name_scope('gradients'):
            for g,v in new_vars:
                tf_last_grad_norm = tf.sqrt(tf.reduce_mean(g**2))
                name = v.name.split('/')[-1]
                name = name.replace(':','')
                tf_gradnorm_summary = tf.summary.scalar('grad_norm_{}'.format(name), tf_last_grad_norm,collections=["train"])

    def _run_optimization(self, feed_dict, idx, epoch):
        if idx==0:
            self._epoch_loss = 0
        if self._currbatch == 0:
            # initialize the accumulated gards
            self._sess.run(self._zero_ops)
        #Weight decaying!!
        #lr = self._params['optimization']['learning_rate']
        #lr = tf.convert_to_tensor(self._params['optimization']['learning_rate'])
        #new_lr = tf.compat.v1.train.exponential_decay(lr, epoch,
        #                                              decay_steps=10,
        #                                              decay_rate=0.98,
        #                                              staircase=False)
        
        #self._params['optimization']['learning_rate'] = self._sess.run(new_lr)
        #global_step = self._sess.run(self._global_step)
        #if global_step%10000 == 0:
        #if tf.math.equal(lr, self._params['optimization']['learning_rate']):
            #lr = self._sess.run(self._params['optimization']['learning_rate'])
            #print('Learning rate chages to : %f, step = %d'%(lr, global_step))
        
        curr_loss = self._sess.run([self.net.loss, self._accum_ops], feed_dict)[0]
        #check global_step
        #print(curr_loss)
        #print(self._sess.run(self._global_step))
        #tf.assign_add(self._global_step, 1)
        #self._global_step += 1

        if self._currbatch < self.params['optimization']['batch_size']-1:
            self._currbatch += 1
        else:
            self._sess.run(self._optimize)
            self._currbatch = 0
        self._epoch_loss += curr_loss
        return curr_loss
    @property
    def max_f1(self):
        return self._max_f1

    @property
    def best_checkpoint(self):
        return self._best_f1_checkpoint, self._best_loss_checkpoint
    

class KFoldTrainer(object):
    #def __init__(self, system, graphs, labels, name='kfold', dbname='MUTAG', n_folds=10):
    def __init__(self, system, graphs, labels, node_attributes=None, name='kfold', dbname='MUTAG', n_folds=10):
        self._system = system
        self._graphs = np.array(graphs)
        if node_attributes is not None:
            self._node_attributes = np.array(node_attributes)
        else:
            self._node_attributes = None
        self._labels = labels
        self._N = len(labels) 
        self._name = name
        self._dbname = dbname
        self._n_folds = n_folds
        self._picked_k_fold = self.draw_kfold()
        self._save_dir = self.system._params['save_dir']
        self._summary_dir = self.system._params['summary_dir']
        
    def train(self):
        n_folds = self._n_folds
        self.train_accs = np.zeros((n_folds))
        self.val_accs   = np.zeros((n_folds))
        self.test_accs  = np.zeros((n_folds))
        self.train_f1s  = np.zeros((n_folds))
        self.val_f1s    = np.zeros((n_folds))
        self.test_f1s   = np.zeros((n_folds))
        self.max_f1s   = np.zeros((n_folds))
        
        y = convertToOneHot(self._labels)
        for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*self._picked_k_fold)):
            if fold == 1:
                raise ValueError('stop')
            assert set.intersection(set(train_idx), set(test_idx)) == set()
            assert set.intersection(set(train_idx), set(val_idx)) == set()
            assert set.union(set(train_idx), set(test_idx), set(val_idx)) == set(np.arange(self._N))
            if self._node_attributes is not None:
                train_dataset = Dataset(self._graphs[train_idx], y[train_idx], node_attributes=self._node_attributes[train_idx])
                test_dataset  = Dataset(self._graphs[test_idx], y[test_idx], node_attributes=self._node_attributes[test_idx])
                val_dataset   = Dataset(self._graphs[val_idx], y[val_idx], node_attributes=self._node_attributes[val_idx])
                print("we use node attributes")
            else:
                train_dataset = Dataset(self._graphs[train_idx], y[train_idx])
                test_dataset  = Dataset(self._graphs[test_idx], y[test_idx])
                val_dataset   = Dataset(self._graphs[val_idx], y[val_idx])
                print("we don't use node attributes")
            print('Split size')
            print('  - Training   : {}/{}'.format(train_dataset.N, self._N ))
            print('  - Testing    : {}/{}'.format(test_dataset.N, self._N ))
            print('  - Validation : {}/{}'.format(val_dataset.N, self._N ))
            
            # Change the saved directory
            #embed()
            self.system._params['save_dir'] = self._save_dir + '_{}'.format(fold)
            self.system._params['summary_dir'] = self._summary_dir + '_{}'.format(fold)
            #embed()
            self.system.train(train_dataset, val_dataset, test_dataset, resume=False)
            #embed()
            #raise ValueError('he')
            checkpoint_f1, checkpoint_loss = self.system.best_checkpoint
            # Average over 10 realization to be even more precise.
            self.system.params['average_validation'] = 1
            
            #pred_train, labels_train,_,_,_ = self.system.predict_dataset(train_dataset, checkpoint=checkpoint_loss)
            #pred_valid, labels_valid,_,_,_ = self.system.predict_dataset(val_dataset, checkpoint=checkpoint_loss)
            #pred_test, labels_test,_,_,_ = self.system.predict_dataset(test_dataset, checkpoint=checkpoint_loss)
            #
            #self.train_f1s[fold]  = f1_score(labels_train, pred_train, average='weighted')
            #self.train_accs[fold] = accuracy_score(labels_train, pred_train)

            #self.val_f1s[fold]  = f1_score(labels_valid, pred_valid, average='weighted')
            #self.val_accs[fold] = accuracy_score(labels_valid, pred_valid)

            #self.test_f1s[fold]  = f1_score(labels_test, pred_test, average='weighted')
            #self.test_accs[fold] = accuracy_score(labels_test, pred_test)
            #
            #self.max_f1s[fold] = self.system.max_f1
            #
            #ret = dict()
            #ret['train_accs'] = self.train_accs
            #ret['val_accs']   = self.val_accs 
            #ret['test_accs']  = self.test_accs 
            #ret['train_f1s']  = self.train_f1s
            #ret['val_f1s']    = self.val_f1s 
            #ret['test_f1s']   = self.test_f1s
            #ret['max_f1s']    = self.max_f1s
            #
            #np.savez('result_vis/{}/f1_{}'.format(self._dbname, self._name), **ret, n_folds=self._n_folds, picked_k_fold=self._picked_k_fold)
        

            pred_train, labels_train,_,_ = self.system.predict_dataset(train_dataset, checkpoint=checkpoint_loss)
            pred_valid, labels_valid,_,_ = self.system.predict_dataset(val_dataset, checkpoint=checkpoint_loss)
            pred_test, labels_test,_, save_info = self.system.predict_dataset(test_dataset, checkpoint=checkpoint_loss, savefeat=True)

            #embed()
            
            self.train_f1s[fold]  = f1_score(labels_train, pred_train, average='weighted')
            self.train_accs[fold] = accuracy_score(labels_train, pred_train)

            self.val_f1s[fold]  = f1_score(labels_valid, pred_valid, average='weighted')
            self.val_accs[fold] = accuracy_score(labels_valid, pred_valid)

            self.test_f1s[fold]  = f1_score(labels_test, pred_test, average='weighted')
            self.test_accs[fold] = accuracy_score(labels_test, pred_test)
            
            self.max_f1s[fold] = self.system.max_f1
            
            ret = dict()
            ret['train_accs'] = self.train_accs
            ret['val_accs']   = self.val_accs 
            ret['test_accs']  = self.test_accs 
            ret['train_f1s']  = self.train_f1s
            ret['val_f1s']    = self.val_f1s 
            ret['test_f1s']   = self.test_f1s
            ret['max_f1s']    = self.max_f1s
            ret['save_info'] = save_info
            #ret['node_embedding'] = node_embedding
            #ret['graph_embedding'] = graph_embedding
            #ret['test_idx'] = test_idx
            
            np.savez('result_vis/{}/loss_{}'.format(self._dbname, self._name), **ret, n_folds=self._n_folds, picked_k_fold=self._picked_k_fold)
        
        return ret
            
    
    def draw_kfold(self):
        skf = StratifiedKFold(self._n_folds, shuffle=True, random_state=12345)

        test_indices, train_indices = [], []
        for _, idx in skf.split(np.zeros(self._N), self._labels):
            test_indices.append(idx)

        val_indices = [test_indices[i - 1] for i in range(self._n_folds)]

        for i in range(self._n_folds):
            train_mask = np.ones(self._N, dtype=np.uint8)
            train_mask[test_indices[i]] = 0
            train_mask[val_indices[i]] = 0
            train_indices.append(train_mask.nonzero()[0].reshape(-1))

        return train_indices, test_indices, val_indices
    
    @property
    def system(self):
        return self._system


# def isnotebook():
#     try:
#         shell = get_ipython().__class__.__name__
#         if shell == 'ZMQInteractiveShell':
#             return True   # Jupyter notebook or qtconsole
#         elif shell == 'TerminalInteractiveShell':
#             return False  # Terminal running IPython
#         else:
#             return False  # Other type (?)
#     except NameError:
#         return False      # Probably standard Python interpreter
    
# if isnotebook():
#     print('Executed from notebook')
#     from tqdm import tqdm_notebook as tqdm
# else:
# from tqdm import tqdm

# class Trainer(object):
#     def __init__(self, config):
#         self.config = config

#         # Need to change dataLoader depending on the task
#         self.dataLoader = dataLoader(config)

#         ## Define Model
#         self.model = Model(config)

#         ## model saver / summary writer
#         self.saver = tf.train.Saver()
#         self.model_saver = tf.train.Saver(self.model.model_vars)
#         self.summary_writer = tf.summary.FileWriter(config.save_dir+config.exp_name)

#         sv = tf.train.Supervisor(logdir=config.save_dir+config.exp_name,
#                                  is_chief=True,
#                                  saver=self.saver,
#                                  summary_op=None,
#                                  summary_writer=self.summary_writer,
#                                  save_summaries_secs=300,
#                                  save_model_secs=config.checkpoint_secs,
#                                  global_step=self.model.model_step)
        
#         gpu_options = tf.GPUOptions(
#                 per_process_gpu_memory_fraction=1.0, allow_growth=True)
#         sess_config = tf.ConfigProto(allow_soft_placement=True,
#                                      gpu_options=gpu_options)
#         self.sess = sv.prepare_or_wait_for_session(config=sess_config)
#         #TODO: change tf.train.Saver --> MonitoredTrainingSession : tf.train.Saver is old expression

#     def _get_summary_writer(self, result):
#         if result['step'] % 100 == 0:
#             #every 100 global steps, it writes to summary.
#             #Why it is 100??
#             return self.summary_writer
#         else:
#             return None

#     def train(self):
#         print("[*] Checking if previous run exists in {}".format(self.config.save_dir+self.config.exp_name))
#         latest_checkpoint = tf.train.latest_checkpoint(self.config.save_dir+self.config.exp_name)
#         if tf.train.latest_checkpoint(self.config.save_dir+self.config.exp_name) is not None:
#             print("[*] Saved result exist! Now loading...")
#             #TODO: It restores the initial value only... something weird!!!!
#             self.saver.restore(
#                     self.sess,
#                     latest_checkpoint)
#             print("[*] Loaded previously trained parameters")
#         else:
#             print("[*] No previous saved file")

#         print("[*] Training starts...")
#         self.model_summary_writer = self.summary_writer
#         summary = tf.Summary()
        
#         #The value we want to see are
#         # 1) training loss
#         # 2) training acc
#         # 3) test loss
#         # 4) test acc
#         avg_loss = 1
#         avg_acc = 0
#         summary.value.add(tag="Train/avg_cross_entropy", simple_value=avg_loss)
#         summary.value.add(tag="Train/avg_accuracy", simple_value=avg_acc)
#         summary.value.add(tag="Test/avg_cross_entropy", simple_value=avg_loss)
#         summary.value.add(tag="Test/avg_accuracy", simple_value=avg_acc)

#         ## Training
# #         epoch_range = tqdm(range(int(self.config.num_epochs)), desc="TR[epoch]")
#         epoch_range = range(int(self.config.num_epochs))
#         for n_epoch in epoch_range:
# #             print('Epoch {}/{}'.format(n_epoch, int(self.config.num_epochs)))
#             losses_in_epoch = []
#             avg_acc = 0
#             self.dataLoader.reset()
#             train_range = tqdm(range(int(self.dataLoader.num_batch_tr)), desc="[per batch]")
#             for batchIdx in train_range:
# #             for batchIdx in range(int(self.dataLoader.num_batch_tr)):
#                 graph_x, y = self.dataLoader.next()
            
#                 num_nodes = graph_x[0][0][1]
#                 deltas, setIdx = getSamples(num_nodes, self.config.num_samples)
#                 feed_dict = {
#                     self.model.x: deltas,                  #Test vectors 
#                     self.model.setIdx: setIdx,             #setIdx for mapping sample to node
#                     self.model.num_nodes: num_nodes,       #number of nodes in graph
#                     self.model.L: graph_x[0][0][0],
#                     self.model.y: y}
                
#                 result = self.model.train(self.sess, feed_dict, self.model_summary_writer,
#                                           with_output=True)
                
#                 self.model_summary_writer = self._get_summary_writer(result)
#                 dist = result['output'] # [1, 2]
#                 confusion = confusion_matrix(np.argmax(y,axis=1), np.argmax(dist, axis=1))
#                 avg_acc += np.diag(confusion).sum()
#                 losses_in_epoch.append(result['loss'])


#             avg_loss = np.mean(losses_in_epoch)
#             acc = avg_acc/float(self.dataLoader.num_batch_tr*self.config.batch_size)
# #             epoch_range.set_description('TR[%d] (loss=%g, acc=%g)'%(result['step'], avg_loss, acc))
#             summary.value[0].simple_value = avg_loss
#             summary.value[1].simple_value = acc
 
#             if n_epoch % 5 == 0:
#                 losses_in_epoch = []
#                 avg_acc = 0
#                 self.dataLoader.reset()
#                 test_range = tqdm(range(int(self.dataLoader.num_batch_te)), desc="TE[epoch]")
# #                 test_range = range(int(self.dataLoader.num_batch_te))
#                 for batchIdx in test_range:
#                     graph_x, y = self.dataLoader.next(is_train=False)
                    
#                     # TODO: at test time we should have mode num_samples 
#                     num_nodes = graph_x[0][0][1]
#                     deltas, setIdx = getSamples(num_nodes, self.config.num_samples)
#                     feed_dict = {
#                         self.model.x: deltas,                  #Test vectors 
#                         self.model.setIdx: setIdx,             #setIdx for mapping sample to node
#                         self.model.num_nodes: num_nodes,       #number of nodes in graph
#                         self.model.L: graph_x[0][0][0],
#                         self.model.y: y}

# #                     feed_dict = {
# #                         self.model.x: graph_x[0][0][2],#Test vectors on graph 0 --> need more proper sampling
# #                         self.model.setIdx: graph_x[0][0][3],#setIdx for mapping sample to node
# #                         self.model.num_nodes: graph_x[0][0][1], #number of nodes in graph 0
# #                         self.model.L: graph_x[0][0][0],
# #                         self.model.y: y}

#                     result = self.model.test(self.sess, feed_dict, None, with_output=True)
#                     dist = result['output'] # [batch_size, 2]

#                     confusion = confusion_matrix(np.argmax(y, axis=1), np.argmax(dist, axis=1))
#                     avg_acc += np.diag(confusion).sum()
#                     losses_in_epoch.append(result['loss'])
#                 avg_loss = np.mean(losses_in_epoch)
#                 acc = avg_acc/float(self.dataLoader.num_batch_te*self.config.batch_size)
# #                 test_range.set_description('TE[epoch] (loss=%g, acc=%g)'%(avg_loss, acc))
#                 summary.value[2].simple_value = avg_loss
#                 summary.value[3].simple_value = acc
            
#             #Is epoch = result['step']
#             self.summary_writer.add_summary(summary, result['step']/float(self.dataLoader.num_batch_tr))
#             self.summary_writer.flush()


#             if n_epoch % self.config.save_step == 0:
#                 #TODO: Need to fix global step... it resets to 0 when re-launch
#                 self.saver.save(self.sess, self.config.save_dir+self.config.exp_name, global_step=n_epoch)







