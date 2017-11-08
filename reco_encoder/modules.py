import os
from datetime import datetime

import glog as log
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from reco_encoder.data import input_layer
from autoencoder.modules import AutoEncoder, SparseBatchAutoEncoder


class AutoEncoderRecommender(object):
  def __init__(self, mode, params):
    self.params = params
    self.mode = mode
    self._model_last_state = None

    if 'model' in self.params:
      self.__load_last_state(self.params['model'])

    self.batch_size = self.params['batch_size']
    self.major = self.params['major']
    self.item_id_ind = self.params['item_id_ind']
    self.user_id_ind = self.params['user_id_ind']
    self.data_delimiter = self.params['data_delimiter']
    self.files_extension = self.params['files_extension']

    if self._model_last_state is None or mode == 'test':
      self.__load_full_data_layer()

    self.__init_model()

    if mode == 'train':
      self.__init_training()
    else:
      self.__init_evaluation()

    self.model_checkpoint = self.params['model_checkpoint'] if 'model_checkpoint' in self.params else 'model/'

    log.info('Hidden layers sizes: {}'.format(self.hidden_layers_sizes))
    log.info('Activation type: {}'.format(self.activation_type))

    if mode == 'train':
      log.info('Learning rate: {}'.format(self.lr))
      log.info('Weight decay: {}'.format(self.weight_decay))
      log.info('Optimizer type: {}'.format(self.optimizer_type))
      log.info('Sampling steps: {}'.format(self.sampling_steps))
      log.info('Walkback steps: {}'.format(self.walkback_steps))
      log.info('Data augmentation probability: {}'.format(self.aug_prob))
      log.info('Data augmentation samples: {}'.format(self.aug_samples))
      log.info('Loss function: {}'.format(self.loss_func_type))
      log.info('Dropout probability: {}'.format(self.dropout_prob))
    else:
      log.info('Loss function: {}'.format(self.loss_func_type))
      log.info('Sampling steps: {}'.format(self.sampling_steps))

    log.info("Total {} found: {}".format(self.major, len(self.train_data_layer.data.keys())))
    log.info("Vector dim: {}".format(self.train_data_layer.vector_dim))

  def __init_model(self):
    if self._model_last_state is None:
      self.hidden_layers_sizes = self.params['hidden_layers_sizes']
      self.activation_type = self.params['activation_type']
      self.last_layer_act = self.params['last_layer_act']
      self.is_constrained = self.params['is_constrained']
      self.dropout_prob = self.params['dropout_prob'] if 'dropout_prob' in self.params else 0
    else:
      self.vector_dim = self._model_last_state['vector_dim']
      self.hidden_layers_sizes = self._model_last_state['hidden_layers_sizes']
      self.activation_type = self._model_last_state['activation_type']
      self.last_layer_act = self._model_last_state['last_layer_act']
      self.is_constrained = self._model_last_state['is_constrained']
      self.dropout_prob = self._model_last_state['dropout_prob']

    self.autoencoder = AutoEncoder(layer_sizes=[self.vector_dim] + [int(l) for l in self.hidden_layers_sizes],
                                   activation_type=self.activation_type,
                                   is_constrained=self.is_constrained,
                                   dp_drop_prob=self.dropout_prob,
                                   last_layer_act=self.last_layer_act)

    if not self._model_last_state is None:
      self.autoencoder.load_state_dict(self._model_last_state['model'])

  def __init_training(self):
    self.__load_train_eval_data_layers()
    self.lr = self.params['lr']
    self.weight_decay = self.params['weight_decay']
    self.num_epochs = self.params['num_epochs']
    self.optimizer_type = self.params['optimizer_type']
    self.loss_func = self.params['loss_func']
    self.loss_func_type = self.loss_func.__class__.__name__
    self.aug_prob = self.params['aug_prob'] if 'aug_prob' in self.params else 0
    self.aug_samples = self.params['aug_samples'] if 'aug_samples' in self.params else 1
    self.aug_full = self.params['aug_full'] if 'aug_full' in self.params else False
    self.summary_frequency = self.params['summary_frequency'] if 'summary_frequency' in self.params else 10
    self.eval_freq = self.params['eval_freq'] if 'eval_freq' in self.params else 5
    self.optimizer_milestones = self.params['optimizer_milestones'] if 'optimizer_milestones' in self.params else []
    self.sampling_steps = self.params['sampling_steps'] if 'sampling_steps' in self.params else 1
    self.walkback_steps = self.params['walkback_steps'] if 'walkback_steps' in self.params else 0
    self.finetune = self.params['finetune'] if 'finetune' in self.params else False
    self.denoise = self.params['denoise'] if 'denoise' in self.params else None

    self.__init_optimizer()
    self.current_epoch = 0

    if not self._model_last_state is None and not self.finetune:
      self.optimizer.load_state_dict(self._model_last_state['optimizer'])
      self.current_epoch = self._model_last_state['last_epoch'] + 1

    self.lr_scheduler = MultiStepLR(self.optimizer, milestones=self.optimizer_milestones, gamma=0.1,
                                    last_epoch=self.current_epoch - 1)
    self.current_epoch = 0

  def __init_evaluation(self):
    self.__load_train_eval_data_layers()
    self.loss_func_type = self.params['loss_func']
    self.plot_roc = self.params['plot_roc'] if 'plot_roc' in self.params else False
    self.sampling_steps = self.params['sampling_steps'] if 'sampling_steps' in self.params else 1

  def __load_full_data_layer(self):
    self.path_to_full_train_data = self.params['path_to_full_train_data']

    full_params = dict()
    full_params['batch_size'] = self.batch_size
    full_params['data_dir'] = self.path_to_full_train_data
    full_params['major'] = self.major
    full_params['itemIdInd'] = self.item_id_ind
    full_params['userIdInd'] = self.user_id_ind
    full_params['delimiter'] = self.data_delimiter
    full_params['extension'] = self.files_extension
    log.info("Loading full training data from: {}".format(self.path_to_full_train_data))
    self.full_data_layer = input_layer.UserItemRecDataProvider(params=full_params)
    self.user_id_map = self.full_data_layer.userIdMap
    self.item_id_map = self.full_data_layer.itemIdMap

    self.vector_dim = self.full_data_layer.vector_dim

    if self.mode != 'test':
      del self.full_data_layer

  def __load_train_eval_data_layers(self):
    if not self._model_last_state is None:
      self.user_id_map = self._model_last_state['user_id_map']
      self.item_id_map = self._model_last_state['item_id_map']
    self.path_to_train_data = self.params['path_to_train_data']
    self.path_to_eval_data = self.params['path_to_eval_data']
    log.info("Loading training data from: {}".format(self.path_to_train_data))
    train_params = {}
    train_params['batch_size'] = self.batch_size
    train_params['data_dir'] = self.path_to_train_data
    train_params['major'] = self.major
    train_params['itemIdInd'] = self.item_id_ind
    train_params['userIdInd'] = self.user_id_ind
    train_params['delimiter'] = self.data_delimiter
    train_params['extension'] = self.files_extension
    train_params['data_dir'] = self.path_to_train_data
    self.train_data_layer = input_layer.UserItemRecDataProvider(params=train_params,
                                                                user_id_map=self.user_id_map,
                                                                item_id_map=self.item_id_map)

    log.info("Loading eval data from: {}".format(self.path_to_eval_data))
    eval_params = dict(train_params)
    eval_params['data_dir'] = self.path_to_eval_data
    self.eval_data_layer = input_layer.UserItemRecDataProvider(params=eval_params,
                                                               user_id_map=self.train_data_layer.userIdMap,
                                                               item_id_map=self.train_data_layer.itemIdMap)

  def __load_test_data_layer(self):
    test_params = {}
    test_params['batch_size'] = self.batch_size
    test_params['major'] = self.major
    test_params['itemIdInd'] = self.item_id_ind
    test_params['userIdInd'] = self.user_id_ind
    test_params['delimiter'] = self.data_delimiter
    test_params['extension'] = self.files_extension
    test_params['data_dir'] = self.path_to_train_data
    test_params['data_dir'] = self.path_to_test_data
    self.test_data_layer = input_layer.UserItemRecDataProvider(params=test_params,
                                                               user_id_map=self.user_id_map,
                                                               item_id_map=self.item_id_map)
    self.test_data_layer.src_data = self.full_data_layer.data

  def __init_optimizer(self):
    if self.optimizer_type == "adam":
      self.optimizer = optim.Adam(self.autoencoder.parameters(),
                                  lr=self.lr,
                                  weight_decay=self.weight_decay)
    elif self.optimizer_type == "adagrad":
      self.optimizer = optim.Adagrad(self.autoencoder.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
    elif self.optimizer_type == "sgd":
      self.optimizer = optim.SGD(self.autoencoder.parameters(),
                                 lr=self.lr, momentum=0.9,
                                 weight_decay=self.weight_decay)
    elif self.optimizer_type == "rmsprop":
      self.optimizer = optim.RMSprop(self.autoencoder.parameters(),
                                     lr=self.lr, momentum=0.9,
                                     weight_decay=self.weight_decay)
    else:
      raise Exception('Unknown optimizer kind')

  def __load_last_state(self, model_file):
    log.info('Loading model from: {}'.format(model_file))
    if not os.path.isfile(model_file):
      raise Exception('No state file found in {}'.format(model_file))
    self._model_last_state = torch.load(model_file)

  def save_state(self):
    checkpoint_file = "{}reco_ae_epoch_{}.model".format(self.model_checkpoint, self.current_epoch)
    log.info("Saving model to {}".format(checkpoint_file))
    current_state = {
      'vector_dim': self.vector_dim,
      'hidden_layers_sizes': self.hidden_layers_sizes,
      'activation_type': self.activation_type,
      'last_layer_act': self.last_layer_act,
      'is_constrained': self.is_constrained,
      'dropout_prob': self.dropout_prob,
      'user_id_map': self.user_id_map,
      'item_id_map': self.item_id_map,
      'last_epoch': self.current_epoch,
      'model': self.autoencoder.state_dict(),
      'optimizer': self.optimizer.state_dict()
    }
    torch.save(current_state, checkpoint_file)

  def run(self):
    if self.mode == 'train':
      self.train()
    elif self.mode == 'eval':
      self.eval()
    elif self.mode == 'test':
      self.test()

  def train(self):
    log.info('Training mode')

    for epoch in range(self.current_epoch, self.num_epochs):
      self.current_epoch = epoch
      log.info('Doing epoch {} of {}'.format(epoch, self.num_epochs))
      self.autoencoder.train()
      total_epoch_loss = 0.0
      num_batches = 0.0
      self.lr_scheduler.step()
      log.info('Epoch Learning Rate: {}'.format(self.lr_scheduler.get_lr()))
      agg_loss = 0
      num_samples = 0
      for i, mb in enumerate(self.train_data_layer.iterate_one_epoch()):
        inputs = mb
        self.optimizer.zero_grad()
        sparse_encoder = SparseBatchAutoEncoder(self.autoencoder, sparse_batch_in=inputs)
        reduced_inputs = sparse_encoder.reduced_batch_in

        for sample, targets in self.__process_training_input(reduced_inputs):
          outputs = sparse_encoder(sample)
          loss, num_ratings = self.compute_loss(outputs, targets)
          loss = loss / num_ratings
          loss.backward(retain_graph=True)
          self.optimizer.step()
          agg_loss += loss.data[0]
          num_samples += 1

        if (i + 1) % self.summary_frequency == 0:
          log.info('[%d, %5d] %s: %.7f' % (epoch, i + 1, self.loss_func_type, agg_loss / num_samples))
          agg_loss = 0
          num_samples = 0

        total_epoch_loss += loss.data[0]
        num_batches += 1

      log.info('Total epoch {} finished training {} loss: {}'.format(epoch, self.loss_func_type,
                                                                     total_epoch_loss / num_batches))

      if epoch % self.eval_freq == 0 or epoch == self.num_epochs - 1:
        self.eval()
        self.save_state()

  def eval(self):
    log.info('Evaluation mode')
    self.autoencoder.eval()

    total_num_ratings = 0.0

    total_epoch_loss = 0.0
    predictions_vec = np.array([])
    groundtruth_vec = np.array([])

    for i, (eval, src) in enumerate(self.eval_data_layer.iterate_one_epoch_eval(src_data_layer=self.train_data_layer)):
      inputs = src
      targets = eval
      sparse_encoder = SparseBatchAutoEncoder(self.autoencoder, sparse_batch_in=inputs, sparse_batch_out=targets)
      inputs = sparse_encoder.reduced_batch_in
      targets = sparse_encoder.reduced_batch_out

      outputs = sparse_encoder(inputs)

      loss, num_ratings = self.compute_loss(outputs, targets)
      total_epoch_loss += loss.data[0]
      total_num_ratings += num_ratings.data[0]

      targets_vec = targets.data.numpy().reshape(-1)
      targets_mask = np.abs(targets_vec) == 1
      targets_vec = targets_vec[targets_mask]
      outputs_vec = outputs.data.numpy().reshape(-1)[targets_mask]
      predictions_vec = np.append(predictions_vec, outputs_vec)
      groundtruth_vec = np.append(groundtruth_vec, targets_vec)

    total_epoch_loss = total_epoch_loss / total_num_ratings

    log.info('Validation Number of ratings: {}'.format(total_num_ratings))
    log.info('Evaluation Loss: {}'.format(total_epoch_loss))

  def test(self):
    print('Testing mode')
    self.autoencoder.eval()

    predictions_vec = np.array([])
    out_f = open('test_results_' + str(datetime.datetime.now()) + '.csv', 'w')
    for i, ((eval, src), user) in enumerate(self.test_data_layer.iterate_one_epoch_eval(for_inf=True)):
      inputs = src
      targets = eval
      sparse_encoder = SparseBatchAutoEncoder(self.autoencoder, sparse_batch_in=inputs, sparse_batch_out=targets)
      outputs = sparse_encoder()
      inputs = sparse_encoder.reduced_batch_in
      targets = sparse_encoder.reduced_batch_out

      targets_vec = targets.data.numpy().reshape(-1)
      outputs_vec = outputs.data.numpy().reshape(-1)

      real_user_id = self.test_data_layer.userIdInverseMap[user]
      for ind, target in enumerate(targets_vec):
        song_id = sparse_encoder.active_outputs_inverse_map[ind]
        real_song_id = self.test_data_layer.itemIdInverseMap[song_id]
        prediction = (outputs_vec[ind] + 1) / 2
        line = ','.join((real_user_id, real_song_id, str(prediction)))
        out_f.write(line + '\n')

    out_f.close()

  def __process_training_input(self, data):
    corrupted_data = self.denoise(data)
    yield (corrupted_data, data)

    for i in range(self.aug_samples):
      sample = F.dropout(data, p=self.aug_prob, training=True, inplace=False)
      corrupted_sample = self.denoise(sample)
      if self.aug_full:
        yield (corrupted_sample, data)
      else:
        yield (corrupted_sample, sample)

  def compute_loss(self, outputs, targets):
    mask = targets != 0
    num_ratings = torch.sum(mask.float())
    return self.loss_func(outputs * mask.float(), targets, mask.float()), num_ratings


class SoftMarginLoss(nn.Module):
  def __init__(self, size_average=True):
    super(SoftMarginLoss, self).__init__()
    self.size_average = size_average

  def forward(self, input, target, mask=None):
    loss = (((- input * target).exp()) + 1).log()
    if not mask is None:
      loss = loss * mask
    loss = loss.sum()
    if self.size_average:
      _num_elements = input.size()[0] * input.size()[1]
      loss = loss / _num_elements
    return loss
