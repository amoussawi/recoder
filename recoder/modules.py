import os

import glog as log
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from recoder.autoencoder.modules import AutoEncoder, SparseBatchAutoEncoder
from recoder.data import RecommendationDataset


class AutoEncoderRecommender(object):

  def __init__(self, mode, params):
    self.params = params
    self.mode = mode
    self._model_last_state = None

    if 'model' in self.params:
      self.__load_last_state(self.params['model'])

    if self.mode == 'model':
      self.__init_model()
      self.autoencoder.eval()
    elif mode == 'train':
      self.__init_training()
    elif mode == 'eval':
      self.__init_evaluation()

    # deleting model state as it became useless
    del self._model_last_state

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
      self.item_based = self._model_last_state['item_based']
      self.user_id_map = self._model_last_state['user_id_map']
      self.item_id_map = self._model_last_state['item_id_map']

    self.__create_model()

  def __create_model(self):
    self.autoencoder = AutoEncoder(layer_sizes=[self.vector_dim] + [int(l) for l in self.hidden_layers_sizes],
                                   activation_type=self.activation_type,
                                   is_constrained=self.is_constrained,
                                   dp_drop_prob=self.dropout_prob,
                                   last_layer_act=self.last_layer_act)

    if not self._model_last_state is None:
      self.autoencoder.load_state_dict(self._model_last_state['model'])

  def __init_common_params(self):
    self.loss_func = self.params['loss_func']
    self.loss_func_type = self.loss_func.__class__.__name__
    self.batch_size = self.params['batch_size']
    self.compute_weights = self.params['compute_weights'] if 'compute_weights' in self.params else lambda x: None

  def __init_training(self):
    self.train_dataset = self.params['train_dataset'] # type: RecommendationDataset
    self.eval_dataset = self.params['eval_dataset'] # type: RecommendationDataset
    self.__init_common_params()
    self.lr = self.params['lr']
    self.weight_decay = self.params['weight_decay']
    self.num_epochs = self.params['num_epochs']
    self.optimizer_type = self.params['optimizer_type']
    self.summary_frequency = self.params['summary_frequency'] if 'summary_frequency' in self.params else 10
    self.eval_epoch_freq = self.params['eval_epoch_freq'] if 'eval_epoch_freq' in self.params else 5
    self.eval_itr_freq = self.params['eval_itr_freq'] if 'eval_itr_freq' in self.params else 0
    self.optimizer_milestones = self.params['optimizer_milestones'] if 'optimizer_milestones' in self.params else []
    self.finetune = self.params['finetune'] if 'finetune' in self.params else False
    self.noise = self.params['noise'] if 'noise' in self.params else lambda x: x
    self.model_checkpoint = self.params['model_checkpoint'] if 'model_checkpoint' in self.params else 'model/'
    self.item_based = self.params['item_based']

    self.users = list(set(self.train_dataset.users + self.eval_dataset.users + self.eval_dataset.target_dataset.users))
    self.items = list(set(self.train_dataset.items + self.eval_dataset.items + self.eval_dataset.target_dataset.items))

    self.vector_dim = len(self.items) if self.item_based else len(self.users)

    self.__build_user_map()
    self.__build_item_map()

    self.__init_model()
    self.__init_optimizer()
    self.current_epoch = 0

    if not self._model_last_state is None and not self.finetune:
      self.optimizer.load_state_dict(self._model_last_state['optimizer'])
      self.current_epoch = self._model_last_state['last_epoch'] + 1

    self.lr_scheduler = MultiStepLR(self.optimizer, milestones=self.optimizer_milestones,
                                    gamma=0.1, last_epoch=self.current_epoch - 1)

    self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                       shuffle=True, collate_fn=self.collate_to_sparse_batch)
    self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=self.batch_size,
                                      shuffle=True, collate_fn=self.collate_to_sparse_batch)

  def __init_evaluation(self):
    self.eval_dataset = self.params['eval_dataset'] # type: RecommendationDataset
    self.__init_common_params()
    self.__init_model()

    self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=self.batch_size,
                                      shuffle=True, collate_fn=self.collate_to_sparse_batch)

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

  def __build_user_map(self):
    if self._model_last_state is not None:
      self.user_id_map = self._model_last_state['user_id_map']
    else:
      self.user_id_map = {}
      for user_id, user in enumerate(self.users):
        self.user_id_map[user] = user_id

  def __build_item_map(self):
    if self._model_last_state is not None:
      self.item_id_map = self._model_last_state['item_id_map']
    else:
      self.item_id_map = {}
      for item_id, item in enumerate(self.items):
        self.item_id_map[item] = item_id

  def save_state(self):
    checkpoint_file = "{}reco_ae_epoch_{}.model".format(self.model_checkpoint, self.current_epoch)
    log.info("Saving model to {}".format(checkpoint_file))
    current_state = {
      'item_based': self.item_based,
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
    log.info('Hidden layers sizes: {}'.format(self.hidden_layers_sizes))
    log.info('Activation type: {}'.format(self.activation_type))
    log.info('Vector dim: {}'.format(self.vector_dim))
    if self.mode == 'train':
      log.info('Learning rate: {}'.format(self.lr))
      log.info('Weight decay: {}'.format(self.weight_decay))
      log.info('Optimizer type: {}'.format(self.optimizer_type))
      log.info('Loss function: {}'.format(self.loss_func_type))
      log.info('Dropout probability: {}'.format(self.dropout_prob))
      self.train()
    elif self.mode == 'eval':
      log.info('Loss function: {}'.format(self.loss_func_type))
      self.eval()

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
      for itr, (input, target) in enumerate(self.train_dataloader):
        sparse_encoder = SparseBatchAutoEncoder(self.autoencoder, sparse_batch_in=input, sparse_batch_out=target)

        reduced_input = sparse_encoder.reduced_batch_in
        reduced_target = sparse_encoder.reduced_batch_out

        transformed_input = self.transform(reduced_input)

        corrupted_input = self.noise(transformed_input)

        transformed_target = self.transform(reduced_target)

        self.optimizer.zero_grad()

        output = sparse_encoder(corrupted_input)

        weights = self.compute_weights(transformed_target)

        loss, normalization = self.compute_loss(output, transformed_target, weights)
        loss = loss / normalization
        loss.backward()
        self.optimizer.step()
        agg_loss += loss.data[0]
        num_samples += 1

        if (itr + 1) % self.summary_frequency == 0:
          log.info('[%d, %5d] %s: %.7f' % (epoch, itr + 1, self.loss_func_type, agg_loss / num_samples))
          agg_loss = 0
          num_samples = 0

        if (itr + 1) % self.eval_itr_freq == 0:
          self.eval()
          self.autoencoder.train()

        total_epoch_loss += loss.data[0]
        num_batches += 1

      log.info('Total epoch {} finished training {} loss: {}'.format(epoch, self.loss_func_type,
                                                                     total_epoch_loss / num_batches))

      if epoch % self.eval_epoch_freq == 0 or epoch == self.num_epochs - 1:
        self.eval()
        self.save_state()

  def eval(self):
    log.info('Evaluation mode')
    self.autoencoder.eval()

    loss_normalization = 0.0

    total_epoch_loss = 0.0

    for itr, (input, target) in enumerate(self.eval_dataloader):
      sparse_encoder = SparseBatchAutoEncoder(self.autoencoder, sparse_batch_in=input, sparse_batch_out=target)
      input = sparse_encoder.reduced_batch_in
      target = sparse_encoder.reduced_batch_out

      transformed_input = self.transform(input)
      transformed_target = self.transform(target)

      output = sparse_encoder(transformed_input)

      weights = self.compute_weights(transformed_target)
      loss, normalization = self.compute_loss(output, transformed_target, weights)
      total_epoch_loss += loss.data[0]
      loss_normalization += normalization.data[0]

    total_epoch_loss = total_epoch_loss / loss_normalization

    log.info('Evaluation Loss: {}'.format(total_epoch_loss))

  def compute_loss(self, output, target, weights=None):
    if weights is not None:
      normalization = torch.sum(weights)
    else:
      normalization = Variable(torch.FloatTensor([1]))
    return self.loss_func(output, target, weights), normalization

  def collate_to_sparse_batch(self, batch):
    _input_batch = [i for i, t in batch]
    _target_batch = [t for d, t in batch]

    _sparse_input_batch = self.__collate_to_sparse_batch(_input_batch)

    if _input_batch[0] is _target_batch[0]: # If target is input no need to re-compute
      _sparse_target_batch = _sparse_input_batch
    else:
      _sparse_target_batch = self.__collate_to_sparse_batch(_target_batch)

    return _sparse_input_batch, _sparse_target_batch

  def __collate_to_sparse_batch(self, batch):
    samples_inds = []
    inter_inds = []
    inter_val = []
    batch_size = len(batch)

    _map = self.item_id_map if self.item_based else self.user_id_map

    for sample_i, sample in enumerate(batch):
      num_values = len(sample)
      samples_inds += [sample_i] * num_values
      for target, inter in sample:
        inter_inds.append(_map[target])
        inter_val.append(inter)

    ind_lt = torch.LongTensor([samples_inds, inter_inds])
    val_ft = torch.FloatTensor(inter_val)

    batch = torch.sparse.FloatTensor(ind_lt, val_ft, torch.Size([batch_size, self.vector_dim]))

    return batch



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

class MSELoss(nn.Module):
  def __init__(self, size_average=True):
    super(MSELoss, self).__init__()
    self.size_average = size_average

  def forward(self, input, target, weights=None):
    if not weights is None:
      input = input * weights.sqrt()
      target = target * weights.sqrt()
    return F.mse_loss(input, target, size_average=self.size_average)
