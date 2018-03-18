import os

import glog as log

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import numpy as np

from recoder.data import RecommendationDataset
from recoder.nn import SparseBatchAutoEncoder
from recoder.recommender import InferenceRecommender
from recoder.utils import MetricEvaluator
import recoder.utils as utils

class Recoder(object):
  def __init__(self, mode, model_file=None, model_params=None,
               train_dataset=None, val_dataset=None, use_cuda=False,
               optimizer_type='sgd', lr=0.001, weight_decay=0, num_epochs=1,
               loss_module=None, batch_size=64, optimizer_lr_milestones=None,
               apply_ns=True):

    self.mode = mode
    self.model_file = model_file
    self.model_params = model_params
    self.optimizer_type = optimizer_type
    self.lr = lr
    self.weight_decay = weight_decay
    self.num_epochs = num_epochs
    self.loss_module = loss_module
    self.batch_size = batch_size
    self.optimizer_lr_milestones = optimizer_lr_milestones if optimizer_lr_milestones else []
    self.train_dataset = train_dataset # type: RecommendationDataset
    self.val_dataset = val_dataset # type: RecommendationDataset
    self.use_cuda = use_cuda
    self.apply_ns = apply_ns

    self.loss_module_name = self.loss_module.__class__.__name__

    self._model_saved_state = None
    self.user_id_map = None
    self.item_id_map = None

    if model_file is not None:
      self.__init_from_model_file(self.model_file)

    if self.mode == 'model':
      self.__init_model()
    elif mode == 'train':
      self.__init_training()

    # summarize model
    log.info('{} mode'.format('GPU' if self.use_cuda else 'CPU'))
    for param, val in self.model_params.items():
      log.info('{}: {}'.format(param, val))
    log.info('Input vector size: {}'.format(self.vector_dim))

    # deleting model state as it became useless
    del self._model_saved_state

  def __init_model(self):
    _model_params = dict(self.model_params)
    hidden_layers_sizes = list(_model_params['hidden_layers_sizes'])
    del _model_params['hidden_layers_sizes']
    layer_sizes = [self.vector_dim] + hidden_layers_sizes

    self.autoencoder = SparseBatchAutoEncoder(layer_sizes=layer_sizes,
                                              **_model_params)

    if not self._model_saved_state is None:
      self.autoencoder.load_state_dict(self._model_saved_state['model'])

    if self.use_cuda:
      self.autoencoder.cuda()

    self.autoencoder.eval()

  def __init_training(self):
    self.users = list(set(self.train_dataset.users + self.val_dataset.users))
    self.items = list(set(self.train_dataset.items + self.val_dataset.items))

    self.vector_dim = len(self.items)

    if self.user_id_map is None:
      self.__build_user_map()

    if self.item_id_map is None:
      self.__build_item_map()

    self.__init_model()
    self.__init_optimizer()

    self.current_epoch = 1

    if not self._model_saved_state is None:
      self.current_epoch = self._model_saved_state['last_epoch'] + 1

    _last_epoch = -1 if self.current_epoch == 1 else (self.current_epoch - 2)

    self.lr_scheduler = MultiStepLR(self.optimizer, milestones=self.optimizer_lr_milestones,
                                    gamma=0.1, last_epoch=_last_epoch)

    self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                       shuffle=True, collate_fn=self.collate_to_sparse_batch)
    self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                      shuffle=True, collate_fn=self.collate_to_sparse_batch)

  def __init_optimizer(self):
    params = []
    for param_name, param in self.autoencoder.named_parameters():
      weight_decay = self.weight_decay

      if 'bias' in param_name:
        weight_decay = 0

      params.append({'params': param, 'weight_decay': weight_decay})

    if self.optimizer_type == "adam":
      self.optimizer = optim.Adam(params, lr=self.lr)
    elif self.optimizer_type == "adagrad":
      self.optimizer = optim.Adagrad(params, lr=self.lr)
    elif self.optimizer_type == "sgd":
      self.optimizer = optim.SGD(params, lr=self.lr, momentum=0.9)
    elif self.optimizer_type == "rmsprop":
      self.optimizer = optim.RMSprop(params, lr=self.lr, momentum=0.9)
    else:
      raise Exception('Unknown optimizer kind')

    if not self._model_saved_state is None:
      self.optimizer.load_state_dict(self._model_saved_state['optimizer'])

  def __init_from_model_file(self, model_file):
    log.info('Loading model from: {}'.format(model_file))
    if not os.path.isfile(model_file):
      raise Exception('No state file found in {}'.format(model_file))
    self._model_saved_state = torch.load(model_file)
    self.model_params = self._model_saved_state['model_params']
    self.item_id_map = self._model_saved_state['item_id_map']
    self.user_id_map = self._model_saved_state['user_id_map']
    self.current_epoch = self._model_saved_state['last_epoch']
    self.model_state_dict = self._model_saved_state['model']
    self.optimizer_state_dict = self._model_saved_state['optimizer']
    self.vector_dim = self._model_saved_state['vector_dim']

  def __build_user_map(self):
    self.user_id_map = {}
    for user_id, user in enumerate(self.users):
      self.user_id_map[user] = user_id

  def __build_item_map(self):
    self.item_id_map = {}
    for item_id, item in enumerate(self.items):
      self.item_id_map[item] = item_id

  def save_state(self, model_checkpoint):
    checkpoint_file = "{}_epoch_{}.model".format(model_checkpoint, self.current_epoch)
    log.info("Saving model to {}".format(checkpoint_file))
    current_state = {
      'vector_dim': self.vector_dim,
      'model_params': self.model_params,
      'user_id_map': self.user_id_map,
      'item_id_map': self.item_id_map,
      'last_epoch': self.current_epoch,
      'model': self.autoencoder.state_dict(),
      'optimizer': self.optimizer.state_dict()
    }
    torch.save(current_state, checkpoint_file)

  def train(self, summary_frequency=0, val_epoch_freq=1,
            model_checkpoint=None, checkpoint_freq=1, metric_eval_params=dict()):
    log.info('Initial Learning Rate: {}'.format(self.lr))
    log.info('Weight decay: {}'.format(self.weight_decay))
    log.info('Batch Size: {}'.format(self.batch_size))
    log.info('Optimizer: {}'.format(self.optimizer_type))
    log.info('Loss Function: {}'.format(self.loss_module_name))

    for epoch in range(self.current_epoch, self.num_epochs + 1):
      self.current_epoch = epoch
      log.info('Epoch {}/{}'.format(epoch, self.num_epochs))
      self.autoencoder.train()
      aggregated_losses = []
      self.lr_scheduler.step()
      log.info('Epoch Learning Rate: {}'.format(self.lr_scheduler.get_lr()[0]))
      for itr, (input, target) in enumerate(self.train_dataloader):
        self.optimizer.zero_grad()

        output, reduced_target = self.autoencoder(input, target=target, full_output=not self.apply_ns)

        loss = self.compute_loss(output, reduced_target)

        loss.backward()
        self.optimizer.step()
        aggregated_losses.append(loss.data[0])

        if (itr + 1) % summary_frequency == 0:
          log.info('[%d, %5d] %s: %.7f' % (epoch, itr + 1, self.loss_module_name, np.mean(aggregated_losses[-summary_frequency:])))

      log.info('Taining average {} loss: {}'.format(self.loss_module_name, 
                                                    aggregated_losses.mean()))

      if epoch % val_epoch_freq == 0 or epoch == self.num_epochs:
        self.validate()
        self.evaluate(self.val_dataset, **metric_eval_params)


      if model_checkpoint and (epoch % checkpoint_freq == 0 or epoch == self.num_epochs):
        self.save_state(model_checkpoint)

  def validate(self):
    self.autoencoder.eval()

    total_loss = 0.0
    num_batches = 1

    for itr, (input, target) in enumerate(self.val_dataloader):

      output, target = self.autoencoder(input, target=target, full_output=not self.apply_ns)

      loss = self.compute_loss(output, target)
      total_loss += loss.data[0]
      num_batches = itr + 1

    avg_loss = total_loss / num_batches

    log.info('Validation Loss: {}'.format(avg_loss))

  def compute_loss(self, output, target):
    # Average loss over samples in a batch
    normalization = Variable(torch.FloatTensor([target.size(0)]))
    if self.use_cuda:
      normalization = normalization.cuda()
    loss = self.loss_module(output, target) / normalization
    return loss

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

    _map = self.item_id_map

    for sample_i, sample in enumerate(batch):
      num_values = len(sample)
      samples_inds += [sample_i] * num_values
      for target, inter in sample:
        inter_inds.append(_map[target])
        inter_val.append(inter)

    ind_lt = torch.LongTensor([samples_inds, inter_inds])
    val_ft = torch.FloatTensor(np.array(inter_val, dtype=np.float))

    batch = torch.sparse.FloatTensor(ind_lt, val_ft, torch.Size([batch_size, self.vector_dim]))

    return batch

  def infer(self, user_hist):
    input, target = self.collate_to_sparse_batch([(user_hist, user_hist)])
    output = self.autoencoder(input).cpu()
    return output

  def evaluate(self, eval_dataset, num_recommendations=100, pool_size=1000,
               k=None, metrics=None):
    self.autoencoder.eval()
    val_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True,
                                collate_fn=lambda x: x)

    k = [100] if k is None else k
    metrics = ['ndcg'] if metrics is None else metrics

    metric_evaluator = MetricEvaluator(k, metrics=metrics)

    num_preds = 0

    recommender = InferenceRecommender(self, num_recommendations,
                                       pool_size=pool_size)

    for i, m_batch in enumerate(val_dataloader):
      input, target = m_batch[0]

      recommendations = recommender.recommend(input)

      assert(len(recommendations)==len(set(recommendations)))

      relevant_songs = list(zip(*target))[0]

      metric_evaluator.evaluate(recommendations, relevant_songs)

      num_preds += 1

    metric_evaluator.summarize()
