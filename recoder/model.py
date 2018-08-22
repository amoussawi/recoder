import os

import glog as log

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss

import numpy as np

from recoder.data import RecommendationDataset
from recoder.metrics import RecommenderEvaluator
from recoder.nn import DynamicAutoencoder
from recoder.recommender import InferenceRecommender
from recoder.nn import MSELoss, MultinomialNLLLoss


class Recoder(object):
  """
  Module to train/evaluate a recommendation Autoencoder-based model

  Args:
    mode (str): the mode of the model, either 'train' or 'model'. 'model' only used when loading a
      pre-trained model.
    model_file (str, optional): the model file. required in 'model' model. and used to continue training
      in 'train' mode.
    hidden_layers (list, optional): Autoencoder hidden layers sizes. required in 'train' mode.
    model_params (dict, optional): the Autoencoder model extra parameters other than layer_sizes.
    train_dataset (RecommendationDataset, optional): train dataset. required in 'train' mode.
    val_dataset (RecommendationDataset, optional): validation dataset. required in 'train' mode.
    use_cuda (bool, optional): use GPU on training/evaluation the model.
    optimizer_type (str, optional): optimizer type (one of 'sgd', 'adam', 'adagrad', 'rmsprop').
    lr (float, optional): learning rate.
    weight_decay (float, optional): weight decay (L2 normalization).
    num_epochs (int, optional): number of epochs to train the model
    loss (str, optional): loss function used to train the model. required on 'train' mode.
      'mse' for `recoder.nn.MSELoss`, 'logistic' for `torch.nn.BCEWithLogitsLoss`,
      and 'logloss' for `recoder.nn.MultinomialNLLLoss`
    loss_params (dict, optional): loss function extra params based on loss module.
    batch_size (int, optional): batch size
    optimizer_lr_milestones (list, optional): optimizer learning rate epochs milestones (0.1 decay).
    num_neg_samples (int, optional): number of negative samples to generate for each user.
      If `-1`, then all possible negative items will be sampled. If `0`, only the positive items from
      the other examples in the mini-batch will be sampled. If `> 0`, then `num_neg_samples` samples
      will be randomly sampled including the negative items from the other examples in the mini-batch.
  """

  def __init__(self, mode, model_file=None, hidden_layers=None, model_params=None,
               train_dataset=None, val_dataset=None, use_cuda=False,
               optimizer_type='sgd', lr=0.001, weight_decay=0, num_epochs=1,
               loss='mse', loss_params=None, batch_size=64, optimizer_lr_milestones=None,
               num_neg_samples=0, num_data_workers=0):

    self.mode = mode
    self.model_file = model_file
    self.hidden_layers = hidden_layers
    self.model_params = model_params if model_params else {}
    self.optimizer_type = optimizer_type
    self.lr = lr
    self.weight_decay = weight_decay
    self.num_epochs = num_epochs
    self.loss = loss
    self.loss_params = loss_params if loss_params else {}
    self.batch_size = batch_size
    self.optimizer_lr_milestones = optimizer_lr_milestones if optimizer_lr_milestones else []
    self.train_dataset = train_dataset # type: RecommendationDataset
    self.val_dataset = val_dataset # type: RecommendationDataset
    self.use_cuda = use_cuda
    self.num_neg_samples = num_neg_samples
    self.num_data_workers = num_data_workers

    if self.use_cuda:
      self.device = torch.device('cuda')
    else:
      self.device = torch.device('cpu')

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
    layer_sizes = [self.vector_dim] + self.hidden_layers

    self.autoencoder = DynamicAutoencoder(layer_sizes=layer_sizes,
                                          **self.model_params)

    if not self._model_saved_state is None:
      self.autoencoder.load_state_dict(self._model_saved_state['model'])

    self.autoencoder = self.autoencoder.to(device=self.device)

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
    self.__init_loss_module()

    self.current_epoch = 1

    if not self._model_saved_state is None:
      self.current_epoch = self._model_saved_state['last_epoch'] + 1

    _last_epoch = -1 if self.current_epoch == 1 else (self.current_epoch - 2)

    self.lr_scheduler = MultiStepLR(self.optimizer, milestones=self.optimizer_lr_milestones,
                                    gamma=0.1, last_epoch=_last_epoch)

    collate_fn = lambda x: self.__collate_batch(x, with_ns=(self.num_neg_samples >= 0))
    self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                       shuffle=True, collate_fn=collate_fn,
                                       num_workers=self.num_data_workers)
    self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                     shuffle=True, collate_fn=collate_fn,
                                     num_workers=self.num_data_workers)

  def __init_loss_module(self):
    if self.loss == 'logistic':
      self.loss_module = BCEWithLogitsLoss(size_average=False, **self.loss_params)
    elif self.loss == 'mse':
      self.loss_module = MSELoss(size_average=False, **self.loss_params)
    elif self.loss == 'logloss':
      self.loss_module = MultinomialNLLLoss(size_average=False)
    else:
      raise ValueError('Unknown loss function {}'.format(self.loss))

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
    self._model_saved_state = torch.load(model_file, map_location='cpu')
    self.hidden_layers = self._model_saved_state['hidden_layers']
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
      'hidden_layers': self.hidden_layers,
      'model_params': self.model_params,
      'user_id_map': self.user_id_map,
      'item_id_map': self.item_id_map,
      'last_epoch': self.current_epoch,
      'model': self.autoencoder.state_dict(),
      'optimizer': self.optimizer.state_dict()
    }
    torch.save(current_state, checkpoint_file)

  def train(self, summary_frequency=0, val_epoch_freq=1,
            model_checkpoint=None, checkpoint_freq=1,
            eval_num_recommendations=None, metrics=None):
    """
    Train the model

    Args:
      summary_frequency (int, optional): batches iterations frequency of averaging loss
        and logging it
      val_epoch_freq (int, optional): epochs frequency of doing a validation pass
      model_checkpoint (str, optional): file where to save the model
      checkpoint_freq (int, optional): epochs frequency of saving a checkpoint the model
      eval_num_recommendations (int, optional): num of recommendations to generate on validation
      metrics (list, optional): list of ``Metric`` used to evaluate the model
    """

    log.info('Initial Learning Rate: {}'.format(self.lr))
    log.info('Weight decay: {}'.format(self.weight_decay))
    log.info('Batch Size: {}'.format(self.batch_size))
    log.info('Optimizer: {}'.format(self.optimizer_type))
    log.info('Loss Function: {}'.format(self.loss))

    for epoch in range(self.current_epoch, self.num_epochs + 1):
      self.current_epoch = epoch
      log.info('Epoch {}/{}'.format(epoch, self.num_epochs))
      self.autoencoder.train()
      aggregated_losses = []
      self.lr_scheduler.step()
      log.info('Epoch Learning Rate: {}'.format(self.lr_scheduler.get_lr()[0]))
      for itr, ((input, input_words), (target, target_words)) in enumerate(self.train_dataloader):
        self.optimizer.zero_grad()

        input = input.to(device=self.device)
        target = target.to(device=self.device)
        if input_words is not None:
          input_words = input_words.to(device=self.device)
        if target_words is not None:
          target_words = target_words.to(device=self.device)

        output = self.autoencoder(input, input_words=input_words,
                                  target_words=target_words)

        loss = self.compute_loss(output, target)

        loss.backward()
        self.optimizer.step()

        if (itr + 1) % summary_frequency == 0:
          aggregated_losses.append(loss.item())
          log.info('[%d, %5d] %s: %.7f' % (epoch, itr + 1, self.loss, aggregated_losses[-1]))

      log.info('Taining average {} loss: {}'.format(self.loss, np.mean(aggregated_losses)))

      if epoch % val_epoch_freq == 0 or epoch == self.num_epochs:
        self.validate()
        if metrics is not None and eval_num_recommendations is not None:
          self.evaluate(self.val_dataset, num_recommendations=eval_num_recommendations,
                        metrics=metrics, batch_size=self.batch_size)


      if model_checkpoint and (epoch % checkpoint_freq == 0 or epoch == self.num_epochs):
        self.save_state(model_checkpoint)

  def validate(self):
    self.autoencoder.eval()

    total_loss = 0.0
    num_batches = 1

    for itr, ((input, input_words), (target, target_words)) in enumerate(self.val_dataloader):

      input = input.to(device=self.device)
      target = target.to(device=self.device)
      if input_words is not None:
        input_words = input_words.to(device=self.device)
      if target_words is not None:
        target_words = target_words.to(device=self.device)

      output = self.autoencoder(input, input_words=input_words,
                                target_words=target_words)

      loss = self.compute_loss(output, target)
      total_loss += loss.item()
      num_batches = itr + 1

    avg_loss = total_loss / num_batches

    log.info('Validation Loss: {}'.format(avg_loss))

  def compute_loss(self, output, target):
    # Average loss over samples in a batch
    normalization = torch.FloatTensor([target.size(0)]).to(device=self.device)
    loss = self.loss_module(output, target) / normalization
    return loss

  def __collate_batch(self, batch, with_ns=True):
    _input_batch = [i for i, t in batch]
    _target_batch = [t for d, t in batch]

    input = self.__collate_interactions_batch(_input_batch, with_ns=with_ns)

    if _input_batch[0] is _target_batch[0]: # If target is input no need to re-compute
      target = input
    else:
      target = self.__collate_interactions_batch(_target_batch, with_ns=with_ns)

    return input, target

  def __collate_interactions_batch(self, batch, with_ns=True):
    batch_size = len(batch)

    samples_inds = []
    inter_inds = []
    inter_vals = []
    for sample_i, user_inters in enumerate(batch):
      num_inters = len(user_inters)
      samples_inds.extend([sample_i] * num_inters)
      for item, inter_val in user_inters:
        inter_inds.append(self.item_id_map[item])
        inter_vals.append(inter_val)

    if self.num_neg_samples >= 0 and with_ns:
      negative_items = np.random.randint(0, self.vector_dim, self.num_neg_samples)
      negative_items = negative_items[np.isin(negative_items, inter_inds, invert=True)]
      num_negative_items = len(negative_items)

      # It's enough to fill the first sample in the batch with negative items
      # the others will be filled by transforming the sparse matrix into dense
      if num_negative_items > 0:
        samples_inds.extend(np.repeat(0, num_negative_items))
        inter_inds.extend(negative_items)
        inter_vals.extend(np.repeat(0, num_negative_items))

      active_columns_ordered = np.unique(inter_inds)
      new_map = dict([(v, ind) for ind, v in enumerate(active_columns_ordered)])
      inter_inds = list(map(lambda x: new_map[x], inter_inds))
      _vector_dim = len(active_columns_ordered)
      active_cols = torch.LongTensor(active_columns_ordered)
    else:
      _vector_dim = self.vector_dim
      active_cols = None

    ind_lt = torch.LongTensor([samples_inds, inter_inds])
    val_ft = torch.FloatTensor(inter_vals)

    batch = torch.sparse.FloatTensor(ind_lt, val_ft, torch.Size([batch_size, _vector_dim]))

    batch = batch.to_dense()
    return batch, active_cols

  def predict(self, users_hist, return_input=False):
    (input, input_words), _ = self.__collate_batch(list(zip(users_hist, users_hist)),
                                                   with_ns=False)
    input = input.to(device=self.device)
    output = self.autoencoder(input, full_output=True).cpu()
    return output, input.cpu() if return_input else output

  def evaluate(self, eval_dataset, num_recommendations, metrics, batch_size=1):
    """
    Evaluates the current model given an evaluation dataset.

    Args:
      eval_dataset (RecommendationDataset): evaluation dataset
      num_recommendations (int): number of top recommendations to consider.
      metrics (list): list of ``Metric`` to use for evaluation.
      batch_size (int, optional): batch size of computations.
    """

    self.autoencoder.eval()
    recommender = InferenceRecommender(self, num_recommendations)

    evaluator = RecommenderEvaluator(recommender, metrics)

    results = evaluator.evaluate(eval_dataset, batch_size=batch_size)

    for metric in results:
      log.info('{}: {}'.format(metric, np.mean(results[metric])))
