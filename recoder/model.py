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
from recoder.losses import MSELoss, MultinomialNLLLoss
import recoder.utils as utils

from tqdm import tqdm


class Recoder(object):
  """
  Module to train/evaluate a recommendation Autoencoder-based model

  Args:
    num_items (int, optional): Number of items to model. This is used as the size of the input layer.
      If not provided, it will be computed from the training dataset passed to ``train``.
    hidden_layers (list, optional): Autoencoder hidden layers sizes. required for training from scratch.
    model_params (dict, optional): the Autoencoder model extra parameters other than `layer_sizes`,
      and `last_layer_act`.
    use_cuda (bool, optional): use GPU on training/evaluation the model.
    optimizer_type (str, optional): optimizer type (one of 'sgd', 'adam', 'adagrad', 'rmsprop').
    loss (str or torch.nn.Module, optional): loss function used to train the model.
      If loss is a ``str``, it should be `mse` for ``recoder.losses.MSELoss``, `logistic` for
      ``torch.nn.BCEWithLogitsLoss``, or `logloss` for ``recoder.losses.MultinomialNLLLoss``. If ``loss``
      is a ``torch.nn.Module``, then that Module will be used as a loss function and make sure that
      the loss reduction is a sum reduction and not an elementwise mean.
    loss_params (dict, optional): loss function extra params based on loss module if ``loss`` is a ``str``.
      Ignored if ``loss`` is a ``torch.nn.Module``.
    index_item_ids (bool, optional): If ``True``, the item ids will be indexed. Used when the item ids
      are strings, or integers but don't start with 0 and can have values much larger than the total
      number of items in the dataset. The item ids index is provided by accessing ``Recoder.item_id_map``,
      which maps from original item id to the new item id. If ``False``, the item ids in the dataset
      should be integers, and the number of items will be equal to `max(item_ids) + 1` assuming item
      ids start with 0. Note: indexing the item ids can slightly slow the training process.
  """

  def __init__(self, num_items=None, hidden_layers=None,
               model_params=None, use_cuda=False, optimizer_type='sgd',
               loss='mse', loss_params=None, index_item_ids=True):

    self.num_items = num_items
    self.hidden_layers = hidden_layers
    self.model_params = model_params if model_params else {}
    self.optimizer_type = optimizer_type
    self.loss = loss
    self.loss_params = loss_params if loss_params else {}
    self.use_cuda = use_cuda
    self.index_item_ids = index_item_ids

    if self.use_cuda:
      self.device = torch.device('cuda')
    else:
      self.device = torch.device('cpu')

    self.autoencoder = None
    self.item_id_map = None
    self.optimizer = None
    self.current_epoch = 1
    self._vector_dim = num_items
    self.items = None

    self.__optimizer_state_dict = None

  def __init_model(self):
    if self.autoencoder is not None:
      return

    layer_sizes = [self._vector_dim] + self.hidden_layers
    _model_params = dict(self.model_params)
    _model_params.pop('last_layer_act', None) # ignore last_layer_act param if it was passed
    self.autoencoder = DynamicAutoencoder(layer_sizes=layer_sizes,
                                          last_layer_act='none',
                                          **_model_params)

    self.autoencoder = self.autoencoder.to(device=self.device)

    self.autoencoder.eval()

  def __init_loss_module(self):
    if issubclass(self.loss.__class__, torch.nn.Module):
      self.loss_module = self.loss
    elif self.loss == 'logistic':
      self.loss_module = BCEWithLogitsLoss(reduction='sum', **self.loss_params)
    elif self.loss == 'mse':
      self.loss_module = MSELoss(reduction='sum', **self.loss_params)
    elif self.loss == 'logloss':
      self.loss_module = MultinomialNLLLoss(reduction='sum')
    else:
      raise ValueError('Unknown loss function {}'.format(self.loss))

  def __init_optimizer(self, lr, weight_decay):
    if self.optimizer is not None:
      self.__optimizer_state_dict = self.optimizer.state_dict()

    params = []
    for param_name, param in self.autoencoder.named_parameters():
      _weight_decay = weight_decay

      if 'bias' in param_name:
        _weight_decay = 0

      params.append({'params': param, 'weight_decay': _weight_decay})

    if self.optimizer_type == "adam":
      self.optimizer = optim.Adam(params, lr=lr)
    elif self.optimizer_type == "adagrad":
      self.optimizer = optim.Adagrad(params, lr=lr)
    elif self.optimizer_type == "sgd":
      self.optimizer = optim.SGD(params, lr=lr, momentum=0.9)
    elif self.optimizer_type == "rmsprop":
      self.optimizer = optim.RMSprop(params, lr=lr, momentum=0.9)
    else:
      raise Exception('Unknown optimizer kind')

    if self.__optimizer_state_dict is not None:
      self.optimizer.load_state_dict(self.__optimizer_state_dict)
      self.__optimizer_state_dict = None # no need for this anymore

  def init_from_model_file(self, model_file):
    """
    Initializes the model from a pre-trained model

    Args:
       model_file (str): the pre-trained model file path
    """
    log.info('Loading model from: {}'.format(model_file))
    if not os.path.isfile(model_file):
      raise Exception('No state file found in {}'.format(model_file))
    model_saved_state = torch.load(model_file, map_location='cpu')
    self.hidden_layers = model_saved_state['hidden_layers']
    self.model_params = model_saved_state['model_params']
    self.item_id_map = model_saved_state['item_id_map']
    self.current_epoch = model_saved_state['last_epoch']
    self._vector_dim = model_saved_state['vector_dim']
    self.loss = model_saved_state['loss']
    self.loss_params = model_saved_state['loss_params']
    self.optimizer_type = model_saved_state['optimizer_type']

    self.__optimizer_state_dict = model_saved_state['optimizer']

    self.__init_model()
    self.autoencoder.load_state_dict(model_saved_state['model'])

  def save_state(self, model_checkpoint_prefix):
    """
    Saves the model state in the path starting with ``model_checkpoint_prefix`` and appending it
    with the model current training epoch

    Args:
      model_checkpoint_prefix (str): the model save path prefix

    Returns:
      the model state file path
    """
    checkpoint_file = "{}_epoch_{}.model".format(model_checkpoint_prefix, self.current_epoch)
    log.info("Saving model to {}".format(checkpoint_file))
    current_state = {
      'vector_dim': self._vector_dim,
      'hidden_layers': self.hidden_layers,
      'model_params': self.model_params,
      'item_id_map': self.item_id_map,
      'last_epoch': self.current_epoch,
      'model': self.autoencoder.state_dict(),
      'optimizer_type': self.optimizer_type,
      'optimizer': self.optimizer.state_dict(),
      'loss': self.loss,
      'loss_params': self.loss_params,
    }
    torch.save(current_state, checkpoint_file)
    return checkpoint_file

  def __init_training(self, train_dataset, lr,
                      weight_decay):
    if self.items is None:
      self.items = train_dataset.items
    else:
      self.items = list(set(self.items + train_dataset.items))

    if self.index_item_ids:
      if self._vector_dim is None:
        self._vector_dim = len(self.items)
      else:
        assert self._vector_dim >= len(self.items), \
          'number of items should be smaller or equal than the vector dimension'

      if self.item_id_map is None:
        self.item_id_map = dict([(item_id, idx) for idx, item_id in enumerate(self.items)])
      else:
        assert len(np.setdiff1d(self.items, list(self.item_id_map.keys()))) == 0, \
          "there are items in the dataset that doesn't exist in the items index"
    else:
      assert type(self.items[0]) is int, 'item ids should be integers, or set index_item_ids to True'

      if self._vector_dim is None:
        self._vector_dim = int(np.max(self.items)) + 1
      else:
        assert self._vector_dim >= int(np.max(self.items)) + 1,\
          'the largest item id should be smaller than vector dimension'

    self.__init_model()
    self.__init_optimizer(lr=lr, weight_decay=weight_decay)
    self.__init_loss_module()

  def train(self, train_dataset, val_dataset=None,
            lr=0.001, weight_decay=0, num_epochs=1,
            batch_size=64, lr_milestones=None,
            num_neg_samples=0, num_data_workers=0,
            model_checkpoint_prefix=None, checkpoint_freq=0,
            eval_freq=0, eval_num_recommendations=None, metrics=None):
    """
    Trains the model

    Args:
      train_dataset (RecommendationDataset): train dataset.
      val_dataset (RecommendationDataset, optional): validation dataset.
      lr (float, optional): learning rate.
      weight_decay (float, optional): weight decay (L2 normalization).
      num_epochs (int, optional): number of epochs to train the model.
      batch_size (int, optional): batch size
      lr_milestones (list, optional): optimizer learning rate epochs milestones (0.1 decay).
      num_neg_samples (int, optional): number of negative samples to generate for each user.
        If `-1`, then all possible negative items will be sampled. If `0`, the negative items
        are sampled with mini-batch based negative sampling. If `> 0`, the negative items
        are sampled with mini-batch based negative sampling in addition to common random negative
        items if needed.
      num_data_workers (int, optional): number of data workers to use for building the mini-batches.
      model_checkpoint_prefix (str, optional): model checkpoint save path prefix
      checkpoint_freq (int, optional): epochs frequency of saving a checkpoint of the model
      eval_freq (int, optional): epochs frequency of doing an evaluation
      eval_num_recommendations (int, optional): num of recommendations to generate on evaluation
      metrics (list, optional): list of ``Metric`` used to evaluate the model
    """
    log.info('{} Mode'.format('CPU' if self.device.type == 'cpu' else 'GPU'))
    log.info('Hidden Layers: {}'.format(self.hidden_layers))
    for param in self.model_params:
      log.info('Model {}: {}'.format(param, self.model_params[param]))
    log.info('Initial Learning Rate: {}'.format(lr))
    log.info('Weight decay: {}'.format(weight_decay))
    log.info('Batch Size: {}'.format(batch_size))
    log.info('Optimizer: {}'.format(self.optimizer_type))
    log.info('LR milestones: {}'.format(lr_milestones))
    log.info('Loss Function: {}'.format(self.loss))
    for param in self.loss_params:
      log.info('Loss {}: {}'.format(param, self.loss_params[param]))

    self.__init_training(train_dataset=train_dataset, lr=lr, weight_decay=weight_decay)

    collate_fn = lambda x: self.__collate_batch(x, with_ns=(num_neg_samples >= 0),
                                                num_neg_samples=num_neg_samples)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_fn,
                                  num_workers=num_data_workers)
    if val_dataset is not None:
      val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_fn,
                                  num_workers=num_data_workers)
    else:
      val_dataloader = None

    if lr_milestones is not None:
      _last_epoch = -1 if self.current_epoch == 1 else (self.current_epoch - 2)
      lr_scheduler = MultiStepLR(self.optimizer, milestones=lr_milestones,
                                 gamma=0.1, last_epoch=_last_epoch)
    else:
      lr_scheduler = None

    self._train(train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                num_epochs=num_epochs,
                current_epoch=self.current_epoch,
                lr_scheduler=lr_scheduler,
                batch_size=batch_size,
                model_checkpoint_prefix=model_checkpoint_prefix,
                checkpoint_freq=checkpoint_freq,
                eval_freq=eval_freq,
                metrics=metrics,
                eval_num_recommendations=eval_num_recommendations)


  def _train(self, train_dataloader, val_dataloader,
             num_epochs, current_epoch, lr_scheduler,
             batch_size, model_checkpoint_prefix, checkpoint_freq,
             eval_freq, metrics, eval_num_recommendations):
    num_batches = len(train_dataloader)
    for epoch in range(current_epoch, num_epochs + 1):
      self.current_epoch = epoch
      self.autoencoder.train()
      aggregated_losses = []
      if lr_scheduler is not None:
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]
      else:
        lr = self.optimizer.defaults['lr']
      description = 'Epoch {}/{} (lr={})'.format(epoch, num_epochs, lr)
      progress_bar = tqdm(range(num_batches), desc=description)
      for batch_itr, (input, target) in enumerate(train_dataloader, 1):
        self.optimizer.zero_grad()

        # building the sparse tensor from
        input_idx, input_val, input_size, input_words = input
        target_idx, target_val, target_size, target_words = target
        input_dense = torch.sparse.FloatTensor(input_idx, input_val, input_size)\
          .to(device=self.device).to_dense()
        target_dense = torch.sparse.FloatTensor(target_idx, target_val, target_size)\
          .to(device=self.device).to_dense()
        if input_words is not None:
          input_words = input_words.to(device=self.device)
        if target_words is not None:
          target_words = target_words.to(device=self.device)

        output = self.autoencoder(input_dense, input_words=input_words,
                                  target_words=target_words)

        loss = self._compute_loss(output, target_dense)

        loss.backward()
        self.optimizer.step()

        aggregated_losses.append(loss.item())
        progress_bar.set_postfix(loss=np.mean(aggregated_losses[-1]), refresh=False)
        progress_bar.update()

      postfix = {'loss': loss.item()}
      if eval_freq > 0 and epoch % eval_freq == 0 and val_dataloader is not None:
        val_loss = self._validate(val_dataloader)
        postfix['val_loss'] = val_loss
        if metrics is not None and eval_num_recommendations is not None:
          results = self._evaluate(val_dataloader.dataset,
                                   num_recommendations=eval_num_recommendations,
                                   metrics=metrics, batch_size=batch_size)
          for metric in results:
            postfix[str(metric)] = np.mean(results[metric])

      if model_checkpoint_prefix and \
          ((checkpoint_freq > 0 and epoch % checkpoint_freq == 0) or epoch == num_epochs):
        self.save_state(model_checkpoint_prefix)

      progress_bar.set_postfix(postfix)
      progress_bar.close()

  def _validate(self, val_dataloader):
    self.autoencoder.eval()

    total_loss = 0.0
    num_batches = 1

    for itr, (input, target) in enumerate(val_dataloader):

      input_idx, input_val, input_size, input_words = input
      target_idx, target_val, target_size, target_words = target
      input_dense = torch.sparse.FloatTensor(input_idx, input_val, input_size) \
        .to(device=self.device).to_dense()
      target_dense = torch.sparse.FloatTensor(target_idx, target_val, target_size).to(device=self.device).to_dense()
      if input_words is not None:
        input_words = input_words.to(device=self.device)
      if target_words is not None:
        target_words = target_words.to(device=self.device)

      output = self.autoencoder(input_dense, input_words=input_words,
                                target_words=target_words)

      loss = self._compute_loss(output, target_dense)
      total_loss += loss.item()
      num_batches = itr + 1

    avg_loss = total_loss / num_batches

    return avg_loss

  def _compute_loss(self, output, target):
    # Average loss over samples in a batch
    normalization = torch.FloatTensor([target.size(0)]).to(device=self.device)
    loss = self.loss_module(output, target) / normalization
    return loss

  def __collate_batch(self, batch, with_ns=True,
                      num_neg_samples=0):
    if type(batch[0]) is tuple:
      # then we have (input, target)
      _input_batch, _target_batch = utils.unzip(batch)
    else:
      # in that case the target is the same as the input
      _input_batch = batch
      _target_batch = None

    input = self.__collate_interactions_batch(_input_batch, with_ns=with_ns,
                                              num_neg_samples=num_neg_samples)

    if _target_batch is None:
      target = input
    else:
      target = self.__collate_interactions_batch(_target_batch, with_ns=with_ns,
                                                 num_neg_samples=num_neg_samples)

    return input, target

  def __collate_interactions_batch(self, batch, with_ns=True,
                                   num_neg_samples=0):
    batch_size = len(batch)

    samples_inds = []
    inter_inds = []
    inter_vals = []
    for sample_i, user_inters in enumerate(batch):
      num_inters = len(user_inters)
      samples_inds.extend([sample_i] * num_inters)
      _inter_inds, _inter_vals = utils.unzip(user_inters)
      if self.index_item_ids:
        _inter_inds = map(lambda item_id: self.item_id_map[item_id], _inter_inds)
      inter_inds.extend(_inter_inds)
      inter_vals.extend(_inter_vals)

    if num_neg_samples >= 0 and with_ns:
      negative_items = np.random.randint(0, self._vector_dim, num_neg_samples)
      negative_items = negative_items[np.isin(negative_items, inter_inds, invert=True)]
      num_negative_items = len(negative_items)

      # It's enough to fill the first sample in the batch with negative items
      # the others will be filled by transforming the sparse matrix into dense
      if num_negative_items > 0:
        samples_inds.extend([0] * num_negative_items)
        inter_inds.extend(negative_items)
        inter_vals.extend([0] * num_negative_items)

      active_columns_ordered = np.unique(inter_inds)
      new_map = dict([(v, ind) for ind, v in enumerate(active_columns_ordered)])
      inter_inds = list(map(lambda x: new_map[x], inter_inds))
      _vector_dim = len(active_columns_ordered)
      active_cols = torch.LongTensor(active_columns_ordered)
    else:
      _vector_dim = self._vector_dim
      active_cols = None

    indices = torch.LongTensor([samples_inds, inter_inds])
    values = torch.FloatTensor(inter_vals)

    return indices, values, torch.Size([batch_size, _vector_dim]), active_cols

  def predict(self, users_hist, return_input=False):
    """
    Predicts the user interactions with all items

    Args:
      users_hist (list): A batch of users' history consisting of list of ``Interaction``
      return_input (bool, optional): whether to return the dense input batch

    Returns:
      if ``return_input`` is ``True`` a tuple of the predictions and the input batch
      is returned, otherwise only the predictions are returned
    """
    if self.autoencoder is None:
      raise Exception('Model not initialized.')

    self.autoencoder.eval()
    input, _ = self.__collate_batch(users_hist, with_ns=False)
    input_idx, input_val, input_size, input_words = input
    input_dense = torch.sparse.FloatTensor(input_idx, input_val, input_size) \
      .to(device=self.device).to_dense()
    output = self.autoencoder(input_dense, full_output=True).cpu()
    return output, input_dense.cpu() if return_input else output

  def _evaluate(self, eval_dataset, num_recommendations, metrics, batch_size=1):
    if self.autoencoder is None:
      raise Exception('Model not initialized')

    self.autoencoder.eval()
    recommender = InferenceRecommender(self, num_recommendations)

    evaluator = RecommenderEvaluator(recommender, metrics)

    results = evaluator.evaluate(eval_dataset, batch_size=batch_size)
    return results

  def evaluate(self, eval_dataset, num_recommendations, metrics, batch_size=1):
    """
    Evaluates the current model given an evaluation dataset.

    Args:
      eval_dataset (RecommendationDataset): evaluation dataset
      num_recommendations (int): number of top recommendations to consider.
      metrics (list): list of ``Metric`` to use for evaluation.
      batch_size (int, optional): batch size of computations.
    """
    results = self._evaluate(eval_dataset, num_recommendations, metrics,
                             batch_size=batch_size)
    for metric in results:
      log.info('{}: {}'.format(metric, np.mean(results[metric])))
