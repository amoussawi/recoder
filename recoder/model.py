import os

import glog as log

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import BCEWithLogitsLoss

import numpy as np

from recoder import __version__
from recoder.data import RecommendationDataset, RecommendationDataLoader, BatchCollator
from recoder.metrics import RecommenderEvaluator
from recoder.nn import FactorizationModel
from recoder.recommender import InferenceRecommender
from recoder.losses import MSELoss, MultinomialNLLLoss

from tqdm import tqdm


class Recoder(object):
  """
  Module to train/evaluate a recommendation :class:`recoder.nn.FactorizationModel`.

  Args:
    model (FactorizationModel): the factorization model to train.
    num_items (int, optional): the number of items to represent. If None, it will
      be computed from the first training dataset passed to ``train()``.
    num_users (int, optional): the number of users to represent. If not provided, it will
      be computed from the first training dataset passed to ``train()``.
    optimizer_type (str, optional): optimizer type (one of 'sgd', 'adam', 'adagrad', 'rmsprop').
    loss (str or torch.nn.Module, optional): loss function used to train the model.
      If loss is a ``str``, it should be `mse` for ``recoder.losses.MSELoss``, `logistic` for
      ``torch.nn.BCEWithLogitsLoss``, or `logloss` for ``recoder.losses.MultinomialNLLLoss``. If ``loss``
      is a ``torch.nn.Module``, then that Module will be used as a loss function and make sure that
      the loss reduction is a sum reduction and not an elementwise mean.
    loss_params (dict, optional): loss function extra params based on loss module if ``loss`` is a ``str``.
      Ignored if ``loss`` is a ``torch.nn.Module``.
    use_cuda (bool, optional): use GPU when training/evaluating the model.
    index_ids (bool, optional): If ``True``, the item and user ids will be indexed. Used when those ids
      are strings, or integers but don't start with 0 and can have values much larger than the total
      number of items/users in the dataset. The ids index is provided by accessing ``Recoder.item_id_map``,
      and ``Recoder.user_id_map`` which maps from original id to the model id.
      If ``False``, the ids in the dataset should be integers, and the number of items/users will be
      equal to `max(item_ids) + 1` assuming item ids start with 0. Note: indexing the item ids can
      slightly slow the training process.
    user_based (bool, optional): If your model is based on users or not. If True, an exception will
      will be raised when there are inconsistencies between the users represented in the model
      and the users in the training datasets.
    item_based (bool, optional): If your model is based on items or not. If True, an exception will
      will be raised when there are inconsistencies between the items represented in the model
      and the items in the training datasets.
  """

  def __init__(self, model: FactorizationModel,
               num_items=None, num_users=None,
               optimizer_type='sgd', loss='mse',
               loss_params=None, use_cuda=False,
               index_ids=True, user_based=True,
               item_based=True):

    self.model = model
    self.num_items = num_items
    self.num_users = num_users
    self.optimizer_type = optimizer_type
    self.loss = loss
    self.loss_params = loss_params if loss_params else {}
    self.use_cuda = use_cuda
    self.index_ids = index_ids
    self.user_based = user_based
    self.item_based = item_based

    if self.use_cuda:
      self.device = torch.device('cuda')
    else:
      self.device = torch.device('cpu')

    self.item_id_map = None
    self.user_id_map = None
    self.optimizer = None
    self.sparse_optimizer = None
    self.current_epoch = 1
    self.items = None
    self.users = None
    self.__model_initialized = False
    self.__optimizer_state_dict = None
    self.__sparse_optimizer_state_dict = None

    self.__inverse_item_id_map = None

  def __init_model(self):
    if self.__model_initialized:
      return

    self.model.init_model(self.num_items, self.num_users)
    self.model = self.model.to(device=self.device)
    self.__model_initialized = True

  def __init_loss_module(self):
    if issubclass(self.loss.__class__, torch.nn.Module):
      self.loss_module = self.loss
    elif self.loss == 'logistic':
      self.loss_module = BCEWithLogitsLoss(reduction='sum', **self.loss_params)
    elif self.loss == 'mse':
      self.loss_module = MSELoss(reduction='sum', **self.loss_params)
    elif self.loss == 'logloss':
      self.loss_module = MultinomialNLLLoss(reduction='sum')
    elif self.loss is None:
      raise ValueError('No loss function defined')
    else:
      raise ValueError('Unknown loss function {}'.format(self.loss))

  def __init_optimizer(self, lr, weight_decay):
    # When continuing training on the same Recoder instance
    if self.optimizer is not None:
      self.__optimizer_state_dict = self.optimizer.state_dict()

    if self.sparse_optimizer is not None:
      self.__sparse_optimizer_state_dict = self.sparse_optimizer.state_dict()

    # Collecting sparse parameter names
    sparse_params_names = []
    sparse_modules = [torch.nn.Embedding, torch.nn.EmbeddingBag]
    for module_name, module in self.model.named_modules():
      if type(module) in sparse_modules and module.sparse:
        sparse_params_names.extend([module_name + '.' + param_name
                                    for param_name, param in module.named_parameters()])

    # Initializing optimizer params
    params = []
    sparse_params = []
    for param_name, param in self.model.named_parameters():
      _weight_decay = weight_decay

      if 'bias' in param_name:
        _weight_decay = 0

      if param_name in sparse_params_names:
        # If module is an embedding layer with sparse gradients
        # then add its parameters to sparse optimizer
        sparse_params.append({'params': param, 'weight_decay': _weight_decay})
      else:
        params.append({'params': param, 'weight_decay': _weight_decay})

    if self.optimizer_type == "adam":
      if len(params) > 0:
        self.optimizer = optim.Adam(params, lr=lr)

      if len(sparse_params) > 0:
        self.sparse_optimizer = optim.SparseAdam(sparse_params, lr=lr)

    elif self.optimizer_type == "adagrad":
      if len(sparse_params) > 0:
        raise ValueError('Sparse gradients optimization not supported with adagrad')

      self.optimizer = optim.Adagrad(params, lr=lr)
    elif self.optimizer_type == "sgd":
      if len(sparse_params) > 0:
        raise ValueError('Sparse gradients optimization not supported with sgd')

      self.optimizer = optim.SGD(params, lr=lr, momentum=0.9)
    elif self.optimizer_type == "rmsprop":
      if len(sparse_params) > 0:
        raise ValueError('Sparse gradients optimization not supported with rmsprop')

      self.optimizer = optim.RMSprop(params, lr=lr, momentum=0.9)
    else:
      raise Exception('Unknown optimizer kind')

    if self.__optimizer_state_dict is not None:
      self.optimizer.load_state_dict(self.__optimizer_state_dict)
      self.__optimizer_state_dict = None # no need for this anymore

    if self.__sparse_optimizer_state_dict is not None and self.sparse_optimizer is not None:
      self.sparse_optimizer.load_state_dict(self.__sparse_optimizer_state_dict)
      self.__sparse_optimizer_state_dict = None

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
    model_params = model_saved_state['model_params']
    self.item_id_map = model_saved_state['item_id_map']
    self.current_epoch = model_saved_state['last_epoch']
    self.loss = model_saved_state.get('loss', None)
    self.loss_params = model_saved_state.get('loss_params', None)
    self.optimizer_type = model_saved_state['optimizer_type']
    self.items = model_saved_state.get('items', None) or self.num_items
    self.users = model_saved_state.get('users', None)
    self.num_items = model_saved_state.get('num_items', None)
    self.num_users = model_saved_state.get('num_users', None)
    self.__optimizer_state_dict = model_saved_state['optimizer']
    self.__sparse_optimizer_state_dict = model_saved_state.get('sparse_optimizer', None)

    # old versions backward compatibility
    if 'hidden_layers' in model_saved_state:
      model_params['hidden_layers'] = model_saved_state['hidden_layers']
    if 'vector_dim' in model_saved_state:
      self.num_items = model_saved_state['vector_dim']

    self.model.load_model_params(model_params)
    self.__init_model()
    self.model.load_state_dict(model_saved_state['model'])

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
      'recoder_version': __version__,
      'model_params': self.model.model_params(),
      'item_id_map': self.item_id_map,
      'user_id_map': self.user_id_map,
      'last_epoch': self.current_epoch,
      'model': self.model.state_dict(),
      'optimizer_type': self.optimizer_type,
      'optimizer': self.optimizer.state_dict(),
      'items': self.items,
      'users': self.users,
      'num_items': self.num_items,
      'num_users': self.num_users
    }

    if self.loss is str:
      current_state['loss'] = self.loss,
      current_state['loss_params'] = self.loss_params

    torch.save(current_state, checkpoint_file)
    return checkpoint_file

  def __init_training(self, train_dataset, lr,
                      weight_decay):
    if self.items is None:
      self.items = train_dataset.items
    else:
      self.items = list(set(self.items + train_dataset.items))

    if self.users is None:
      self.users = train_dataset.users
    else:
      self.users = list(set(self.users + train_dataset.users))

    if self.index_ids:
      if self.num_users is None:
        self.num_users = len(self.users)
      elif self.user_based:
        assert self.num_users >= len(self.users), \
          'Number of users in the model should be greater or equal than the number of users in the dataset.' \
          'If your model is independent of users, set user_dependent to False in Recoder constructor.'

      if self.user_id_map is None:
        self.user_id_map = dict([(user_id, idx) for idx, user_id in enumerate(self.users)])
      elif self.user_based:
        assert len(np.setdiff1d(self.users, list(self.user_id_map.keys()))) == 0, \
          "There are users in the dataset that doesn't exist in the items index." \
          "If your model is not based on users, set user_based to False in Recoder constructor."

      if self.num_items is None:
        self.num_items = len(self.items)
      elif self.item_based:
        assert self.num_items >= len(self.items), \
          'Number of items in the model should be greater or equal than the number of items in the dataset.' \
          'If your model is not based on items, set item_based to False in Recoder constructor.'

      if self.item_id_map is None:
        self.item_id_map = dict([(item_id, idx) for idx, item_id in enumerate(self.items)])
      elif self.item_based:
        assert len(np.setdiff1d(self.items, list(self.item_id_map.keys()))) == 0, \
          "There are items in the dataset that doesn't exist in the items index." \
          "If your model is not based on items, set item_based to False in Recoder constructor."
    else:
      assert type(self.items[0]) is int, 'item ids should be integers, or set index_ids to True'
      assert type(self.users[0]) is int, 'user ids should be integers, or set index_ids to True'

      if self.num_items is None:
        self.num_items = int(np.max(self.items)) + 1
      elif self.item_based:
        assert self.num_items >= int(np.max(self.items)) + 1,\
          'The largest item id should be smaller than number of items.' \
          'If your model is not based on items, set item_based to False in Recoder constructor.'

      if self.num_users is None:
        self.num_users = int(np.max(self.users)) + 1
      elif self.user_based:
        assert self.num_users >= int(np.max(self.users)) + 1,\
          'The largest user id should be smaller than number of users.' \
          'If your model is not based on users, set user_based to False in Recoder constructor.'

    self.__init_model()
    self.__init_optimizer(lr=lr, weight_decay=weight_decay)
    self.__init_loss_module()

  def train(self, train_dataset, val_dataset=None,
            lr=0.001, weight_decay=0, num_epochs=1,
            batch_size=64, lr_milestones=None,
            num_neg_samples=0, num_sampling_users=0, num_data_workers=0,
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
      num_sampling_users (int, optional): number of users to consider for sampling items.
        This is useful for increasing the number of negative samples in mini-batch based negative
        sampling while keeping the batch-size small. If 0, then num_sampling_users will
        be equal to batch_size.
      num_data_workers (int, optional): number of data workers to use for building the mini-batches.
      checkpoint_freq (int, optional): epochs frequency of saving a checkpoint of the model
      model_checkpoint_prefix (str, optional): model checkpoint save path prefix
      eval_freq (int, optional): epochs frequency of doing an evaluation
      eval_num_recommendations (int, optional): num of recommendations to generate on evaluation
      metrics (list[Metric], optional): list of ``Metric`` used to evaluate the model
    """
    log.info('{} Mode'.format('CPU' if self.device.type == 'cpu' else 'GPU'))
    model_params = self.model.model_params()
    for param in model_params:
      log.info('Model {}: {}'.format(param, model_params[param]))
    log.info('Initial Learning Rate: {}'.format(lr))
    log.info('Weight decay: {}'.format(weight_decay))
    log.info('Batch Size: {}'.format(batch_size))
    log.info('Optimizer: {}'.format(self.optimizer_type))
    log.info('LR milestones: {}'.format(lr_milestones))
    log.info('Loss Function: {}'.format(self.loss))
    for param in self.loss_params:
      log.info('Loss {}: {}'.format(param, self.loss_params[param]))

    if num_sampling_users == 0:
      num_sampling_users = batch_size

    assert num_sampling_users >= batch_size and num_sampling_users % batch_size == 0, \
      "number of sampling users should be a multiple of the batch size"

    self.__init_training(train_dataset=train_dataset, lr=lr, weight_decay=weight_decay)

    train_dataloader = RecommendationDataLoader(train_dataset, batch_size=batch_size,
                                                vector_dim=self.num_items,
                                                num_neg_samples=num_neg_samples,
                                                item_id_map=self.item_id_map,
                                                user_id_map=self.user_id_map,
                                                num_sampling_users=num_sampling_users,
                                                num_workers=num_data_workers)
    if val_dataset is not None:
      val_dataloader = RecommendationDataLoader(val_dataset, batch_size=batch_size,
                                                vector_dim=self.num_items,
                                                num_neg_samples=num_neg_samples,
                                                item_id_map=self.item_id_map,
                                                user_id_map=self.user_id_map,
                                                num_sampling_users=num_sampling_users,
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
      self.model.train()
      aggregated_losses = []
      if lr_scheduler is not None:
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]
      else:
        lr = self.optimizer.defaults['lr']
      description = 'Epoch {}/{} (lr={})'.format(epoch, num_epochs, lr)
      progress_bar = tqdm(range(num_batches), desc=description)
      for batch_itr, (input, target) in enumerate(train_dataloader, 1):
        if self.optimizer is not None:
          self.optimizer.zero_grad()

        if self.sparse_optimizer is not None:
          self.sparse_optimizer.zero_grad()

        if target is None:
          target_items = input.items
        else:
          target_items = target.items

        loss = self.__compute_loss(input, target)

        loss.backward()
        if self.optimizer is not None:
          self.optimizer.step()

        if self.sparse_optimizer is not None:
          self.sparse_optimizer.step()

        aggregated_losses.append(loss.item())
        progress_bar.set_postfix(loss=np.mean(aggregated_losses[-1]),
                                 num_items=target_items.size(0),
                                 refresh=False)
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
    self.model.eval()

    total_loss = 0.0
    num_batches = 1

    for itr, (input, target) in enumerate(val_dataloader):
      loss = self.__compute_loss(input, target)
      total_loss += loss.item()
      num_batches = itr + 1

    avg_loss = total_loss / num_batches

    return avg_loss

  def __compute_loss(self, input, target):
    input_items = input.items
    input_users = input.users
    input_dense = torch.sparse.FloatTensor(input.indices, input.values, input.size) \
      .to(device=self.device).to_dense()
    if input_items is not None:
      input_items = input_items.to(device=self.device)
    if input_users is not None:
      input_users = input_users.to(device=self.device)

    if target is not None:
      target_items = target.items
      target_users = target.users
      target_dense = torch.sparse.FloatTensor(target.indices, target.values, target.size) \
        .to(device=self.device).to_dense()
      if target_items is not None:
        target_items = target_items.to(device=self.device)
      if target_users is not None:
        target_users = target_users.to(device=self.device)
    else:
      target_dense = input_dense
      target_items = input_items
      target_users = input_users

    output = self.model(input_dense, input_items=input_items,
                        input_users=input_users, target_items=target_items,
                        target_users=target_users)

    # Average loss over samples in a batch
    normalization = torch.FloatTensor([target_dense.size(0)]).to(device=self.device)
    loss = self.loss_module(output, target_dense) / normalization
    return loss

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
    if self.model is None:
      raise Exception('Model not initialized.')

    self.model.eval()

    batch_collator = BatchCollator(batch_size=len(users_hist), vector_dim=self.num_items,
                                   num_neg_samples=-1, item_id_map=self.item_id_map,
                                   user_id_map=self.user_id_map)

    input = batch_collator.collate(users_hist)
    batch = input[0]
    input_dense = torch.sparse.FloatTensor(batch.indices, batch.values, batch.size) \
      .to(device=self.device).to_dense()
    output = self.model(input_dense, input_users=batch.users.to(device=self.device)).cpu()
    return output, input_dense.cpu() if return_input else output

  def _evaluate(self, eval_dataset, num_recommendations, metrics, batch_size=1):
    if self.model is None:
      raise Exception('Model not initialized')

    self.model.eval()
    recommender = InferenceRecommender(self, num_recommendations)

    evaluator = RecommenderEvaluator(recommender, metrics)

    results = evaluator.evaluate(eval_dataset, batch_size=batch_size)
    return results

  def recommend(self, users_hist, num_recommendations):
    """
    Generate list of recommendations for each user in ``users_hist``.

    Args:
      users_hist (list[UserInteractions]): list of users interactions.
      num_recommendations: number of recommendations to generate.

    Returns:
      list: list of recommended items for each user in users_hist.
    """
    output, input = self.predict(users_hist, return_input=True)
    input = input.numpy()
    output = output.data.numpy()

    output[input > 0] = - np.inf

    top_ind = np.argpartition(-output, num_recommendations, axis=1)
    top_ind = top_ind[:, :num_recommendations]
    top_output = output[np.arange(output.shape[0])[:, None], top_ind]

    top_sorted_reset_ind = np.argsort(-top_output, axis=1)

    top_sorted_ind = top_ind[np.arange(top_ind.shape[0])[:, None], top_sorted_reset_ind]

    # Map the recommended model item ids back to original item ids
    inverse_item_id_map = self.__get_inverse_item_id_map()
    if inverse_item_id_map is not None:
      item_id_mapper = np.vectorize(lambda item_id: inverse_item_id_map[item_id])
      recommendations = item_id_mapper(top_sorted_ind)
    else:
      recommendations = top_sorted_ind.tolist()

    return recommendations

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

  def __get_inverse_item_id_map(self):
    if self.item_id_map is None:
      return None

    if self.__inverse_item_id_map is None:
      self.__inverse_item_id_map = {model_item_id:item_id
                                    for item_id, model_item_id in self.item_id_map.items()}

    return self.__inverse_item_id_map