import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import recoder.utils as utils

from concurrent.futures import ProcessPoolExecutor


class UserInteractions:
  """
  Represents the interactions of a user

  Args:
    user (str or int): user id
    items (list): list of items the user interacted with
    values (list): values of the interactions between the user and the items
  """
  def __init__(self, user, items, values):
    self.user = user
    self.items = items
    self.values = values


def _dataframe_to_interactions(dataframe, user_col='user',
                               item_col='item', inter_col='inter'):
  grouped_data_df = dataframe.groupby(by=user_col)

  users = list(grouped_data_df.groups.keys())

  interactions = {}

  for user in users:
    user_data = grouped_data_df.get_group(user)
    interactions[user] = UserInteractions(user=user, items=user_data[item_col].tolist(),
                                          values=user_data[inter_col].tolist())

  return interactions


class RecommendationDataset(Dataset):
  """
  Represents a :class:`torch.utils.data.Dataset` that iterates through the users interactions with items.

  Indexing the dataset will return a tuple of the user input and target :class:`UserInteractions`
  if a ``target_dataset`` is provided, otherwise the target is ``None``.

  Args:
    target_dataset (RecommendationDataset, optional): RecommendationDataset that contains
      the interactions to recommend.
  """

  def __init__(self, target_dataset=None):
    self.target_dataset = target_dataset # type: RecommendationDataset
    self.__interactions = {}
    self.users = []
    self.items = []

  def fill_from_dataframe(self, dataframe, num_workers=0,
                          user_col='user', item_col='item',
                          inter_col='inter'):
    """
    Fills the dataset from a ``pandas.DataFrame`` where each row represents an interaction
    between a user and an item and the value of that interaction.

    Args:
      dataframe (pandas.DataFrame): DataFrame that contains user-item interactions
      num_workers (int, optional): Number of workers to use to fill the data
      user_col (str, optional): user column name
      item_col (str, optional): item column name
      inter_col (str, optional): interaction value column name
    """
    dataframe_users = dataframe[user_col].unique().tolist()
    dataframe_items = dataframe[item_col].unique().tolist()
    self.users = list(set(self.users + dataframe_users))
    self.items = list(set(self.items + dataframe_items))

    if num_workers > 0:
      pool_exec = ProcessPoolExecutor(num_workers)
      futures = []
      chunk_size = int(len(dataframe_users) / num_workers) + 1
      users_chunks = [dataframe_users[offset:offset + chunk_size]
                      for offset in range(0, len(dataframe_users), chunk_size)]

      for users_chunk in users_chunks:
        dataframe_chunk = dataframe[dataframe[user_col].isin(users_chunk)]
        futures.append(pool_exec.submit(_dataframe_to_interactions, dataframe_chunk,
                                        user_col=user_col, item_col=item_col,
                                        inter_col=inter_col))

      for future in futures:
        self.__interactions.update(future.result())

      pool_exec.shutdown()
    else:
      self.__interactions.update(_dataframe_to_interactions(dataframe, user_col=user_col,
                                                            item_col=item_col, inter_col=inter_col))

  def __len__(self):
    return len(self.users)

  def __getitem__(self, index):
    if index >= len(self):
      raise IndexError(index)

    user = self.users[index]
    if self.target_dataset is None:
      return self.__interactions[user], None
    else:
      return self.__interactions[user], self.target_dataset.__interactions[user]


class RecommendationDataLoader:
  """
  A ``DataLoader`` similar to ``torch.utils.data.DataLoader`` that handles
  :class:`RecommendationDataset` and generate batches with negative sampling.

  Args:
    dataset (RecommendationDataset): dataset from which to load the data
    batch_size (int): number of samples per batch
    vector_dim (int): the dimension size of the input vector (number of items)
    num_neg_samples (int, optional): number of negative samples to generate for each user.
      If `-1`, then all possible negative items will be sampled. If `0`, the negative items
      are sampled with mini-batch based negative sampling. If `> 0`, the negative items
      are sampled with mini-batch based negative sampling in addition to common random negative
      items if needed.
    num_sampling_users (int, optional): number of users to consider for mini-batch based negative
      sampling. This is useful for increasing the number of negative samples while keeping the
      batch-size small. If 0, then num_sampling_users will be equal to batch_size.
    item_id_map (dict, optional): a map from original item id to the model item id
    user_id_map (dict, optional): a map from original user id to the model user id
    num_workers (int, optional): how many subprocesses to use for data loading.
  """
  def __init__(self, dataset, batch_size,
               vector_dim, num_neg_samples=-1,
               num_sampling_users=0, item_id_map=None,
               user_id_map=None, num_workers=0):
    self.dataset = dataset # type: RecommendationDataset
    self.num_sampling_users = num_sampling_users
    self.num_workers = num_workers
    self.batch_size = batch_size
    self.vector_dim = vector_dim
    self.num_neg_samples = num_neg_samples
    self.item_id_map = item_id_map
    self.user_id_map = user_id_map

    if self.num_sampling_users == 0:
      self.num_sampling_users = batch_size

    self.batch_collator = BatchCollator(batch_size=self.batch_size, vector_dim=self.vector_dim,
                                        num_neg_samples=self.num_neg_samples, item_id_map=self.item_id_map,
                                        user_id_map=user_id_map)

    self._dataloader = DataLoader(dataset, batch_size=self.num_sampling_users,
                                  shuffle=True, num_workers=num_workers,
                                  collate_fn=self.__collate_input_target)

  def __generator(self):
    for input, target in self._dataloader:
      for batch_ind in range(len(input)):
        if target is None:
          yield input[batch_ind], None
        else:
          yield input[batch_ind], target[batch_ind]

  def __collate_input_target(self, batch):
    _input_batch, _target_batch = utils.unzip(batch)

    input = self.batch_collator.collate(_input_batch)

    if _target_batch[0] is None:
      target = None
    else:
      target = self.batch_collator.collate(_target_batch)

    return input, target

  def __iter__(self):
    return self.__generator()

  def __len__(self):
    return int(np.ceil(len(self.dataset) / self.batch_collator.batch_size))


class Batch:
  """
  Represents a sparse batch of users and items interactions.

  Args:
    users (torch.LongTensor): users that are in the batch
    items (torch.LongTensor): items that are in the batch
    indices (torch.LongTensor): the indices of the interactions in the sparse matrix
    values (torch.LongTensor): the values of the interactions
    size (torch.Size): the size of the sparse interactions matrix
  """
  def __init__(self, users, items,
               indices, values, size):
    self.users = users
    self.items = items
    self.indices = indices
    self.values = values
    self.size = size


class BatchCollator:
  """
  Collator of lists of :class:`UserInteractions`. It collates the samples into multiple :class:`Batch`
  based on ``batch_size``.

  Args:
    batch_size (int): number of samples per batch
    vector_dim (int): the dimension size of the input vector (number of items)
    num_neg_samples (int, optional): number of negative samples to generate for each user.
      If `-1`, then all possible negative items will be sampled. If `0`, the negative items
      are sampled with mini-batch based negative sampling. If `> 0`, the negative items
      are sampled with mini-batch based negative sampling in addition to common random negative
      items if needed.
    item_id_map (dict, optional): a map from original item id to the model item id
    user_id_map (dict, optional): a map from original user id to the model user id
  """
  def __init__(self, batch_size, vector_dim,
               num_neg_samples=-1, item_id_map=None,
               user_id_map=None):
    self.batch_size = batch_size
    self.vector_dim = vector_dim
    self.item_id_map = item_id_map
    self.user_id_map = user_id_map
    self.num_neg_samples = num_neg_samples

  def collate(self, samples):
    """
    Collates samples of :class:`UserInteractions` into batches of size ``batch_size``.

    Args:
      samples (list): list of lists of :class:`UserInteractions`.

    Returns:
      list[Batch]: list of batches.
    """
    users_inds = []
    items_inds = []
    inter_vals = []
    batch_users = []
    examples_offsets = []
    for sample_i, user_interactions in enumerate(samples):
      num_inters = len(user_interactions.items)
      users_inds.extend([sample_i % self.batch_size] * num_inters)
      user = user_interactions.user
      user_items, user_inter_vals = user_interactions.items, user_interactions.values

      # mapping the original user and item ids to the model item ids if available
      if self.item_id_map is not None:
        user_items = map(lambda item_id: self.item_id_map[item_id], user_items)
      if self.user_id_map is not None:
        user = self.user_id_map[user]

      batch_users.append(user)
      items_inds.extend(user_items)
      inter_vals.extend(user_inter_vals)
      if (sample_i + 1) % self.batch_size == 0 or (sample_i + 1) == len(samples):
        examples_offsets.append(len(users_inds))

    if self.num_neg_samples >= 0:
      negative_items = np.random.randint(0, self.vector_dim, self.num_neg_samples)
      negative_items = negative_items[np.isin(negative_items, items_inds, invert=True)]
      num_negative_items = len(negative_items)

      # It's enough to fill the first sample in the batch with negative items
      # the others will be filled by transforming the sparse matrix into dense
      if num_negative_items > 0:
        users_inds.extend([0] * num_negative_items)
        items_inds.extend(negative_items)
        inter_vals.extend([0] * num_negative_items)

      # The positive item ids in the batch
      batch_items = np.unique(items_inds)

      # Reindex the item ids so they start with 0 and make
      # max(new_ids) = len(batch_items) so that the sparse
      # batch only contains non-zero columns
      new_map = dict([(v, ind) for ind, v in enumerate(batch_items)])
      items_inds = list(map(lambda x: new_map[x], items_inds))

      _vector_dim = len(batch_items)
      batch_items = torch.LongTensor(batch_items)
    else:
      _vector_dim = self.vector_dim
      batch_items = None

    batch_users = torch.LongTensor(batch_users)
    slices = []
    prev_offset = 0
    for itr, example_offset in enumerate(examples_offsets):
      slice_batch_users = batch_users[itr*self.batch_size : (itr+1)*self.batch_size]
      slice_users_inds = users_inds[prev_offset : example_offset]
      slice_items_inds = items_inds[prev_offset : example_offset]
      slice_inter_vals = inter_vals[prev_offset : example_offset]
      prev_offset = example_offset

      indices = torch.LongTensor([slice_users_inds, slice_items_inds])
      values = torch.FloatTensor(slice_inter_vals)

      slice_size = slice_users_inds[-1] + 1

      slices.append(Batch(items=batch_items, users=slice_batch_users,
                          indices=indices, values=values,
                          size=torch.Size([slice_size, _vector_dim])))

    return slices
