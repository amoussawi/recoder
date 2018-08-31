from torch.utils.data import Dataset

import numpy as np

import collections


__Interaction = collections.namedtuple('__Interaction', ['item_id', 'inter'])

class Interaction(__Interaction):
  """
  Represents a single interaction of a user with an item.

  Args:
    item_id (int or str): item id
    inter (float): interaction value
  """
  pass


class RecommendationDataset(Dataset):
  """
  Represents a ``Dataset`` that iterates through the users interactions with items.

  Indexing the dataset will return a tuple of the user input and target list of ``Interaction``.
  In case a ``target_dataset`` was not provided, which is usually the case for training,
  the target is the same as input, otherwise the target is the user interactions in the ``target_dataset``.

  Args:
    data (pandas.DataFrame): DataFrame that contains user-item interactions
    target_dataset (RecommendationDataset, optional): RecommendationDataset that contains
      the interactions to recommend.
    user_col (str, optional): user column name
    item_col (str, optional): item column name
    inter_col (str, optional): interaction column name
  """

  def __init__(self, data, target_dataset=None,
               user_col='user', item_col='item', inter_col='inter'):

    self.data = data

    self.user_col = user_col
    self.item_col = item_col
    self.inter_col = inter_col
    self.target_dataset = target_dataset # type: RecommendationDataset

    self.__load_data()

  def __load_data(self):
    self.users = self.data[self.user_col].unique().tolist()
    self.items = self.data[self.item_col].unique().tolist()

    if self.target_dataset is not None:
      self.users = list(set(self.users + self.target_dataset.users))
      self.items = list(set(self.items + self.target_dataset.items))

    _grouped_data_df = self.data.groupby(by=self.user_col)

    if self.target_dataset is None:
      self.__groups = list(_grouped_data_df.groups.keys())
    else:
      _groups = list(_grouped_data_df.groups.keys())
      # to avoid having groups not in the target dataset
      self.__groups = list(np.intersect1d(_groups, self.target_dataset.__groups))

    self.__groups_index = {g:i for i, g in enumerate(self.__groups)}
    self.__grouped_data = _grouped_data_df

    self.__interactions = [-1] * len(self.__groups)

  def __get_interactions(self, index):
    if self.__interactions[index] != -1:
      return self.__interactions[index]

    group = self.__groups[index]
    _data = self.__grouped_data.get_group(group)

    interactions = []
    for item_id, inter in zip(_data[self.item_col], _data[self.inter_col]):
      interactions.append(Interaction(item_id=item_id, inter=inter))

    self.__interactions[index] = interactions
    return interactions

  def __len__(self):
    return len(self.__interactions)

  def __getitem__(self, index):
    interactions = self.__get_interactions(index)

    if self.target_dataset is not None:
      _group = self.__groups[index]
      target_index = self.target_dataset.__groups_index[_group]
      target_interactions = self.target_dataset.__get_interactions(target_index)
    else:
      target_interactions = None

    if target_interactions is None:
      return interactions
    else:
      return interactions, target_interactions

  def preload(self):
    """
    Preloads the data into memory
    """
    for i in range(len(self)):
      self[i]
