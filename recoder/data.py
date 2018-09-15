from torch.utils.data import Dataset

import collections
from concurrent.futures import ProcessPoolExecutor


__Interaction = collections.namedtuple('__Interaction', ['item_id', 'inter'])

class Interaction(__Interaction):
  """
  Represents a single interaction of a user with an item. It can
  be accessed as a tuple of item id and interaction value.

  Args:
    item_id (int or str): item id
    inter (float): interaction value
  """
  pass


def _dataframe_to_interactions(dataframe, user_col='user',
                               item_col='item', inter_col='inter'):
  grouped_data_df = dataframe.groupby(by=user_col)

  users = list(grouped_data_df.groups.keys())

  interactions = {}

  for user in users:
    user_data = grouped_data_df.get_group(user)
    interactions[user] = [Interaction(item_id, inter)
                          for item_id, inter in zip(user_data[item_col], user_data[inter_col])]

  return interactions


class RecommendationDataset(Dataset):
  """
  Represents a ``torch.utils.data.Dataset`` that iterates through the users interactions with items.

  If a ``target_dataset`` is provided, indexing the dataset will return a tuple of the user input
  and target list of ``Interaction``, otherwise only the input list of ``Interaction`` is returned.

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
      return self.__interactions[user]
    else:
      return self.__interactions[user], self.target_dataset.__interactions[user]
