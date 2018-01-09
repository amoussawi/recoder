import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch
import glog as log

class RecommendationDataset(Dataset):

  def __init__(self, data_file, user_based=True, target_dataset=None,
               user_id_map=None, item_id_map=None, user_dtype=np.int32,
               item_dtype=np.int32, inter_dtype=np.int32, user_col='user',
               item_col='item', inter_col='inter'):

    self.data_file = data_file

    self.user_based = user_based

    self.user_dtype = user_dtype
    self.item_dtype = item_dtype
    self.inter_dtype = inter_dtype

    self.user_col = user_col
    self.item_col = item_col
    self.inter_col = inter_col
    self.target_dataset = target_dataset # type: RecommendationDataset

    self.user_id_map = user_id_map
    self.item_id_map = item_id_map

    self.user_id_col = str(self.user_col) + '_id'
    self.item_id_col = str(self.item_col) + '_id'

    self.index_col = self.user_id_col if self.user_based else self.item_id_col
    self.target_col = self.item_id_col if self.user_based else self.user_id_col

    self.__item_cur_id = 0
    self.__user_cur_id = 0

    self.__load_data()

  def __load_data(self):
    log.info('Loading dataset from: ' + str(self.data_file))
    _pd_dtype = {
      self.inter_col: self.inter_dtype,
    }

    self.data = pd.read_csv(self.data_file, dtype=_pd_dtype)

    self.__build_users()
    self.__build_items()

    self.sample_dim = len(self.item_id_map) if self.user_based else len(self.user_id_map)

    _grouped_data_df = self.data.groupby(by=self.index_col)
    self.__groups = list(_grouped_data_df.groups.keys())

    self.__grouped_data = _grouped_data_df

  def __build_users(self):
    if self.user_id_map is None:
      self.user_id_map = {}

    self.data[self.user_id_col] = self.data[self.user_col].apply(
      func=lambda user: self.__map_user(user),
    )
    del self.data[self.user_col]
    self.data[self.user_id_col] = self.data[self.user_id_col].astype(self.user_dtype)

    self.user_map = dict([(v,k) for k,v in self.user_id_map.items()])

  def __build_items(self):
    if self.item_id_map is None:
      self.item_id_map = {}

    self.data[self.item_id_col] = self.data[self.item_col].apply(
      func=lambda item: self.__map_item(item),
    )
    del self.data[self.item_col]
    self.data[self.item_id_col] = self.data[self.item_id_col].astype(self.item_dtype)

    self.item_map = dict([(v,k) for k,v in self.item_id_map.items()])

  def __map_user(self, user):
    if not user in self.user_id_map:
      self.user_id_map[user] = self.__user_cur_id
      self.__user_cur_id += 1
    return self.user_id_map[user]

  def __map_item(self, item):
    if not item in self.item_id_map:
      self.item_id_map[item] = self.__item_cur_id
      self.__item_cur_id += 1
    return self.item_id_map[item]

  def get_user_id(self, user):
    return self.user_id_map[user]

  def get_item_id(self, item):
    return self.item_id_map[item]

  def get_user(self, user_id):
    return self.user_map[user_id]

  def get_item(self, item_id):
    return self.item_map[item_id]

  def __len__(self):
    return len(self.__groups)

  def __getitem__(self, index):
    _group = self.__groups[index]
    _data = self.__grouped_data.get_group(_group)

    _pairs = list(zip(_data[self.target_col], _data[self.inter_col]))

    if self.target_dataset is None:
      _target_data = _data
      _target_pairs = _pairs
    else:
      _target_data = self.target_dataset.__grouped_data.get_group(_group)
      _target_pairs = list(zip(_target_data[self.target_col], _target_data[self.inter_col]))

    return _pairs, _target_pairs

  def collate_to_sparse_batch(self, batch):
    _data_batch = [d for d, t in batch]
    _target_batch = [t for d, t in batch]

    _sparse_data_batch = self.__collate_to_sparse_batch(_data_batch)

    if self.target_dataset is not None:
      _sparse_target_batch = self.__collate_to_sparse_batch(_target_batch)
    else:
      _sparse_target_batch = _sparse_data_batch

    return _sparse_data_batch, _sparse_target_batch

  def __collate_to_sparse_batch(self, batch):
    samples_inds = []
    inter_inds = []
    inter_val = []
    batch_size = len(batch)

    for sample_i, sample in enumerate(batch):
      num_values = len(sample)
      samples_inds += [sample_i] * num_values
      for target, inter in sample:
        inter_inds.append(target)
        inter_val.append(inter)

    ind_lt = torch.LongTensor([samples_inds, inter_inds])
    val_ft = torch.FloatTensor(inter_val)

    batch = torch.sparse.FloatTensor(ind_lt, val_ft, torch.Size([batch_size, self.sample_dim]))

    return batch

