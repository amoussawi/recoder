# Copyright (c) 2017 NVIDIA Corporation
"""Data Layer Classes"""
from os import listdir, path
from random import shuffle
import torch
from data_utils.utils import read_csv
import numpy as np

class UserItemRecDataProvider:
  def __init__(self, params, user_id_map=None, item_id_map=None, only_maps=False):
    self._params = params
    self._data_dir = self.params['data_dir']
    self._extension = ".txt" if 'extension' not in self.params else self.params['extension']
    self._i_id = 0 if 'itemIdInd' not in self.params else self.params['itemIdInd']
    self._u_id = 1 if 'userIdInd' not in self.params else self.params['userIdInd']
    self._r_id = 2 if 'ratingInd' not in self.params else self.params['ratingInd']
    self._major = 'items' if 'major' not in self.params else self.params['major']
    if not (self._major == 'items' or self._major == 'users'):
      raise ValueError("Major must be 'users' or 'items', but got {}".format(self._major))

    self._major_ind = self._i_id if self._major == 'items' else self._u_id
    self._minor_ind = self._u_id if self._major == 'items' else self._i_id
    self._delimiter = '\t' if 'delimiter' not in self.params else self.params['delimiter']
    self.rating_np_dtype = params['rating_np_dtype'] if 'rating_np_dtype' in params else np.float
    self.minor_np_dtype = params['minor_np_dtype'] if 'minor_np_dtype' in params else np.int
    if user_id_map is None or item_id_map is None:
      self._build_maps()
    else:
      self._user_id_map = user_id_map
      self._item_id_map = item_id_map
      self._user_id_inverse_map = dict()
      self._item_id_inverse_map = dict()
      for user in self._user_id_map:
        self._user_id_inverse_map[self._user_id_map[user]] = user
      for item in self._item_id_map:
        self._item_id_inverse_map[self._item_id_map[item]] = item

    major_map = self._item_id_map if self._major == 'items' else self._user_id_map
    minor_map = self._user_id_map if self._major == 'items' else self._item_id_map
    self._vector_dim = len(minor_map)

    src_files = [path.join(self._data_dir, f)
                  for f in listdir(self._data_dir)
                  if path.isfile(path.join(self._data_dir, f)) and f.endswith(self._extension)]

    self._batch_size = self.params['batch_size']
    self.only_maps = only_maps
    self.data = dict()
    if not self.only_maps:
      for source_file in src_files:
        for row in read_csv(source_file,has_columns=False,delimiter=self._delimiter):
          if len(row)<3:
            raise ValueError('Encountered badly formatted line in {}'.format(source_file))
          key = major_map[row[self._major_ind]]
          value = minor_map[row[self._minor_ind]]
          rating = float(row[self._r_id])
          #print("Key: {}, Value: {}, Rating: {}".format(key, value, rating))
          if key not in self.data:
            self.data[key] = {'minor':[], 'rating': []}
          self.data[key]['minor'].append(value)
          self.data[key]['rating'].append(rating)

      for major in self.data:
        self.data[major]['minor'] = np.array(self.data[major]['minor'], dtype=self.minor_np_dtype)
        self.data[major]['rating'] = np.array(self.data[major]['rating'], dtype=self.rating_np_dtype)

  def _build_maps(self):
    self._user_id_map = dict()
    self._item_id_map = dict()
    self._user_id_inverse_map = dict()
    self._item_id_inverse_map = dict()

    src_files = [path.join(self._data_dir, f)
                 for f in listdir(self._data_dir)
                 if path.isfile(path.join(self._data_dir, f)) and f.endswith(self._extension)]

    u_id = 0
    i_id = 0
    for source_file in src_files:
      for row in read_csv(source_file,has_columns=False,delimiter=self._delimiter):
        if len(row)<3:
          raise ValueError('Encountered badly formatted line in {}'.format(source_file))

        u_id_orig = row[self._u_id]
        if u_id_orig not in self._user_id_map:
          self._user_id_map[u_id_orig] = u_id
          self._user_id_inverse_map[u_id] = u_id_orig
          u_id += 1

        i_id_orig = row[self._i_id]
        if i_id_orig not in self._item_id_map:
          self._item_id_map[i_id_orig] = i_id
          self._item_id_inverse_map[i_id] = i_id_orig
          i_id += 1


  def iterate_one_epoch(self):
    data = self.data
    keys = list(data.keys())
    shuffle(keys)
    s_ind = 0
    e_ind = self._batch_size
    while s_ind < e_ind and e_ind <= len(keys):
      local_ind = 0
      inds1 = []
      inds2 = []
      vals = []
      for ind in range(s_ind, e_ind):
        inds2 += [np.int(v) for v in data[keys[ind]]['minor']]
        inds1 += [local_ind]*len(data[keys[ind]]['minor'])
        vals += [np.float(v) for v in data[keys[ind]]['rating']]
        local_ind += 1

      i_torch = torch.LongTensor([inds1, inds2])
      v_torch = torch.FloatTensor(vals)

      mini_batch = torch.sparse.FloatTensor(i_torch, v_torch, torch.Size([e_ind - s_ind, self._vector_dim]))

      s_ind += self._batch_size
      e_ind = min(e_ind + self._batch_size, len(keys))

      yield  mini_batch

  def iterate_one_epoch_eval(self, src_data_layer, for_inf=False):
    keys = list(self.data.keys())
    s_ind = 0
    src_data = src_data_layer.data
    shuffle(keys)
    s_ind = 0
    e_ind = self._batch_size
    while s_ind < e_ind and e_ind <= len(keys):
      local_ind = 0
      inds1 = []
      inds2 = []
      vals = []
      src_inds1 = []
      src_inds2 = []
      src_vals = []
      for ind in range(s_ind, e_ind):
        inds1 += [local_ind] * len(self.data[keys[ind]]['minor'])
        inds2 += [np.int(v) for v in self.data[keys[ind]]['minor']]
        vals += [np.float(v) for v in self.data[keys[ind]]['rating']]

        src_inds1 += [local_ind] * len(src_data[keys[ind]]['minor'])
        src_inds2 += [np.int(v) for v in src_data[keys[ind]]['minor']]
        src_vals += [np.float(v) for v in src_data[keys[ind]]['rating']]
        local_ind += 1

      i_torch = torch.LongTensor([inds1, inds2])
      v_torch = torch.FloatTensor(vals)

      src_i_torch = torch.LongTensor([src_inds1, src_inds2])
      src_v_torch = torch.FloatTensor(src_vals)

      mini_batch = (torch.sparse.FloatTensor(i_torch, v_torch, torch.Size([e_ind - s_ind, self._vector_dim])),
                    torch.sparse.FloatTensor(src_i_torch, src_v_torch, torch.Size([e_ind - s_ind, self._vector_dim])))

      s_ind += self._batch_size
      e_ind = min(e_ind + self._batch_size, len(keys))

      if not for_inf:
        yield  mini_batch
      else:
        yield mini_batch, keys[s_ind - 1]

  @property
  def vector_dim(self):
    return self._vector_dim

  @property
  def userIdMap(self):
    return self._user_id_map

  @property
  def itemIdMap(self):
    return self._item_id_map

  @property
  def userIdInverseMap(self):
    return self._user_id_inverse_map

  @property
  def itemIdInverseMap(self):
    return self._item_id_inverse_map

  @property
  def params(self):
    return self._params
