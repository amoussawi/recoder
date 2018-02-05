import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as weight_init
import torch.sparse as sparse
from torch.autograd import Variable

import recoder.autoencoder.functional as functional


class AutoEncoder(nn.Module):
  def __init__(self, layer_sizes, activation_type='selu', last_layer_act='none',
               is_constrained=False, dp_drop_prob=0.0):
    super(AutoEncoder, self).__init__()

    self._activation_type = activation_type
    if type(self._activation_type) is str:
      self._activation_type = [self._activation_type] * (len(layer_sizes) - 1)

    self._e_activation_type = self._activation_type
    self.layer_sizes = layer_sizes

    self._d_activation_type = list(self._activation_type)
    self._d_activation_type.reverse()
    del self._d_activation_type[0]
    self._d_activation_type.append(last_layer_act)

    self._dp_drop_prob = dp_drop_prob
    if type(self._dp_drop_prob) is float:
      self._dp_drop_prob = [self._dp_drop_prob] * (len(layer_sizes) - 1)

    self._last_layer_act = last_layer_act
    self.encode_w = nn.ParameterList(
      [nn.Parameter(torch.rand(layer_sizes[i + 1], layer_sizes[i])) for i in range(len(layer_sizes) - 1)])
    for ind, w in enumerate(self.encode_w):
      weight_init.xavier_uniform(w)

    self.encode_b = nn.ParameterList(
      [nn.Parameter(torch.zeros(layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)])

    reversed_enc_layers = list(reversed(layer_sizes))

    self.is_constrained = is_constrained
    if not is_constrained:
      self.decode_w = nn.ParameterList(
      [nn.Parameter(torch.rand(reversed_enc_layers[i + 1], reversed_enc_layers[i])) for i in range(len(reversed_enc_layers) - 1)])
      for ind, w in enumerate(self.decode_w):
        weight_init.xavier_uniform(w)
    else:
      self.decode_w = list(reversed(self.encode_w))
      for ind, w in enumerate(self.decode_w):
        self.decode_w[ind] = w.permute(1,0)


    self.decode_b = nn.ParameterList(
      [nn.Parameter(torch.zeros(reversed_enc_layers[i + 1])) for i in range(len(reversed_enc_layers) - 1)])

  def update_constrained_decode_w(self):
    self.decode_w = list(reversed(self.encode_w))
    for ind, w in enumerate(self.decode_w):
      self.decode_w[ind] = w.permute(1,0)

  def encode(self, x):
    return functional.encode(x, self.encode_w, self.encode_b,
                             self._e_activation_type, dp_drop_prob=self._dp_drop_prob,
                             training=self.training)

  def decode(self, z):
    if self.is_constrained and self.training:
      self.update_constrained_decode_w()
    return functional.decode(z, self.decode_w, self.decode_b,
                             self._d_activation_type, dp_drop_prob=self._dp_drop_prob,
                             training=self.training)


  def forward(self, x):
    return self.decode(self.encode(x))


class SparseBatchAutoEncoder(nn.Module):

  def __init__(self, auto_encoder, sparse_batch_in, sparse_batch_out=None, full_output=False):
    super(SparseBatchAutoEncoder, self).__init__()

    self.auto_encoder = auto_encoder

    self._activation_type = auto_encoder._activation_type
    self._e_activation_type = auto_encoder._e_activation_type
    self._d_activation_type = auto_encoder._d_activation_type

    self._dp_drop_prob = auto_encoder._dp_drop_prob
    self.full_output = full_output
    self.training = auto_encoder.training
    self.is_constrained = auto_encoder.is_constrained

    self._sparse_batch_in = sparse_batch_in
    self.reduced_batch_in, self.active_inputs, self.active_inputs_map, self.active_inputs_inverse_map \
                              = self.__generate_reduced_batch(self._sparse_batch_in)

    if sparse_batch_out is None:
      self._sparse_batch_out = self._sparse_batch_in
      self.active_outputs = self.active_inputs
      self.active_outputs_map = self.active_inputs_map
      self.active_outputs_inverse_map = self.active_inputs_inverse_map
    else:
      self._sparse_batch_out = sparse_batch_out
      self.reduced_batch_out, self.active_outputs, self.active_outputs_map, self.active_outputs_inverse_map \
                    = self.__generate_reduced_batch(self._sparse_batch_out)

    self.__init_encode_w()
    self.__init_decode_w()


  def __generate_reduced_batch(self, sparse_batch):
    if type(sparse_batch) is Variable:
      sparse_batch = sparse_batch.data

    if type(sparse_batch) is not sparse.FloatTensor:
      raise ValueError('expected a torch.sparse.FloatTensor')

    active_dim = sparse_batch._indices()[1]
    active_dim = torch.from_numpy(np.unique(active_dim.numpy()))
    active_dim_map = {}
    active_dim_inverse_map = {}
    for ind, inp in enumerate(active_dim):
      active_dim_map[inp] = ind
      active_dim_inverse_map[ind] = inp

    reduced_batch_size = torch.Size([sparse_batch.size(0),active_dim.size(0)])
    reduced_batch = torch.zeros(reduced_batch_size)

    _indices = sparse_batch._indices()
    for i in range(_indices.size(1)):
      reduced_batch[_indices[0][i]][active_dim_map[_indices[1][i]]] = sparse_batch._values()[i]

    return Variable(reduced_batch), Variable(active_dim), active_dim_map, active_dim_inverse_map

  def __init_encode_w(self):
    active_inputs = self.active_inputs
    _encode_w = self.auto_encoder.encode_w
    _encode_b = self.auto_encoder.encode_b

    _last = len(_encode_w) - 1

    self.encode_w = [_encode_w[0].index_select(1, active_inputs)] \
                    + [_encode_w[i] for i in range(1,len(_encode_w))]

    self.encode_b = _encode_b

  def __init_decode_w(self):
    _decode_w = self.auto_encoder.decode_w
    _decode_b = self.auto_encoder.decode_b

    if not self.full_output:
      active_outputs = self.active_outputs

      _last = len(_decode_w) - 1

      self.decode_w = [_decode_w[i] for i in range(len(_decode_w) - 1)] \
                      + [_decode_w[_last].index_select(0, active_outputs) ]

      self.decode_b = [_decode_b[i] for i in range(len(_decode_b) - 1)] \
                      + [_decode_b[_last].index_select(0, active_outputs)]
    else:
      self.decode_w = _decode_w
      self.decode_b = _decode_b

  def encode(self, x):
    return functional.encode(x, self.encode_w, self.encode_b,
                             self._e_activation_type, dp_drop_prob=self._dp_drop_prob,
                             training=self.training)

  def decode(self, z):
    if self.is_constrained and self.training:
      self.auto_encoder.update_constrained_decode_w()
      self.__init_decode_w()

    return functional.decode(z, self.decode_w, self.decode_b,
                             self._d_activation_type, dp_drop_prob=self._dp_drop_prob,
                             training=self.training)

  def forward(self, x=None):
    if x is None:
      x = self.reduced_batch_in
    return self.decode(self.encode(x))


class Noise(nn.Module):
  def __init__(self):
    super(Noise, self).__init__()

  def forward(self, input):
    raise NotImplementedError


class BernoulliNoise(Noise):
  def __init__(self, pb, noise=0):
    super(BernoulliNoise, self).__init__()
    self.pb = pb
    self.noise = noise

  def forward(self, input):
    corrupted_input = functional.bernoulli_noise(input=input, pb=self.pb, noise=self.noise)
    return corrupted_input


class GaussianNoise(Noise):
  def __init__(self):
    super(GaussianNoise, self).__init__()

  def forward(self, input):
    raise NotImplementedError
