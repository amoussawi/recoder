import torch
from torch import nn
from recoder.functional import activation
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class SparseBatchAutoEncoder(nn.Module):

  def __init__(self, layer_sizes, activation_type='selu', last_layer_act='none',
               is_constrained=False, dropout_prob=0.0, noise_prob=0.0):
    super().__init__()
    self.num_embeddings = layer_sizes[0]
    self.embedding_size = layer_sizes[1]
    self.activation_type = activation_type
    self.last_layer_act = last_layer_act
    self.is_constrained = is_constrained
    self.layer_sizes = layer_sizes
    self.dropout_prob = dropout_prob
    self.noise_prob = noise_prob

    self.__create_encoding_layers()
    self.__create_decoding_layers()

    self.noise_layer = None
    if self.noise_prob > 0.0:
      self.noise_layer = nn.Dropout(p=self.noise_prob)

    self.dropout_layer = None
    if self.dropout_prob > 0.0:
      self.dropout_layer = nn.Dropout(p=self.dropout_prob)

    if self.is_constrained:
      self.__tie_weights()

  def __create_encoding_layers(self):
    self.en_embedding_layer = nn.Embedding(self.num_embeddings, self.embedding_size)

    self.__en_linear_embedding_layer = LinearEmbedding(self.en_embedding_layer, input_based=True)
    self.encoding_layers = nn.Sequential(*self.__create_coding_layers(self.layer_sizes[1:]))

    nn.init.xavier_uniform(self.en_embedding_layer.weight)
    nn.init.constant(self.__en_linear_embedding_layer.bias, 0)

  def __create_decoding_layers(self):
    _decoding_layers = self.__create_coding_layers(list(reversed(self.layer_sizes[1:])))
    if self.is_constrained:
      # This way the decoding layers are not stored as submodules and not optimized
      self.decoding_layers = _decoding_layers

      # Only register decoding layers biases as parameters
      for ind, decoding_layer in enumerate(self.decoding_layers):
        self.register_parameter('decoding_layer_bias_' + str(ind), decoding_layer.bias)
    else:
      self.decoding_layers = nn.Sequential(*_decoding_layers)

    if self.is_constrained:
      self.de_embedding_layer = self.en_embedding_layer
    else:
      self.de_embedding_layer = nn.Embedding(self.num_embeddings, self.embedding_size)

    self.__de_linear_embedding_layer = LinearEmbedding(self.de_embedding_layer, input_based=False)

    nn.init.xavier_uniform(self.de_embedding_layer.weight)
    nn.init.constant(self.__de_linear_embedding_layer.bias, 0)

  def __create_coding_layers(self, layer_sizes):
    layers = []
    for ind, layer_size in enumerate(layer_sizes[1:], 1):
      layer = nn.Linear(layer_sizes[ind-1], layer_size)
      layers.append(layer)
      torch.nn.init.xavier_uniform(layer.weight)
      torch.nn.init.constant(layer.bias, 0)

    return layers

  def __generate_reduced_batch(self, sparse_batch: torch.sparse.FloatTensor):
    if type(sparse_batch) is Variable:
      sparse_batch = sparse_batch.data

    # Getting the active indices along dim 1
    active_dim = sparse_batch._indices()[1]
    active_dim = torch.from_numpy(np.unique(active_dim.numpy()))
    active_dim_map = {}
    for ind, inp in enumerate(active_dim):
      active_dim_map[inp] = ind

    reduced_batch_size = torch.Size([sparse_batch.size(0),active_dim.size(0)])
    reduced_batch = torch.zeros(reduced_batch_size)

    # Fill the reduced batch with sparse batch values
    _indices = sparse_batch._indices()
    for i in range(_indices.size(1)):
      reduced_batch[_indices[0][i]][active_dim_map[_indices[1][i]]] = sparse_batch._values()[i]

    return Variable(reduced_batch), active_dim.long()

  def __tie_weights(self):
    for el, dl in zip(self.encoding_layers, reversed(self.decoding_layers)):
      dl.weight = nn.Parameter(el.weight.data.t())

  def forward(self, input: torch.sparse.FloatTensor, target=None,
              full_output=True):
    assert full_output or (not full_output and target is not None)

    if self.is_constrained:
      self.__tie_weights()

    reduced_input, in_active_embeddings = self.__generate_reduced_batch(input)

    dense_target = None
    out_active_embeddings = None
    if target is not None and full_output:
      dense_target = target.to_dense()
    elif target is not None:
      dense_target, out_active_embeddings = self.__generate_reduced_batch(target)

    z = reduced_input
    if self.noise_prob > 0.0:
      z = self.noise_layer(z)

    z = self.__en_linear_embedding_layer(in_active_embeddings, z)
    z = activation(z, self.activation_type)

    for encoding_layer in self.encoding_layers:
      z = activation(encoding_layer(z), self.activation_type)

    if self.dropout_prob > 0.0:
      z = self.dropout_layer(z)

    for decoding_layer in self.decoding_layers:
      z = activation(decoding_layer(z), self.activation_type)

    if not full_output:
      z = self.__de_linear_embedding_layer(out_active_embeddings, z)
    else:
      z = self.__de_linear_embedding_layer(None, z)

    z = activation(z, self.last_layer_act)

    if dense_target is None:
      return z
    else:
      return z, dense_target


class LinearEmbedding(nn.Module):

  def __init__(self, embedding_layer: nn.Embedding, input_based=True, bias=True):
    super().__init__()
    self.embedding_layer = embedding_layer
    self.input_based = input_based
    self.in_features = embedding_layer.num_embeddings if input_based else embedding_layer.embedding_dim
    self.out_features = embedding_layer.embedding_dim if input_based else embedding_layer.num_embeddings
    if bias:
      self.bias = nn.Parameter(torch.Tensor(self.out_features))
    else:
      self.bias = None

  def forward(self, x, y):
    if x is not None:
      _weight = self.embedding_layer(Variable(x))
      _bias = self.bias if self.input_based else self.bias.index_select(0, Variable(x))
    else:
      _weight = self.embedding_layer.weight
      _bias = self.bias

    if self.input_based:
      return F.linear(y, _weight.t(), _bias)
    else:
      return F.linear(y, _weight, _bias)
