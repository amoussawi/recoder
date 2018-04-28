import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from recoder.functional import activation

import numpy as np

class SparseBatchAutoEncoder(nn.Module):
  """
  An AutoEncoder module that processes sparse tensors efficiently for training.

  This module accepts as input a sparse tensor of shape (N, M), where N = mini-batch size,
  and M = input vector size. It will reshape and densify the sparse tensor to only use the
  columns that are non-zero.

  The efficiency of the model comes while training. A target sparse tensor can be passed, which will also
  be reshaped and densified, similar to the input, and returned as a dense tensor to be used
  with the output in the loss function. The columns of the target tensor that are non-zero are
  used to select which output units are necessary to compute for that batch. This can be done by setting
  full_output to False. This is useful when it's not necessary to reconstruct the zeros (explicit feedback)
  or when negative sampling on the zero output units is needed as the case in implicit feedback
  recommendations.

  Setting full_output to True will simply return a target.to_dense() and having the whole
  output units computed.

  Args:
    layer_sizes (list): autoencoder layer sizes. only input and encoder layers.
    activation_type (str, optional): activation function to use for hidden layers.
      all activations in torch.nn.functional are supported
    last_layer_act (str, optional): output layer activation function.
    is_constrained (bool, optional): constraining model by using the encoder weights in the
      decoder (tying the weights).
    dropout_prob (float, optional): dropout probability at the bottleneck layer
    noise_prob (float, optional): dropout (noise) probability at the input layer

  Examples::

    >>>> autoencoder = SparseBatchAutoEncoder([500,100])
    >>>> batch_size = 32
    >>>> i = torch.LongTensor([np.arange(45) % batch_size, np.arange(45)])
    >>>> v = torch.FloatTensor(np.ones(45))
    >>>> sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size([32,500]))
    >>>> dense_output = autoencoder(sparse_tensor)
    >>>> dense_output
       0.0850  0.9490  ...   0.2430  0.5323
       0.3519  0.4816  ...   0.9483  0.2497
            ...         ⋱         ...
       0.8744  0.8194  ...   0.5755  0.2090
       0.5006  0.9532  ...   0.8333  0.4330
      [torch.FloatTensor of size 32x500]
    >>>> dense_output, dense_target = autoencoder(sparse_tensor, target=sparse_tensor, full_output=False)
    >>>> dense_output # shape = 32x45 because only 45 columns are non-zero
       0.0850  0.9490  ...   0.2430  0.5323
       0.3519  0.4816  ...   0.9483  0.2497
            ...         ⋱         ...
       0.8744  0.8194  ...   0.5755  0.2090
       0.5006  0.9532  ...   0.8333  0.4330
      [torch.FloatTensor of size 32x45]
  """

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
    self.use_cuda = False

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
      for ind, decoding_layer in enumerate(_decoding_layers):
        # Deleting layer weight to unregister it as a parameter
        # Only register decoding layers biases as parameters
        del decoding_layer.weight

      # Reset the decoding layers weights as encoding layers weights tranpose
      # These won't be registered as model parameters
      for el, dl in zip(self.encoding_layers, reversed(_decoding_layers)):
        dl.weight = el.weight.t()

      self.de_embedding_layer = self.en_embedding_layer
    else:
      self.de_embedding_layer = nn.Embedding(self.num_embeddings, self.embedding_size)

    self.decoding_layers = nn.Sequential(*_decoding_layers)

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

    indices = sparse_batch._indices()

    nonzero_inputs = indices[1]
    nonzero_inputs = torch.from_numpy(np.unique(nonzero_inputs.numpy()))
    batch_items_map = {}
    for ind, inp in enumerate(nonzero_inputs):
      batch_items_map[inp] = ind

    reduced_indices = torch.LongTensor(indices.size())
    reduced_indices[0, :] = indices[0, :]
    reduced_indices[1, :] = torch.LongTensor([batch_items_map[ind] for ind in indices[1, :]])

    reduced_sparse_batch = torch.sparse.FloatTensor(reduced_indices, sparse_batch._values(),
                                                    torch.Size([sparse_batch.size(0), len(nonzero_inputs)]))
    reduced_batch = reduced_sparse_batch.to_dense()

    return Variable(reduced_batch), nonzero_inputs.long()

  def __tie_weights(self):
    for el, dl in zip(self.encoding_layers, reversed(self.decoding_layers)):
      dl.weight = el.weight.t()

  def cuda(self, device=None):
    self.use_cuda = True
    return super().cuda()

  def forward(self, input: torch.sparse.FloatTensor, target=None,
              full_output=True):

    assert full_output or (not full_output and target is not None)

    if self.is_constrained:
      self.__tie_weights()

    reduced_input, in_active_embeddings = self.__generate_reduced_batch(input)

    dense_target = None
    out_active_embeddings = None
    if target is not None and full_output:
      dense_target = Variable(target.to_dense())
    elif target is not None:
      dense_target, out_active_embeddings = self.__generate_reduced_batch(target)

    if self.use_cuda:
      reduced_input = reduced_input.cuda()
      in_active_embeddings = in_active_embeddings.cuda()

    if self.use_cuda and dense_target is not None:
      dense_target = dense_target.cuda()

    if self.use_cuda and out_active_embeddings is not None:
      out_active_embeddings = out_active_embeddings.cuda()

    # Normalize the input
    z = F.normalize(reduced_input, p=2, dim=1)
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

    z = self.__de_linear_embedding_layer(out_active_embeddings, z)

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


class MSELoss(nn.Module):
  """
  Computes the weighted mean squared error loss.

  The weight for an observation x:

  .. math::
    w = 1 + confidence \\times x

  Args:
    confidence (float, optional): the weighting of positive observations.
    size_average (bool, optional): whether to average the loss over the number observations in the
      input tensors
  """

  def __init__(self, confidence=0, size_average=True):
    super(MSELoss, self).__init__()
    self.size_average = size_average
    self.confidence = confidence

  def forward(self, input, target):
    weights = 1 + self.confidence * (target > 0).float()
    loss = F.mse_loss(input, target, reduce=False)
    weighted_loss = weights * loss
    if self.size_average:
      return weighted_loss.mean()
    else:
      return weighted_loss.sum()


class MultinomialNLLLoss(nn.Module):
  """
  Computes the negative log-likelihood of the multinomial distribution.

  .. math::
    \ell(x, y) = L = - y \cdot \log(softmax(x))

  Args:
    size_average (bool, optional): whether to average the loss over the number observations in the
      input tensors
  """

  def __init__(self, size_average=True):
    super(MultinomialNLLLoss, self).__init__()
    self.size_average = size_average

  def forward(self, input, target):
    loss = - target * F.log_softmax(input, dim=1)
    loss = loss.sum()

    if self.size_average:
      loss = loss.mean()
    else:
      loss = loss.sum()

    return loss
