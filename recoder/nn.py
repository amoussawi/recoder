import torch
from torch import nn
import torch.nn.functional as F


def activation(x, act):
  if act == 'none': return x
  func = getattr(torch, act)
  return func(x)


class DynamicAutoencoder(nn.Module):
  """
  An Autoencoder module that processes variable size vectors. This is
  particularly efficient for cases where we only want to reconstruct sub-samples
  of a large sparse vector and not the whole vector, i.e negative sampling for
  collaborative filtering and NLP.

  Let `F` be a `DynamicAutoencoder` function that reconstructs vectors of size `d`,
  let `X` be a matrix of size `Bxd` where `B` is the batch size, and
  let `Z` be a sub-matrix of `X` and `I` be a vector of any length, such that `1 <= I[i] <= d`
  and `Z = X[:, I]`. The reconstruction of `Z` is `F(Z, I)`. See `Examples`.

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

    >>>> autoencoder = DynamicAutoencoder([500,100])
    >>>> batch_size = 32
    >>>> input = torch.rand(batch_size, 5)
    >>>> input_words = torch.LongTensor([10, 126, 452, 29, 34])
    >>>> output = autoencoder(input, input_words=input_words)
    >>>> output
       0.0850  0.9490  ...   0.2430  0.5323
       0.3519  0.4816  ...   0.9483  0.2497
            ...         ⋱         ...
       0.8744  0.8194  ...   0.5755  0.2090
       0.5006  0.9532  ...   0.8333  0.4330
      [torch.FloatTensor of size 32x5]
    >>>>
    >>>> # predicting a different target of words
    >>>> target_words = torch.LongTensor([31, 14, 95, 49, 10, 36, 239])
    >>>> output = autoencoder(input, input_words=input_words, target_words=target_words)
    >>>> output
       0.5446  0.5468  ...   0.9854  0.6465
       0.0564  0.1238  ...   0.5645  0.6576
            ...         ⋱         ...
       0.0498  0.6978  ...   0.8462  0.2135
       0.6540  0.5686  ...   0.6540  0.4330
      [torch.FloatTensor of size 32x7]
    >>>>
    >>>> # reconstructing the whole vector
    >>>> input = torch.rand(batch_size, 500)
    >>>> output = autoencoder(input)
    >>>> output
       0.0865  0.9054  ...   0.8987  0.0456
       0.9852  0.6540  ...   0.1205  0.8488
            ...         ⋱         ...
       0.4650  0.3540  ...   0.5646  0.5605
       0.6940  0.2140  ...   0.9820  0.5405
      [torch.FloatTensor of size 32x500]
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

    nn.init.xavier_uniform_(self.en_embedding_layer.weight)
    nn.init.constant_(self.__en_linear_embedding_layer.bias, 0)

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

    nn.init.xavier_uniform_(self.de_embedding_layer.weight)
    nn.init.constant_(self.__de_linear_embedding_layer.bias, 0)

  def __create_coding_layers(self, layer_sizes):
    layers = []
    for ind, layer_size in enumerate(layer_sizes[1:], 1):
      layer = nn.Linear(layer_sizes[ind-1], layer_size)
      layers.append(layer)
      torch.nn.init.xavier_uniform_(layer.weight)
      torch.nn.init.constant_(layer.bias, 0)

    return layers

  def __tie_weights(self):
    for el, dl in zip(self.encoding_layers, reversed(self.decoding_layers)):
      dl.weight = el.weight.t()

  def forward(self, input, input_words=None,
              target_words=None, full_output=False):

    if target_words is None and not full_output:
      target_words = input_words

    if full_output:
      target_words = None

    if self.is_constrained:
      self.__tie_weights()

    # Normalize the input
    z = F.normalize(input, p=2, dim=1)
    if self.noise_prob > 0.0:
      z = self.noise_layer(z)

    z = self.__en_linear_embedding_layer(input_words, z)
    z = activation(z, self.activation_type)

    for encoding_layer in self.encoding_layers:
      z = activation(encoding_layer(z), self.activation_type)

    if self.dropout_prob > 0.0:
      z = self.dropout_layer(z)

    for decoding_layer in self.decoding_layers:
      z = activation(decoding_layer(z), self.activation_type)

    z = self.__de_linear_embedding_layer(target_words, z)

    z = activation(z, self.last_layer_act)

    return z


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
      _weight = self.embedding_layer(x)
      _bias = self.bias if self.input_based else self.bias.index_select(0, x)
    else:
      _weight = self.embedding_layer.weight
      _bias = self.bias

    if self.input_based:
      return F.linear(y, _weight.t(), _bias)
    else:
      return F.linear(y, _weight, _bias)
