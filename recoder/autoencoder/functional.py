import torch
import torch.nn.functional as F
from torch.autograd import Variable

def activation(input, kind):
  if kind == 'selu':
    return F.selu(input)
  elif kind == 'relu':
    return F.relu(input)
  elif kind == 'relu6':
    return F.relu6(input)
  elif kind == 'sigmoid':
    return F.sigmoid(input)
  elif kind == 'tanh':
    return F.tanh(input)
  elif kind == 'elu':
    return F.elu(input)
  elif kind == 'lrelu':
    return F.leaky_relu(input)
  elif kind == 'none':
    return input
  else:
    raise ValueError('Unknown non-linearity type')


def encode(x, encode_w, encode_b, activation_type, dp_drop_prob=None, training=False):
  _dp_drop_prob = dp_drop_prob
  if _dp_drop_prob is None:
    _dp_drop_prob = [0.0] * len(encode_w)

  if len(_dp_drop_prob) != len(encode_w):
    raise ValueError('dropout values list and list of weights should have equal length')

  if len(activation_type) != len(encode_w):
    raise ValueError('layers activation types list and list of weights should have equal length')

  for ind, w in enumerate(encode_w):
    x = activation(input=F.linear(input=x, weight=w,bias=encode_b[ind]),kind=activation_type[ind])
    if _dp_drop_prob[ind] > 0:
      x = F.dropout(x, p=_dp_drop_prob[ind], training=training, inplace=False)
  return x


def decode(z, decode_w, decode_b, activation_type, dp_drop_prob=None, training=False):
  return encode(z, decode_w, decode_b, activation_type, dp_drop_prob=dp_drop_prob, training=training)


def bernoulli_noise(input, pb, noise=0):
  _noise = torch.bernoulli(torch.ones(input.size()) * (1 - pb))
  _noise[_noise == 0] = noise
  _noise = Variable(_noise)
  corrupted_input = input * _noise
  return corrupted_input
