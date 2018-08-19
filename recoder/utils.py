import numpy as np


def unzip(l):
  a, b = list(zip(*l))
  return list(a), list(b)


def normalize(x, axis=None):
  """
  Returns the normalization of `x` along `axis`.

  Args:
    x (np.array): matrix or vector
    axis (int, optional): the axis along which to compute the normalization
  """
  return x / np.linalg.norm(x, axis=axis).reshape(-1, 1)
