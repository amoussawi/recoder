import numpy as np


def unzip(l):
  """
  Returns the inverse operation of `zip` on `list`.

  Args:
    l (list): the list to unzip
  """
  return list(map(list, zip(*l)))


def normalize(x, axis=None):
  """
  Returns the normalization of `x` along `axis`.

  Args:
    x (np.array): matrix or vector
    axis (int, optional): the axis along which to compute the normalization
  """
  return x / np.linalg.norm(x, axis=axis).reshape(-1, 1)
