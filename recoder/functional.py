import torch

def activation(x, act):
  if act == 'none': return x
  func = getattr(torch.nn.functional, act)
  return func(x)
