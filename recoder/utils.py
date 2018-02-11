import numpy as np

def average_precision(x, y, k):
  tp = 0

  k = [len(x)] + k
  k = sorted(k)

  y = set(y) if type(y) is not set else y
  average_precision = 0
  average_precision_k = {}

  for _k, _x in enumerate(x, 1):
    if _x in y:
      tp += 1
      delta_recall = 1.0
    else:
      delta_recall = 0

    prec_k = tp / (_k)

    average_precision += prec_k * delta_recall

    if _k in k:
      average_precision_k[_k] = average_precision / len(y)

  for ki, _k in enumerate(k):
    if _k not in average_precision_k:
      average_precision_k[_k] = average_precision_k[k[ki-1]]

  return average_precision_k

def recall(x, y, k):
  tp = 0

  k = [len(x)] + k
  k = sorted(k)

  y = set(y) if type(y) is not set else y
  recall_k = {}

  for _k, _x in enumerate(x, 1):
    if _x in y:
      tp += 1

    if _k in k:
      recall_k[_k] = tp / len(y)

  for ki, _k in enumerate(k):
    if _k not in recall_k:
      recall_k[_k] = recall_k[k[ki-1]]

  return recall_k

def dcg(x, y, k):
  _dcg = 0.0

  k = [len(x)] + k
  k = sorted(k)

  y = set(y) if type(y) is not set else y
  dcg_k = {}

  for _k, _x in enumerate(x, 1):
    if _x in y:
      _dcg += 1 / np.log2(_k + 1)

    if _k in k:
      dcg_k[_k] = _dcg

  for ki, _k in enumerate(k):
    if _k not in dcg_k:
      dcg_k[_k] = dcg_k[k[ki-1]]

  return dcg_k

def ndcg(x, y, k):
  dcg_k = dcg(x, y, k)
  idcg_k = dcg(y, y, k)

  ndcg_k = {}
  for _k in k:
    ndcg_k[_k] = dcg_k[_k] / idcg_k[_k]

  return ndcg_k

class MetricEvaluator(object):

  def __init__(self, k, metrics=None):
    self.k = k
    self.metrics = metrics if metrics is not None else ['ap','recall','ndcg']

    self._s_ap_k = dict([(_k, 0) for _k in k])
    self._s_recall_k = dict([(_k, 0) for _k in k])
    self._s_ndcg_k = dict([(_k, 0) for _k in k])

    self.ap_k = dict([(_k, 0) for _k in k])
    self.recall_k = dict([(_k, 0) for _k in k])
    self.ndcg_k = dict([(_k, 0) for _k in k])

    self.num_rec = 0

  def evaluate(self, x, y):
    self.num_rec += 1

    ap_k = recall_k = ndcg_k = {}

    if 'ap' in self.metrics:
      ap_k = average_precision(x, y, self.k)
      for _k in self.k:
        self._s_ap_k[_k] += ap_k[_k]
        self.ap_k[_k] = self._s_ap_k[_k] / self.num_rec

    if 'recall' in self.metrics:
      recall_k = recall(x, y, self.k)
      for _k in self.k:
        self._s_recall_k[_k] += recall_k[_k]
        self.recall_k[_k] = self._s_recall_k[_k] / self.num_rec

    if 'ndcg' in self.metrics:
      ndcg_k = ndcg(x, y, self.k)
      for _k in self.k:
        self._s_ndcg_k[_k] += ndcg_k[_k]
        self.ndcg_k[_k] = self._s_ndcg_k[_k] / self.num_rec

    return {'ap': ap_k, 'recall': recall_k, 'ndcg': ndcg_k}