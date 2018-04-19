import numpy as np

from torch.utils.data import DataLoader

import recoder.utils as utils


def average_precision(x, y, k, normalize=True):
  x = x[:k]
  x_in_y = np.isin(x, y, assume_unique=True).astype(np.int)

  tp = x_in_y.cumsum()  # true positives at every position in x_in_y
  precision = tp / (1 + np.arange(len(x)))  # precision at every position
  precision_drecall = np.multiply(precision, x_in_y)  # precision * delta_recall at every position

  normalization = min(k, len(y)) if normalize else len(y)
  ap = precision_drecall.sum() / normalization

  return ap


def recall(x, y, k, normalize=True):
  x = x[:k]
  x_in_y = np.isin(x, y, assume_unique=True).astype(np.int)
  normalization = min(k, len(y)) if normalize else len(y)
  _recall = x_in_y.sum() / normalization

  return _recall


def dcg(x, y, k):
  x = x[:k]
  x_in_y = np.isin(x, y, assume_unique=True).astype(np.int)
  cg = x_in_y / np.log2(2 + np.arange(len(x)))  # cumulative gain at every position
  _dcg = cg.sum()

  return _dcg


def ndcg(x, y, k):
  dcg_k = dcg(x, y, k)
  idcg_k = dcg(y, y, k)
  ndcg_k = dcg_k / idcg_k
  return ndcg_k


class Metric(object):

  def __init__(self, metric_name):
    self.metric_name = metric_name

  def __str__(self):
    return self.metric_name

  def __hash__(self):
    return self.metric_name.__hash__()

  def evaluate(self, x, y):
    raise NotImplementedError


class AveragePrecision(Metric):

  def __init__(self, k, normalize=True):
    super().__init__(metric_name='AveragePrecision@{}'.format(k))
    self.k = k
    self.normalize = normalize

  def evaluate(self, x, y):
    return average_precision(x, y, k=self.k, normalize=self.normalize)


class Recall(Metric):

  def __init__(self, k, normalize=True):
    super().__init__(metric_name='Recall@{}'.format(k))
    self.k = k
    self.normalize = normalize

  def evaluate(self, x, y):
    return recall(x, y, k=self.k, normalize=self.normalize)


class NDCG(Metric):

  def __init__(self, k):
    super().__init__(metric_name='NDCG@{}'.format(k))
    self.k = k

  def evaluate(self, x, y):
    return ndcg(x, y, k=self.k)


class RecommenderEvaluator(object):

  def __init__(self, recommender, metrics):
    self.recommender = recommender
    self.metrics = metrics

  def evaluate(self, eval_dataset, batch_size=1):
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True,
                                collate_fn=lambda _: _)

    results = {}
    for metric in self.metrics:
      results[metric] = []

    for batch in dataloader:
      input, target = utils.unzip(batch)

      recommendations = self.recommender.recommend(input)

      relevant_songs = [utils.unzip(l)[0] for l in target]

      for x, y in zip(recommendations, relevant_songs):
        for metric in self.metrics:
          results[metric].append(metric.evaluate(x, y))

    return results
