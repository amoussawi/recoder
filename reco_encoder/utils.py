
def average_precision(x, y, k):
  tp = 0
  y = set(y) if type(y) is not set else y
  average_precision = 0
  average_precision_k = {}

  for _k, _x in enumerate(x):
    if _x in y:
      tp += 1
      delta_recall = 1.0
    else:
      delta_recall = 0

    prec_k = tp / (_k + 1)

    average_precision += prec_k * delta_recall

    if (_k + 1) in k:
      average_precision_k[_k + 1] = average_precision / len(y)

  return average_precision_k

def recall(x, y, k):
  tp = 0

  y = set(y) if type(y) is not set else y
  recall_k = {}

  for _k, _x in enumerate(x):
    if _x in y:
      tp += 1

    if (_k + 1) in k:
      recall_k[_k + 1] = tp / len(y)

  return recall_k
