
def average_precision(preds, documents):
  num_correct_preds = 0
  average_precision = 0
  for k, pred in enumerate(preds):
    if pred in documents:
      num_correct_preds += 1
      prec_k = num_correct_preds / (k + 1)
      average_precision += prec_k

  average_precision /= len(documents)

  return average_precision

def recall(preds, documents, k):
  num_correct_preds = 0
  preds = preds[:k]
  for document in documents:
    if document in preds:
      num_correct_preds += 1

  recall = num_correct_preds / len(documents)

  return recall