import numpy as np
import pandas as pd
import pytest

from recoder.data import RecommendationDataset
from recoder.metrics import Recall, NDCG
from recoder.model import Recoder
from recoder.nn import DynamicAutoencoder, MatrixFactorization

import os


@pytest.mark.parametrize("exp_recall_20,exp_recall_50,exp_ndcg_100",[
  (0.40, 0.43, 0.45),
])
def test_model(exp_recall_20, exp_recall_50, exp_ndcg_100):
  data_dir = 'tests/data/'
  model_dir = '/tmp/'

  train_df = pd.read_csv(data_dir + 'train.csv')
  val_df = pd.read_csv(data_dir + 'val.csv')

  # keep the items that exist in the training dataset
  val_df = val_df[val_df.sid.isin(train_df.sid.unique())]

  train_dataset = RecommendationDataset()
  val_dataset = RecommendationDataset(target_dataset=train_dataset)

  train_dataset.fill_from_dataframe(dataframe=train_df, num_workers=4, user_col='uid',
                                    item_col='sid', inter_col='watched')

  val_dataset.fill_from_dataframe(dataframe=val_df, user_col='uid',
                                  item_col='sid', inter_col='watched')

  use_cuda = False
  model = DynamicAutoencoder(hidden_layers=[200], activation_type='tanh',
                             noise_prob=0.5, sparse=False)
  trainer = Recoder(model=model, use_cuda=use_cuda, optimizer_type='adam',
                    loss='logloss')

  trainer.train(train_dataset=train_dataset, val_dataset=val_dataset,
                batch_size=500, lr=1e-3, weight_decay=2e-5,
                num_epochs=30, num_neg_samples=0)

  # assert model metrics
  recall_20 = Recall(k=20, normalize=True)
  recall_50 = Recall(k=50, normalize=True)
  ndcg_100 = NDCG(k=100)

  results = trainer._evaluate(eval_dataset=val_dataset, num_recommendations=100,
                              metrics=[recall_20, recall_50, ndcg_100], batch_size=500)

  for metric, value in list(results.items()):
    results[metric] = np.mean(results[metric])

  assert np.isclose(results[recall_20], exp_recall_20, atol=0.01, rtol=0)
  assert np.isclose(results[recall_50], exp_recall_50, atol=0.01, rtol=0)
  assert np.isclose(results[ndcg_100], exp_ndcg_100, atol=0.01, rtol=0)

  # Save the model and evaluate again
  model_checkpoint = model_dir + 'test_model.model'
  state_file = trainer.save_state(model_checkpoint)

  model = DynamicAutoencoder(sparse=False)
  trainer = Recoder(model=model, use_cuda=use_cuda,
                    optimizer_type='adam', loss='logloss')

  trainer.init_from_model_file(state_file)

  results = trainer._evaluate(eval_dataset=val_dataset, num_recommendations=100,
                              metrics=[recall_20, recall_50, ndcg_100], batch_size=500)

  for metric, value in list(results.items()):
    results[metric] = np.mean(results[metric])

  assert np.isclose(results[recall_20], exp_recall_20, atol=0.01, rtol=0)
  assert np.isclose(results[recall_50], exp_recall_50, atol=0.01, rtol=0)
  assert np.isclose(results[ndcg_100], exp_ndcg_100, atol=0.01, rtol=0)

  os.remove(state_file)
