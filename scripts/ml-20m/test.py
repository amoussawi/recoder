import glog as log
import numpy as np

import pandas as pd

from recoder.data import RecommendationDataset
from recoder.modules import Recoder
from recoder.recommender import InferenceRecommender, SimilarityRecommender
from recoder.embedding import AnnoyEmbeddingsIndex, MemCacheEmbeddingsIndex
from recoder.metrics import AveragePrecision, Recall, NDCG, RecommenderEvaluator

root_dir = './'
data_dir = root_dir + 'data/ml-20m/pro_sg/'
model_dir = root_dir + 'models/ml-20m/'

common_params = {
  'user_col': 'uid',
  'item_col': 'sid',
  'inter_col': 'watched',
}

method = 'similarity'
model_file = model_dir + 'bce_no_ns_d_0.0_n_0.5_200_epoch_100.model'
index_file = model_dir + 'bce_ns_d_0.0_n_0.5_200_epoch_100.model.index'

num_recommendations = 100

if method == 'inference':
  model = Recoder(mode='model', model_file=model_file)
  recommender = InferenceRecommender(model, num_recommendations)
elif method == 'similarity':
  embeddings_index = AnnoyEmbeddingsIndex(index_file=index_file)
  embeddings_index.load()
  cache_embeddings_index = MemCacheEmbeddingsIndex(embeddings_index)
  recommender = SimilarityRecommender(cache_embeddings_index, num_recommendations, scale=1, n=50)

val_te_df = pd.read_csv(data_dir + 'test_te.csv')
val_tr_df = pd.read_csv(data_dir + 'test_tr.csv')
val_te_dataset = RecommendationDataset(data=val_te_df, **common_params)
val_tr_dataset = RecommendationDataset(data=val_tr_df, **common_params,
                                       target_dataset=val_te_dataset)

metrics = [AveragePrecision(k=100),Recall(k=20), Recall(k=50), Recall(k=100), NDCG(k=100)]
evaluator = RecommenderEvaluator(recommender, metrics)

metrics_accumulated = evaluator.evaluate(val_tr_dataset, batch_size=500)

for metric in metrics_accumulated:
  log.info('{}: {}'.format(metric, np.mean(metrics_accumulated[metric])))