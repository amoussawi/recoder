import glog as log
import numpy as np
from torch.utils.data import DataLoader

from recoder.data import RecommendationDataset
from recoder.modules import AutoEncoderRecommender
from recoder.utils import MetricEvaluator
from recoder.recommender import InferenceRecommender, SimilarityRecommender
from recoder.embedding import EmbeddingsIndex

root_dir = './'
data_dir = root_dir + 'data/msd/'
model_dir = root_dir + 'models/msd/'

def filter(data, item_id_map):
  available_songs = list(item_id_map.keys())
  filtered_data = data[data.song.isin(available_songs)]
  filtered_data.loc[:,'listens'] = (filtered_data['listens'] > 0).astype(np.int8)
  return filtered_data

common_params = {
  'item_based': True,
  'inter_dtype': np.int16,
  'user_col': 'user',
  'item_col': 'song',
  'inter_col': 'listens',
}

method = 'inference'
model_file = model_dir + 'a_15_d_0_128x128x128x256_reco_ae_epoch_2.model'
index_file = model_dir + 'a_15_d_0_128x128x128x256_reco_ae_epoch_2.model.index'

num_recommendations = 500

item_id_map = {}

if method == 'inference':
  model = AutoEncoderRecommender(mode='model', params={
    'model': model_file,
  })
  item_id_map = model.item_id_map
  recommender = InferenceRecommender(model, num_recommendations)
elif method == 'similarity':
  embeddings_index = EmbeddingsIndex(index_file=index_file)
  embeddings_index.load()
  recommender = SimilarityRecommender(embeddings_index, num_recommendations, scale=50)
  item_id_map = embeddings_index.id_map

val_dataset = RecommendationDataset(data_file=data_dir + 'val/val.csv', **common_params,
                                    data_apply_fn=lambda data: filter(data, item_id_map))

val_src_dataset = RecommendationDataset(data_file=data_dir + 'val_src/val.csv',
                                        target_dataset=val_dataset, **common_params,
                                        data_apply_fn=lambda data: filter(data, item_id_map))

val_dataloader = DataLoader(val_src_dataset,batch_size=1, shuffle=True,
                            collate_fn=lambda x: x)

k = [40, 80, 120, 160, 200, 500]
metric_evaluator = MetricEvaluator(k, metrics=['ap','recall'])

num_preds = 0

total_num_preds = -1

inverse_id_map = dict([(v,k) for k,v in item_id_map.items()])

for i, m_batch in enumerate(val_dataloader):
  input, target = m_batch[0]

  recommendations = recommender.recommend(input)

  assert(len(recommendations)==len(set(recommendations)))

  relevant_songs = list(zip(*target))[0]

  metric_evaluator.evaluate(recommendations, relevant_songs)

  num_preds += 1

  if total_num_preds != -1 and num_preds % total_num_preds == 0:
    break

ap_k = metric_evaluator.ap_k
recall_k = metric_evaluator.recall_k
ndcg_k = metric_evaluator.ndcg_k

for _k in k:
  log.info('Mean AP@{}: {}'.format(_k, ap_k[_k]))

for _k in k:
  log.info('Mean Recall@{}: {}'.format(_k, recall_k[_k]))

for _k in k:
  log.info('Mean NDCG@{}: {}'.format(_k, ndcg_k[_k]))
