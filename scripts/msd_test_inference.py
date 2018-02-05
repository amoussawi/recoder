import heapq

import glog as log
import numpy as np
from torch.utils.data import DataLoader

from recoder.autoencoder.modules import SparseBatchAutoEncoder
from recoder.data import RecommendationDataset
from recoder.modules import AutoEncoderRecommender
from recoder.utils import average_precision, recall

root_dir = './'
data_dir = root_dir + 'data/msd/'
model_dir = root_dir + 'models/msd/'

common_params = {
  'item_based': True,
  'inter_dtype': np.int16,
  'user_col': 'user',
  'item_col': 'song',
  'inter_col': 'listens',
}

transform = lambda x: (x > 0).float()

model = AutoEncoderRecommender(mode='model',params={
  'model': model_dir + 'a_15_d_0_128x128x128x256_reco_ae_epoch_2.model',
})

def filter(data):
  available_songs = list(model.item_id_map.keys())
  filtered_data = data[data.song.isin(available_songs)]
  return filtered_data

val_dataset = RecommendationDataset(data_file=data_dir + 'val/val.csv', **common_params,
                                    data_apply_fn=filter)
val_src_dataset = RecommendationDataset(data_file=data_dir + 'val_src/val.csv',
                                        target_dataset=val_dataset, **common_params,
                                        data_apply_fn=filter)

val_dataloader = DataLoader(val_src_dataset,batch_size=1, shuffle=True,
                            collate_fn=model.collate_to_sparse_batch)

num_recommendations = 500
k = [40, 80, 120, 160, 200, 500]
mean_ap_k = dict([(_k,0) for _k in k])
mean_recall_k = dict([(_k,0) for _k in k])

num_preds = 0

total_num_preds = -1
model.autoencoder.eval()
inverse_id_map = dict([(v,k) for k,v in model.item_id_map.items()])

for i, (input, target) in enumerate(val_dataloader):

  sparse_encoder = SparseBatchAutoEncoder(model.autoencoder, sparse_batch_in=input,
                                          sparse_batch_out=target, full_output=True)
  reduced_input = sparse_encoder.reduced_batch_in
  reduced_target = sparse_encoder.reduced_batch_out

  transformed_input = transform(reduced_input)
  transformed_target = transform(reduced_target)

  output = sparse_encoder(transformed_input)

  target_np = transformed_target.data.numpy().reshape(-1)
  output_np = output.data.numpy().reshape(-1)

  top_k = heapq.nlargest(num_recommendations*2, enumerate(output_np), key=lambda k: k[1])

  preds = []
  recommendations = []
  for song_id, pred in top_k:
    if len(recommendations) == num_recommendations:
      break
    if not song_id in sparse_encoder.active_inputs.data:
      recommendations.append(song_id)
      preds.append(pred)

  relevant_songs = []
  for ind, target in enumerate(target_np):
    song_id = sparse_encoder.active_outputs_inverse_map[ind]
    if not song_id in sparse_encoder.active_inputs.data:
      relevant_songs.append(song_id)

  ap_k = average_precision(recommendations, relevant_songs, k)
  recall_k = recall(recommendations, relevant_songs, k)

  for _k in k:
    mean_ap_k[_k] += ap_k[_k]
    mean_recall_k[_k] += recall_k[_k]

  num_preds += 1

  if total_num_preds != -1 and num_preds % total_num_preds == 0:
    break

for _k in k:
  mean_ap_k[_k] /= num_preds
  mean_recall_k[_k] /= num_preds

for _k in k:  
  log.info('Mean AP@{}: {}'.format(_k, mean_ap_k[_k]))
  log.info('Mean Recall@{}: {}'.format(_k, mean_recall_k[_k]))
