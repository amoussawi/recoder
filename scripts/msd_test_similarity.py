from recoder.embedding import EmbeddingsIndex
from recoder.data import RecommendationDataset
import numpy as np
from torch.utils.data import DataLoader
import heapq
from recoder.utils import average_precision, recall
import glog as log

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

embeddings_index = EmbeddingsIndex(index_file=model_dir+'a_15_d_0_128x128x128x256_reco_ae_epoch_2.model.index')
embeddings_index.load()

def filter(data):
  available_songs = list(embeddings_index.id_map.keys())
  filtered_data = data[data.song.isin(available_songs)]
  return filtered_data

val_dataset = RecommendationDataset(data_file=data_dir + 'val/val.csv', **common_params,
                                    data_apply_fn=filter)
val_src_dataset = RecommendationDataset(data_file=data_dir + 'val_src/val.csv',
                                        target_dataset=val_dataset, **common_params,
                                        data_apply_fn=filter)

val_dataloader = DataLoader(val_src_dataset,batch_size=1,shuffle=True,
                            collate_fn=lambda mb: mb)

num_recommendations = 500
k = [40, 80, 120, 160, 200, 500]
mean_ap_k = dict([(_k,0) for _k in k])
mean_recall_k = dict([(_k,0) for _k in k])

num_preds = 0
total_num_preds = -1

for i, m_batch in enumerate(val_dataloader):
  input, target = m_batch[0]

  listened_songs = list(zip(*input))[0]

  num_s = 100
  nn_songs = []

  for song_id in listened_songs:
    if song_id in embeddings_index.id_map:
      nn_songs += embeddings_index.get_nns_by_id(song_id, num_s, search_k=10000)

  nn_songs = set(nn_songs)

  recommendations_heap = []
  for song_id in nn_songs:
    if song_id in listened_songs: continue
    song_score = 0
    for l_song_id in listened_songs:
      if l_song_id in embeddings_index.id_map:
        song_score += embeddings_index.get_similarity(song_id, l_song_id)**50

    if len(recommendations_heap) < num_recommendations:
      heapq.heappush(recommendations_heap, (song_score, song_id))
    elif recommendations_heap[0][0] < song_score:
      heapq.heapreplace(recommendations_heap, (song_score, song_id))

  recommendations = []
  while recommendations_heap:
    recommendations.append(heapq.heappop(recommendations_heap)[1])
  recommendations.reverse()

  assert(len(recommendations)==len(set(recommendations)))

  relevant_songs = list(zip(*target))[0]

  ap_k = average_precision(recommendations, relevant_songs, k)
  recall_k = recall(recommendations, relevant_songs, k)

  prev_recall = 0
  prev_ap = 0
  for _k in k:
    mean_ap_k[_k] += ap_k[_k] if _k in ap_k else prev_ap
    mean_recall_k[_k] += recall_k[_k] if _k in ap_k else prev_recall
    prev_ap = ap_k[_k] if _k in ap_k else prev_ap
    prev_recall = recall_k[_k] if _k in ap_k else prev_recall

  num_preds += 1

  if total_num_preds != -1 and num_preds % total_num_preds == 0:
    break

for _k in k:
  mean_ap_k[_k] /= num_preds
  mean_recall_k[_k] /= num_preds

for _k in k:  
  log.info('Mean AP@{}: {}'.format(_k, mean_ap_k[_k]))
  log.info('Mean Recall@{}: {}'.format(_k, mean_recall_k[_k]))
