import numpy as np
import pandas as pd

from recoder.data import RecommendationDataset
from recoder.modules import AutoEncoderRecommender, MSELoss

root_dir = './'
data_dir = root_dir + 'data/msd/'
model_dir = root_dir + 'models/msd/'

def log_scale_weights(targets, alpha, epsilon):
  weights = 1 + alpha * ((1 + targets / epsilon).log())
  return weights

def uniform_weighting(targets, alpha):
  binary_targets = (targets > 0).float()
  weights = 1 + alpha * binary_targets
  return weights

inactive_songs = []
inactive_users = []

def inactivity_filter_train(data: pd.DataFrame, min_user_count, min_song_count):
  global inactive_songs
  global inactive_users

  song_user_counts = data.groupby(by=['song'],as_index=False).count()
  inactive_songs = song_user_counts[song_user_counts.user < min_song_count].song

  user_song_counts = data.groupby(by=['user'],as_index=False).count()
  inactive_users = user_song_counts[user_song_counts.song < min_user_count].user

  filtered_data = data[(data.user.isin(inactive_users)==False) & (data.song.isin(inactive_songs)==False)]
  return filtered_data

def inactive_filter_eval(data):
  filtered_data = data[(data.user.isin(inactive_users)==False) & (data.song.isin(inactive_songs)==False)]
  return filtered_data


common_params = {
  'item_based': True,
  'inter_dtype': np.int16,
  'user_col': 'user',
  'item_col': 'song',
  'inter_col': 'listens',
}

data_apply_fn = lambda data: inactivity_filter_train(data, min_user_count=20, min_song_count=50)
train_dataset = RecommendationDataset(data_file=data_dir + 'full_train/full_train.csv', **common_params,
                                      data_apply_fn=data_apply_fn)
eval_target_dataset = RecommendationDataset(data_file=data_dir + 'val/val.csv', **common_params,
                                            data_apply_fn=inactive_filter_eval)
eval_dataset = RecommendationDataset(data_file=data_dir + 'val_src/val.csv',
                                    target_dataset=eval_target_dataset, **common_params,
                                    data_apply_fn=inactive_filter_eval)

del inactive_users
del inactive_songs

params = {
  'hidden_layers_sizes': [128,128,128,256],
  'activation_type': 'selu',
  'is_constrained': True,
  'item_based': True,
  'batch_size': 64,
  'dropout_prob': [0.0,0.0,0.0,0.5],
  'optimizer_type': 'sgd',
  'lr': 0.1,
  'weight_decay': 1e-4,
  'num_epochs': 3,
  'optimizer_milestones': [2,3],
  'summary_frequency': 100,
  'loss_func': MSELoss(size_average=False),
  'last_layer_act': 'none',
  'eval_epoch_freq': 1,
  'eval_itr_freq': 1000,
  'train_dataset': train_dataset,
  'eval_dataset': eval_dataset,
  'model_checkpoint': model_dir + 'a_15_d_0_128x128x128x256_',
  # 'noise': BernoulliNoise(pb=0.2, noise=0),
  # 'model': model_dir + 'a_15_d_0_reco_ae_epoch_0.model',
  'compute_weights': lambda x: uniform_weighting(x, alpha=15),
  'transform': lambda x: (x > 0).float(),
}

mode = 'train'
trainer = AutoEncoderRecommender(mode=mode, params=params)

try:
  trainer.run()
except (KeyboardInterrupt, SystemExit):
  trainer.save_state()
  raise