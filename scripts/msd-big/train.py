import pandas as pd
import glog

from recoder.model import Recoder
from recoder.data import RecommendationDataset
from recoder.metrics import AveragePrecision, Recall, NDCG

import multiprocessing as mp


data_dir = 'data/msd-big/'
model_dir = 'models/msd-big/'

common_params = {
  'user_col': 'uid',
  'item_col': 'sid',
  'inter_col': 'listen',
}

glog.info('Loading Data...')
train_df = pd.read_csv(data_dir + 'train.csv')
val_tr_df = pd.read_csv(data_dir + 'validation_tr.csv')
val_te_df = pd.read_csv(data_dir + 'validation_te.csv')

train_dataset = RecommendationDataset()
val_te_dataset = RecommendationDataset()
val_tr_dataset = RecommendationDataset(target_dataset=val_te_dataset)

train_dataset.fill_from_dataframe(dataframe=train_df, **common_params)
val_te_dataset.fill_from_dataframe(dataframe=val_te_df, **common_params)
val_tr_dataset.fill_from_dataframe(dataframe=val_tr_df, **common_params)

use_cuda = True

model_params = {
  'activation_type': 'tanh',
  'noise_prob': 0.5,
}

trainer = Recoder(hidden_layers=[200], model_params=model_params,
                  use_cuda=use_cuda, optimizer_type='adam',
                  loss='logistic', index_item_ids=False)

# trainer.init_from_model_file(model_dir + 'bce_ns_d_0.0_n_0.5_200_epoch_50.model')
model_checkpoint = model_dir + 'bce_ns_d_0.0_n_0.5_200'

metrics = [Recall(k=20, normalize=True), Recall(k=50, normalize=True),
           NDCG(k=100)]

try:
  trainer.train(train_dataset=train_dataset, val_dataset=val_tr_dataset,
                batch_size=500, lr=1e-3, weight_decay=2e-5,
                num_epochs=100, num_neg_samples=0,
                lr_milestones=[60, 80],
                num_data_workers=mp.cpu_count() if use_cuda else 0,
                model_checkpoint_prefix=model_checkpoint,
                checkpoint_freq=10, eval_num_recommendations=100,
                metrics=metrics, eval_freq=10)
except (KeyboardInterrupt, SystemExit):
  trainer.save_state(model_checkpoint)
  raise
