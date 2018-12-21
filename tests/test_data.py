from recoder.data import RecommendationDataset, RecommendationDataLoader, BatchCollator

import pandas as pd
import numpy as np
import torch

import pytest

def generate_dataframe():
  data = pd.DataFrame()
  data['user'] = np.random.randint(0, 100, 1000)
  data['item'] = np.random.randint(0, 200, 1000)
  data['inter'] = np.ones(1000)
  data = data.drop_duplicates(['user', 'item']).reset_index(drop=True)
  return data

@pytest.fixture
def input_dataframe():
  return generate_dataframe()


@pytest.fixture
def target_dataframe():
  return generate_dataframe()


@pytest.mark.parametrize("num_workers",
                         [0, 4])
def test_RecommendationDataset(input_dataframe, num_workers):
  dataset = RecommendationDataset()
  dataset.fill_from_dataframe(input_dataframe, num_workers=num_workers)

  assert len(dataset) == len(np.unique(input_dataframe['user']))

  replica_df = pd.DataFrame(input_dataframe)

  for index in range(len(dataset)):
    user_interactions, _ = dataset[index]
    user = user_interactions.user
    assert len(user_interactions.items) == len(replica_df[replica_df.user == user])

    for item_id, inter_val in zip(user_interactions.items, user_interactions.values):
      assert len(replica_df[(replica_df.user == user)
                            & (replica_df.item == item_id)
                            & (replica_df.inter == inter_val)]) > 0
      replica_df = replica_df[~ ((replica_df.user == user)
                                & (replica_df.item == item_id)
                                & (replica_df.inter == inter_val))]

    assert len(user_interactions.items) > 0

  # check that both the returned list of interactions and the dataframe contain
  # the same of interactions
  assert len(replica_df) == 0


def test_RecommendationDataset_target(input_dataframe, target_dataframe):
  target_dataset = RecommendationDataset()
  dataset = RecommendationDataset(target_dataset=target_dataset)

  dataset.fill_from_dataframe(input_dataframe)
  target_dataset.fill_from_dataframe(target_dataframe)

  test_index = np.random.randint(0, len(dataset))

  input_interactions, target_interactions = dataset[test_index]

  assert len(input_interactions.items) > 0 and len(target_interactions.items) > 0

  assert input_interactions.items != target_interactions.items


@pytest.mark.parametrize("batch_size,num_sampling_users",
                         [(5, 0),
                          (5, 10)])
def test_RecommendationDataLoader(input_dataframe, target_dataframe,
                                  batch_size, num_sampling_users):
  common_users = input_dataframe.merge(target_dataframe, how='inner', on='user').user.unique()
  input_dataframe = input_dataframe[input_dataframe.user.isin(common_users)]
  target_dataframe = target_dataframe[target_dataframe.user.isin(common_users)]

  target_dataset = RecommendationDataset()
  dataset = RecommendationDataset(target_dataset=target_dataset)

  dataset.fill_from_dataframe(input_dataframe)
  target_dataset.fill_from_dataframe(target_dataframe)

  dataloader = RecommendationDataLoader(dataset, batch_size=batch_size,
                                        vector_dim=len(dataset.items),
                                        num_neg_samples=0,
                                        num_sampling_users=num_sampling_users)

  for batch_idx, (input, target) in enumerate(dataloader, 1):
    input_idx, input_val, input_size, input_items = input.indices, input.values, input.size, input.items
    input_dense = torch.sparse.FloatTensor(input_idx, input_val, input_size).to_dense()

    target_idx, target_val, target_size, target_words = target.indices, target.values, target.size, target.items
    target_dense = torch.sparse.FloatTensor(target_idx, target_val, target_size).to_dense()

    assert target is not None

    assert input_dense.size(0) == batch_size \
           or batch_idx == len(dataloader) and input_dense.size(0) == len(dataset) % batch_size
    assert input_dense.size(1) == len(input_items)

@pytest.mark.parametrize("batch_size",
                         [1, 2, 5, 10, 13])
def test_BatchCollator(input_dataframe, batch_size):
  dataset = RecommendationDataset()
  dataset.fill_from_dataframe(input_dataframe)

  batch_collator = BatchCollator(batch_size=batch_size, vector_dim=200,
                                 num_neg_samples=0)

  big_batch = [sample for sample, _ in dataset]
  batches = batch_collator.collate(big_batch)

  assert len(batches) == np.ceil(len(dataset) / batch_size)

  current_batch = 0
  for batch in batches:
    input_idx, input_val, input_size, input_words = batch.indices, batch.values, batch.size, batch.items
    input_dense = torch.sparse.FloatTensor(input_idx, input_val, input_size).to_dense()

    num_values_per_user = [len(inters.items) for inters in big_batch[current_batch:current_batch+batch_size]]

    assert (input_dense > 0).float().sum(dim=1).tolist() == num_values_per_user

    item_idx_map = {item_id:item_idx for item_idx, item_id in enumerate(input_words.tolist())}

    for user_idx, inters in enumerate(big_batch[current_batch:current_batch+batch_size]):
      for item_id, val in zip(inters.items, inters.values):
        assert item_id in input_words
        assert input_dense[user_idx, item_idx_map[item_id]] == val

    current_batch += batch_size
