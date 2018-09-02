from recoder.data import RecommendationDataset

import pandas as pd

import numpy as np

import pytest


@pytest.fixture
def input_dataframe():
  data = pd.DataFrame()
  data['user'] = np.random.randint(1, 50, 300)
  data['item'] = np.random.randint(1, 200, 300)
  data['inter'] = np.ones(300)
  data = data.drop_duplicates(['user', 'item'])
  return data


@pytest.fixture
def target_dataframe():
  data = pd.DataFrame()
  data['user'] = np.random.randint(1, 50, 300)
  data['item'] = np.random.randint(1, 200, 300)
  data['inter'] = np.ones(300)
  data = data.drop_duplicates(['user', 'item'])
  return data


@pytest.mark.parametrize("num_workers",
                         [0, 4])
def test_RecommendationDataset(input_dataframe, num_workers):
  dataset = RecommendationDataset()
  dataset.fill_from_dataframe(input_dataframe, num_workers=num_workers)

  assert len(dataset) == len(np.unique(input_dataframe['user']))

  replica_df = pd.DataFrame(input_dataframe)

  for index in range(len(dataset)):
    user = dataset.users[index]
    interactions = dataset[index]

    assert len(interactions) == len(replica_df[replica_df.user == user])

    for item_id, inter_val in interactions:
      assert len(replica_df[(replica_df.user == user)
                            & (replica_df.item == item_id)
                            & (replica_df.inter == inter_val)]) > 0
      replica_df = replica_df[~ ((replica_df.user == user)
                                & (replica_df.item == item_id)
                                & (replica_df.inter == inter_val))]

    assert not type(interactions) is tuple
    assert len(interactions) > 0

  # check that both the returned list of interactions and the dataframe contain
  # the same of interactions
  assert len(replica_df) == 0


def test_RecommendationDataset_target(input_dataframe, target_dataframe):
  target_dataset = RecommendationDataset()
  dataset = RecommendationDataset(target_dataset=target_dataset)

  dataset.fill_from_dataframe(target_dataframe)
  target_dataset.fill_from_dataframe(input_dataframe)

  test_index = np.random.randint(0, len(dataset))

  input_interactions, target_interactions = dataset[test_index]

  assert len(input_interactions) > 0 and len(target_interactions) > 0

  assert input_interactions != target_interactions
