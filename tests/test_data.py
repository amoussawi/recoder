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

  return data


@pytest.fixture
def target_dataframe():
  data = pd.DataFrame()
  data['user'] = np.random.randint(1, 50, 300)
  data['item'] = np.random.randint(1, 200, 300)
  data['inter'] = np.ones(300)

  return data


def test_RecommendationDataset(input_dataframe):
  dataset = RecommendationDataset(data=input_dataframe)

  assert len(dataset) == len(np.unique(input_dataframe['user']))

  test_index = np.random.randint(0, len(dataset))

  input_interactions, target_interactions = dataset[test_index]

  assert len(input_interactions) > 0 and len(target_interactions) > 0

  assert input_interactions == target_interactions


def test_RecommendationDataset_target(input_dataframe, target_dataframe):
  target_dataset = RecommendationDataset(data=target_dataframe)
  dataset = RecommendationDataset(data=input_dataframe,
                                  target_dataset=target_dataset)

  assert len(dataset) == len(np.intersect1d(np.unique(input_dataframe['user']),
                                            np.unique(target_dataframe['user'])))

  test_index = np.random.randint(0, len(dataset))

  input_interactions, target_interactions = dataset[test_index]

  assert len(input_interactions) > 0 and len(target_interactions) > 0

  assert input_interactions != target_interactions




