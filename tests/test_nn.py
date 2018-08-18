from recoder.nn import DynamicAutoencoder

import torch

import pytest


@pytest.fixture
def autoencoder():
  autoencoder = DynamicAutoencoder([500, 300, 200])
  return autoencoder


def test_DynamicAutoencoder(autoencoder):
  assert autoencoder.en_embedding_layer.embedding_dim == 300
  assert autoencoder.de_embedding_layer.embedding_dim == 300

  assert len(autoencoder.encoding_layers) == 1
  assert len(autoencoder.decoding_layers) == 1

  assert autoencoder.encoding_layers[0].weight.size(0) == 200
  assert autoencoder.decoding_layers[0].weight.size(1) == 200


def test_forward(autoencoder):
  batch_size = 32
  input = torch.rand(batch_size, 5)
  input_words = torch.LongTensor([10, 126, 452, 29, 34])
  output = autoencoder(input, input_words=input_words)

  assert output.size(0) == batch_size
  assert output.size(1) == input_words.size(0)

  target_words = torch.LongTensor([31, 14, 95, 49, 10, 36, 239])
  output = autoencoder(input, input_words=input_words,
                       target_words=target_words)
  assert output.size(0) == batch_size
  assert output.size(1) == target_words.size(0)

  output = autoencoder(input, input_words=input_words,
                       full_output=True)
  assert output.size(0) == batch_size
  assert output.size(1) == 500
