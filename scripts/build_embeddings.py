from recoder.embedding import EmbeddingsIndex
from recoder.modules import AutoEncoderRecommender
import torch

def build_embeddings_all_layers(model):
  num_layers = len(model.autoencoder.layer_sizes)
  weights_v = list(model.autoencoder.parameters())[:num_layers-1]
  weights = [w.data for w in weights_v]
  weights.reverse()

  embeddings = weights[0]
  for w in weights[1:]:
    embeddings = torch.matmul(embeddings, w)

  return embeddings

def build_embeddings_first_layer(model):
  return list(model.autoencoder.parameters())[0].data

model_file = ''
model = AutoEncoderRecommender(mode='model',params={'model':model_file})

index = EmbeddingsIndex(embeddings=build_embeddings_first_layer(model),
                        index_file=model_file+'.index', id_map=model.item_id_map)

index.build()
