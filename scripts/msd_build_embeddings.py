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

model_file='models/msd/a_15_d_0_128x128x128x256_reco_ae_epoch_2.model'
model = AutoEncoderRecommender(mode='model',params={'model':model_file})

index = EmbeddingsIndex(embeddings=build_embeddings_first_layer(model),
                        index_file=model_file+'.index', id_map=model.item_id_map)
index.build()
embedding = index.get_embedding('SOWEJXA12A6701C574')
print(index.get_nns_by_id('SOWEJXA12A6701C574',n=10,search_k=10000))
print(index.get_nns_by_embedding(embedding,n=10))
print(index.get_similarity('SOWEJXA12A6701C574','SOMJFPG12A58A7DD95'))
