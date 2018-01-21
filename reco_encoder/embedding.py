import annoy as an
from reco_encoder.modules import AutoEncoderRecommender
import torch

class EmbeddingsIndex(object):

  def __init__(self, model=None, index_file=None,
               is_item_index=True, n_trees=10):
    self.model = model # type: AutoEncoderRecommender
    self.index_file = index_file
    self.n_trees = n_trees
    self.is_item_index = is_item_index

  def build(self):
    self.__build_index()

  def load(self):
    self.__load_index()

  def __build_index(self):
    num_layers = len(self.model.autoencoder.layer_sizes)
    self.embedding_size = self.model.autoencoder.layer_sizes[-1]

    self.index = an.AnnoyIndex(self.embedding_size, metric='angular')

    weights_v = list(self.model.autoencoder.parameters())[:num_layers-1]
    weights = [w.data for w in weights_v]
    weights.reverse()

    embeddings = weights[0]
    for w in weights[1:]:
      embeddings = torch.matmul(embeddings, w)

    for embedding_ind in range(embeddings.size(1)):
      embedding = embeddings.index_select(1, torch.LongTensor([embedding_ind]))
      embedding_np = embedding.numpy()
      self.index.add_item(embedding_ind, embedding_np)

    self.index.build(self.n_trees)

    self.id_map = self.model.item_id_map if self.is_item_index else self.model.user_id_map
    self.inverse_id_map = dict([(v,k) for k,v in self.id_map.items()])

    if self.index_file:
      embeddings_file = self.index_file + '.embeddings'
      state = {
        'embedding_size': self.embedding_size,
        'is_item_index': self.is_item_index,
        'id_map': self.id_map,
        'embeddings_file': embeddings_file,
      }
      self.index.save(embeddings_file)
      torch.save(state, self.index_file)

  def __load_index(self):
    state = torch.load(self.index_file)
    self.embedding_size = state['embedding_size']
    self.is_item_index = state['is_item_index']
    self.id_map = state['id_map']
    embeddings_file = state['embeddings_file']
    self.index = an.AnnoyIndex(self.embedding_size, metric='angular')
    self.index.load(embeddings_file)
    self.inverse_id_map = dict([(v,k) for k,v in self.id_map.items()])

  def get_embedding(self, id):
    return self.index.get_item_vector(self.id_map[id])

  def get_nns_by_id(self, id, n, search_k=-1, include_distances=False):
    nearest_indices = self.index.get_nns_by_item(self.id_map[id], n, search_k=search_k,
                                                 include_distances=include_distances)
    nearest_ids = [self.inverse_id_map[ind] for ind in nearest_indices]
    return nearest_ids

  def get_nns_by_embedding(self, embedding, n, search_k=-1, include_distances=False):
    nearest_indices = self.index.get_nns_by_vector(embedding, n, search_k=search_k,
                                                   include_distances=include_distances)
    nearest_ids = [self.inverse_id_map[ind] for ind in nearest_indices]
    return nearest_ids

  def get_similarity(self, id1, id2):
    distance = self.index.get_distance(self.id_map[id1], self.id_map[id2])
    cosine_similarity = 1 - (distance**2) / 2 # range from -1 to 1
    similarity = (cosine_similarity + 1) / 2 # range from 0 to 1
    return similarity
