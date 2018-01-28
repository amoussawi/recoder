import annoy as an
import torch
import glog as log

class EmbeddingsIndex(object):

  def __init__(self, embeddings=None, index_file=None,
               id_map=None, n_trees=10):
    self.embeddings = embeddings
    self.index_file = index_file
    self.n_trees = n_trees
    self.id_map = id_map

  def build(self):
    self.__build_index()

  def load(self):
    self.__load_index()

  def __build_index(self):
    self.embedding_size = self.embeddings.size(0)

    self.index = an.AnnoyIndex(self.embedding_size, metric='angular')

    for embedding_ind in range(self.embeddings.size(1)):
      embedding = self.embeddings.index_select(1, torch.LongTensor([embedding_ind]))
      embedding_np = embedding.numpy()
      self.index.add_item(embedding_ind, embedding_np)

    self.index.build(self.n_trees)

    self.inverse_id_map = dict([(v,k) for k,v in self.id_map.items()])

    if self.index_file:
      embeddings_file = self.index_file + '.embeddings'
      state = {
        'embedding_size': self.embedding_size,
        'id_map': self.id_map,
        'embeddings_file': embeddings_file,
      }

      self.index.save(embeddings_file)
      torch.save(state, self.index_file)

  def __load_index(self):
    log.info('Loading index file from {}'.format(self.index_file))
    state = torch.load(self.index_file)
    self.embedding_size = state['embedding_size']
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
