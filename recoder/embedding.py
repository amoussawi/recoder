import annoy as an

import pickle

import glog as log


class EmbeddingsIndex(object):

  def get_embedding(self, embedding_id):
    raise NotImplementedError

  def get_nns_by_id(self, embedding_id, n):
    raise NotImplementedError

  def get_nns_by_embedding(self, embedding, n):
    raise NotImplementedError

  def get_similarity(self, id1, id2):
    raise NotImplementedError


class AnnoyEmbeddingsIndex(EmbeddingsIndex):

  def __init__(self, embeddings=None, index_file=None,
               id_map=None, n_trees=10, search_k=-1,
               include_distances=False):
    self.embeddings = embeddings
    self.index_file = index_file
    self.n_trees = n_trees
    self.id_map = id_map
    self.search_k = search_k
    self.include_distances = include_distances

  def build(self):
    self.__build_index()

  def load(self):
    self.__load_index()

  def __build_index(self):
    self.embedding_size = self.embeddings.shape[1]

    self.index = an.AnnoyIndex(self.embedding_size, metric='angular')

    for embedding_ind in range(self.embeddings.shape[0]):
      embedding = self.embeddings[embedding_ind, :]
      self.index.add_item(embedding_ind, embedding)

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
      with open(self.index_file, 'wb') as _index_file:
        pickle.dump(state, _index_file)

  def __load_index(self):
    log.info('Loading index file from {}'.format(self.index_file))
    with open(self.index_file, 'rb') as _index_file:
      state = pickle.load(_index_file)
    self.embedding_size = state['embedding_size']
    self.id_map = state['id_map']
    embeddings_file = state['embeddings_file']
    self.index = an.AnnoyIndex(self.embedding_size, metric='angular')
    self.index.load(embeddings_file)
    self.inverse_id_map = dict([(v,k) for k,v in self.id_map.items()])

  def get_embedding(self, embedding_id):
    return self.index.get_item_vector(self.id_map[embedding_id])

  def get_nns_by_id(self, embedding_id, n):
    nearest_indices = self.index.get_nns_by_item(self.id_map[embedding_id], n, search_k=self.search_k,
                                                 include_distances=self.include_distances)
    nearest_ids = [self.inverse_id_map[ind] for ind in nearest_indices]
    return nearest_ids

  def get_nns_by_embedding(self, embedding, n):
    nearest_indices = self.index.get_nns_by_vector(embedding, n, search_k=self.search_k,
                                                   include_distances=self.include_distances)
    nearest_ids = [self.inverse_id_map[ind] for ind in nearest_indices]
    return nearest_ids

  def get_similarity(self, id1, id2):
    distance = self.index.get_distance(self.id_map[id1], self.id_map[id2])
    cosine_similarity = 1 - (distance**2) / 2 # range from -1 to 1
    similarity = (cosine_similarity + 1) / 2 # range from 0 to 1
    return similarity


class MemCacheEmbeddingsIndex(EmbeddingsIndex):

  def __init__(self, embedding_index):
    self.embedding_index = embedding_index # type: EmbeddingsIndex
    self.__nns_cache = {}

  def get_embedding(self, embedding_id):
    return self.embedding_index.get_embedding(embedding_id)

  def get_nns_by_embedding(self, embedding, n):
    return self.embedding_index.get_nns_by_embedding(embedding, n)

  def get_nns_by_id(self, embedding_id, n):
    if embedding_id not in self.__nns_cache:
      self.__nns_cache[embedding_id] = self.embedding_index.get_nns_by_id(embedding_id, n)
    return self.__nns_cache[embedding_id]

  def get_similarity(self, id1, id2):
    return self.embedding_index.get_similarity(id1, id2)
