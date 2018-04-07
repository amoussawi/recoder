from recoder.embedding import EmbeddingsIndex

import numpy as np

import recoder.utils as utils

from sklearn.metrics.pairwise import cosine_similarity


class Recommender(object):

  def recommend(self, users_hist):
    raise NotImplementedError

class SimilarityRecommender(Recommender):

  def __init__(self, embeddings_index: EmbeddingsIndex,
               num_recommendations, n=1, scale=1):
    self.embeddings_index = embeddings_index
    self.scale = scale
    self.num_recommendations = num_recommendations
    self.n = n

  def __recommend_single(self, user_hist):
    user_items = np.array(utils.unzip(user_hist)[0])

    items_pool = [self.embeddings_index.get_nns_by_id(item_id, self.n)
                  for item_id in user_items]

    items_pool = np.unique(items_pool)
    filtered_items = items_pool[np.isin(items_pool, user_items, invert=True)]

    items_scores = self.__compute_scores(filtered_items, user_items)

    if len(items_scores) > self.num_recommendations:
      top_ind_not_sorted = np.argpartition(-items_scores, self.num_recommendations)
      top_ind_not_sorted = top_ind_not_sorted[:self.num_recommendations]
    else:
      top_ind_not_sorted = np.arange(len(items_scores))

    top_sorted_reset_ind = np.argsort(-items_scores[top_ind_not_sorted])

    top_ind_sorted = top_ind_not_sorted[top_sorted_reset_ind]
    top_items = filtered_items[top_ind_sorted]

    return top_items

  def __compute_scores(self, items_pool, user_items):
    pool_embeddings = np.array([self.embeddings_index.get_embedding(item_id)
                                for item_id in items_pool])
    user_embeddings = np.array([self.embeddings_index.get_embedding(item_id)
                                for item_id in user_items])

    scores = cosine_similarity(pool_embeddings, user_embeddings) # range: -1 to 1
    scores = (scores + 1) / 2 # range: 0 to 1

    scaled_scores = np.power(scores, self.scale)

    agg_scores = np.sum(scaled_scores, axis=1)

    return agg_scores

  def recommend(self, users_hist):
    recommendations = [self.__recommend_single(user_hist)
                       for user_hist in users_hist]
    return recommendations


class InferenceRecommender(Recommender):

  def __init__(self, model,
               num_recommendations):
    self.model = model
    self.num_recommendations = num_recommendations
    self.item_id_inverse_map = dict([(v,k) for k,v in model.item_id_map.items()])

  def recommend(self, users_hist):
    output, input = self.model.infer(users_hist, return_input=True)
    input = input.numpy()
    output = output.data.numpy()

    output[input > 0] = - np.inf

    top_ind = np.argpartition(-output, self.num_recommendations, axis=1)
    top_ind = top_ind[:, :self.num_recommendations]
    top_output = output[np.arange(output.shape[0])[:, None], top_ind]

    top_sorted_reset_ind = np.argsort(-top_output, axis=1)

    top_sorted_ind = top_ind[np.arange(top_ind.shape[0])[:, None], top_sorted_reset_ind]

    item_id_mapper = np.vectorize(lambda item_id: self.item_id_inverse_map[item_id])

    recommendations = item_id_mapper(top_sorted_ind)

    return recommendations
