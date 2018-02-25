from recoder.autoencoder.modules import SparseBatchAutoEncoder
from recoder.modules import AutoEncoderRecommender
from recoder.embedding import EmbeddingsIndex
import heapq

class Recommender(object):

  def recommend(self, user_hist):
    raise NotImplementedError

class SimilarityRecommender(Recommender):

  def __init__(self, embeddings_index: EmbeddingsIndex,
               num_recommendations, scale=1, search_k=10000):
    self.embeddings_index = embeddings_index
    self.scale = scale
    self.num_recommendations = num_recommendations
    self.search_k = search_k

  def recommend(self, user_hist):
    user_items = list(zip(*user_hist))[0]

    num_s = max(int(2 * self.num_recommendations / len(user_items)), 2)

    items_pool = []
    for item_id in user_items:
      if item_id in self.embeddings_index.id_map:
        items_pool += self.embeddings_index.get_nns_by_id(item_id, num_s, search_k=self.search_k)

    items_pool = set(items_pool)

    recommendations_heap = []
    for item_id in items_pool:
      if item_id in user_items: continue
      item_score = 0
      for user_item in user_items:
        if user_item in self.embeddings_index.id_map:
          item_score += self.embeddings_index.get_similarity(item_id, user_item) ** self.scale

      if len(recommendations_heap) < self.num_recommendations:
        heapq.heappush(recommendations_heap, (item_score, item_id))
      elif recommendations_heap[0][0] < item_score:
        heapq.heapreplace(recommendations_heap, (item_score, item_id))

    recommendations = []
    while recommendations_heap:
      recommendations.append(heapq.heappop(recommendations_heap)[1])
    recommendations.reverse()

    return recommendations

class InferenceRecommender(Recommender):

  def __init__(self, model: AutoEncoderRecommender,
               num_recommendations):
    self.model = model
    self.num_recommendations = num_recommendations
    self.item_id_inverse_map = dict([(v,k) for k,v in model.item_id_map.items()])

  def recommend(self, user_hist):
    output = self.model.infer(user_hist)
    output_np = output.data.numpy().reshape(-1)

    top_k = heapq.nlargest(self.num_recommendations * 10, enumerate(output_np), key=lambda k: k[1])
    listened_songs = list(list(zip(*user_hist))[0])

    recommendations = []
    for item_model_id, pred in top_k:
      if len(recommendations) == self.num_recommendations:
        break
      if not self.item_id_inverse_map[item_model_id] in listened_songs:
        recommendations.append(self.item_id_inverse_map[item_model_id])

    return recommendations
