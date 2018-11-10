# Tutorial

In this quick tutorial, we will show how to:
- Train an Autoencoder model and a Matrix Factorization model
for implicit feedback collaborative filtering.
- Build your own Factorization model and train it.
- Do negative sampling to speed-up training.
- Evaluate the trained model.

### Training

#### Autoencoder Model
```python
import pandas as pd

from recoder.model import Recoder
from recoder.data import RecommendationDataset
from recoder.nn import DynamicAutoencoder


# train_df is a dataframe where each row is a user-item interaction
# and the value of that interaction
train_df = pd.read_csv('train.csv')

train_dataset = RecommendationDataset()
train_dataset.fill_from_dataframe(dataframe=train_df, user_col='user',
                                  item_col='item', inter_col='inter',
                                  num_workers=4)

# Define your model
model = DynamicAutoencoder(hidden_layers=[200], activation_type='tanh',
                           noise_prob=0.5, sparse=True)

# Recoder takes a factorization model and trains it
recoder = Recoder(model=model, use_cuda=True,
                  optimizer_type='adam', loss='logistic')

recoder.train(train_dataset=train_dataset, batch_size=500,
              lr=1e-3, weight_decay=2e-5, num_epochs=100,
              num_data_workers=4, num_neg_samples=0)
```

#### Matrix Factorization Model
Same as training Autoencoder model, just replace the Autoencoder with a Matrix Factorization
Model.

```python
import pandas as pd

from recoder.model import Recoder
from recoder.data import RecommendationDataset
from recoder.nn import MatrixFactorization

# train_df is a dataframe where each row is a user-item interaction
# and the value of that interaction
train_df = pd.read_csv('train.csv')

train_dataset = RecommendationDataset()
train_dataset.fill_from_dataframe(dataframe=train_df, user_col='user',
                                  item_col='item', inter_col='inter',
                                  num_workers=4)

# Define your model
model = MatrixFactorization(embedding_size=200, activation_type='tanh',
                            dropout_prob=0.5, sparse=True)

# Recoder takes a factorization model and trains it
recoder = Recoder(model=model, use_cuda=True,
                  optimizer_type='adam', loss='logistic')

recoder.train(train_dataset=train_dataset, batch_size=500,
              lr=1e-3, weight_decay=2e-5, num_epochs=100,
              num_data_workers=4, num_neg_samples=0)
```


#### Your Own Factorization Model
If you want to build your own Factorization model with the objective
of reconstructing the interactions matrix, all you have to do is implement
``recoder.nn.FactorizationModel`` interface.

```python
import pandas as pd

from recoder.model import Recoder
from recoder.data import RecommendationDataset
from recoder.nn import FactorizationModel

# Implement your model
class YourModel(FactorizationModel):

  def init_model(self, num_items=None, num_users=None):
    # Initializes your model with the number of items and users.
    pass

  def model_params(self):
    # Returns your model parameters in a dict.
    # Used by Recoder when saving the model.
    pass

  def load_model_params(self, model_params):
    # Loads the model parameters into the model.
    # Used by Recoder when loading the model from a snapshot.
    pass

  def forward(self, input, input_users=None,
              input_items=None, target_users=None,
              target_items=None):
    # A forward pass on the model
    # input_users are the users in the input batch
    # input_items are the items in the input batch
    # target_items are the items to be predicted
    pass


# train_df is a dataframe where each row is a user-item interaction
# and the value of that interaction
train_df = pd.read_csv('train.csv')

train_dataset = RecommendationDataset()
train_dataset.fill_from_dataframe(dataframe=train_df, user_col='user',
                                  item_col='item', inter_col='inter',
                                  num_workers=4)

# Define your model
model = YourModel()

# Recoder takes a factorization model and trains it
recoder = Recoder(model=model, use_cuda=True,
                  optimizer_type='adam', loss='logistic')

recoder.train(train_dataset=train_dataset, batch_size=500,
              lr=1e-3, weight_decay=2e-5, num_epochs=100,
              num_data_workers=4, num_neg_samples=0)
```

#### Save your model

```python
# You can save your model while training at different epoch checkpoints using
# model_checkpoint_prefix and checkpoint_freq params

# model state file prefix that will be appended by epoch number
model_checkpoint_prefix = 'models/model_'

recoder.train(train_dataset=train_dataset, batch_size=500,
              lr=1e-3, weight_decay=2e-5, num_epochs=100,
              num_data_workers=4, num_neg_samples=0,
              model_checkpoint_prefix=model_checkpoint_prefix,
              checkpoint_freq=10)

# or you can directly call recoder.save_state
recoder.save_state(model_checkpoint_prefix)
```

#### Continue training
```python
import pandas as pd

from recoder.model import Recoder
from recoder.data import RecommendationDataset
from recoder.nn import DynamicAutoencoder


# train_df is a dataframe where each row is a user-item interaction
# and the value of that interaction
train_df = pd.read_csv('train.csv')

train_dataset = RecommendationDataset()
train_dataset.fill_from_dataframe(dataframe=train_df, user_col='user',
                                  item_col='item', inter_col='inter',
                                  num_workers=4)

# your saved model
model_file = 'models/your_model'

# Initialize your model
# No need to set model parameters since they will be loaded
# when initializing Recoder from a saved model
model = DynamicAutoencoder()


# Initialize Recoder
recoder = Recoder(model=model, use_cuda=True)
recoder.init_from_model_file(model_file)

recoder.train(train_dataset=train_dataset, batch_size=500,
              lr=1e-3, weight_decay=2e-5, num_epochs=100,
              num_data_workers=4, num_neg_samples=0)
```

#### Tips

Recoder supports training with sparse gradients. Sparse gradients training is only
supported currently by the ``torch.optim.SparseAdam`` optimizer. This is specially helpful
for training big embedding layers such as the users and items embedding
layers in the Autoencoder and MatrixFactorization models. Set the ``sparse`` parameter
in ``Autoencoder`` and ``MatrixFactorization`` to ``True`` in order to return sparse gradients
and this can lead to 1.5-2x training speed-up. If you want to build your own model and have
the embedding layers return sparse gradients, ``Recoder`` should be able to detect that.

### Negative sampling

There are two methods to do negative sampling:
- Mini-batch based negative sampling
- Random negative sampling

#### Mini-batch based negative sampling

Mini-batch based negative sampling is based on the simple idea of sampling, for each
user, only the negative items that the other users in the mini-batch have interacted
with. This sampling procedure is biased toward popular items and in order to tune the
sampling probability of each negative item, one has to tune the training batch-size.
Mini-batch based negative sampling can speed-up training by 2-4x while having a small
drop in recommendation performance.

- To use mini-batch based negative sampling, you have to set ``num_neg_samples`` to ``0`` in
``Recoder.train`` and tune it with the ``batch_size``:

```python
recoder.train(train_dataset=train_dataset, batch_size=500,
              lr=1e-3, weight_decay=2e-5, num_epochs=100,
              num_data_workers=4, num_neg_samples=0)
```

- For large datasets with large number of items, we need a large
number of negative samples, hence a large batch size, which makes
the batch not fit into memory and expensive to train on. In that case,
we can simply generate the sparse batch with a large batch size and
then slice it into smaller batches, and train on the small batches.
To do this you can fix the ``batch_size`` to a specific value, and
instead tune the ``num_sampling_users`` in order to increase the number
of negative samples.

```python
recoder.train(train_dataset=train_dataset, batch_size=500,
              num_sampling_users=2000, lr=1e-3, weight_decay=2e-5,
              num_epochs=100, num_data_workers=4, num_neg_samples=0)
```

#### Random negative sampling
To do random negative sampling, set ``num_neg_samples`` to the number
of random negative samples to generate per batch. Note for batch sizes
greater than 1, the random negative sampling is not actually random,
because the examples within the batch will share those random negative
samples.


### Evaluation
You can evaluate your model with different metrics. Currently, there
are 3 metrics implemented: Recall, NDCG, and Average Precision. You can
also implement your own ``recoder.metrics.Metric``.

#### Evaluating your model while training

```python
import pandas as pd

from recoder.model import Recoder
from recoder.data import RecommendationDataset
from recoder.nn import DynamicAutoencoder
from recoder.metrics import AveragePrecision, Recall, NDCG


# train_df is a dataframe where each row is a user-item interaction
# and the value of that interaction
train_df = pd.read_csv('train.csv')

# validation set. Split your val set into two splits.
# One split will be used as input to the model to
# generate predictions, and the other is which the
# model predictions will be evaluated on
val_input_df = pd.read_csv('test_input.csv')
val_target_df = pd.read_csv('test_output.csv')

train_dataset = RecommendationDataset()
train_dataset.fill_from_dataframe(dataframe=train_df, user_col='user',
                                  item_col='item', inter_col='inter',
                                  num_workers=4)

val_target_dataset = RecommendationDataset()
val_dataset = RecommendationDataset(target_dataset=val_target_dataset)

val_target_dataset.fill_from_dataframe(dataframe=val_target_df, user_col='user',
                                       item_col='item', inter_col='inter',
                                       num_workers=4)

val_dataset.fill_from_dataframe(dataframe=val_input_df, user_col='user',
                                item_col='item', inter_col='inter',
                                num_workers=4)

# Define your model
model = DynamicAutoencoder(hidden_layers=[200], activation_type='tanh',
                           noise_prob=0.5, sparse=True)

# Initialize your metrics
metrics = [Recall(k=20, normalize=True), Recall(k=50, normalize=True),
           NDCG(k=100)]

# Recoder takes a factorization model and trains it
recoder = Recoder(model=model, use_cuda=True,
                  optimizer_type='adam', loss='logistic')

recoder.train(train_dataset=train_dataset,
              val_dataset=val_tr_dataset, batch_size=500,
              lr=1e-3, weight_decay=2e-5, num_epochs=100,
              num_data_workers=4, num_neg_samples=0,
              metrics=metrics, eval_num_recommendations=100,
              eval_freq=5)
```

#### Evaluating your model after training
```python
import pandas as pd

from recoder.model import Recoder
from recoder.data import RecommendationDataset
from recoder.nn import DynamicAutoencoder
from recoder.metrics import AveragePrecision, Recall, NDCG


# test set. Split your test set into two splits.
# One split will be used as input to the model to
# generate predictions, and the other is which the
# model predictions will be evaluated on
test_input_df = pd.read_csv('test_input.csv')
test_target_df = pd.read_csv('test_output.csv')

test_target_dataset = RecommendationDataset()
test_dataset = RecommendationDataset(target_dataset=test_target_dataset)

test_target_dataset.fill_from_dataframe(dataframe=test_target_df, user_col='user',
                                        item_col='item', inter_col='inter',
                                        num_workers=4)

test_dataset.fill_from_dataframe(dataframe=test_input_df, user_col='user',
                                 item_col='item', inter_col='inter',
                                 num_workers=4)

# your saved model
model_file = 'models/your_model'

# Initialize your model
# No need to set model parameters since they will be loaded
# when initializing Recoder from a saved model
model = DynamicAutoencoder()

# Initialize your metrics
metrics = [Recall(k=20, normalize=True), Recall(k=50, normalize=True),
           NDCG(k=100)]

# Initialize Recoder
recoder = Recoder(model=model, use_cuda=True)
recoder.init_from_model_file(model_file)

# Evaluate on the top 100 recommendations
num_recommendations = 100

recoder.evaluate(eval_dataset=test_dataset, num_recommendations=num_recommendations,
                 metrics=metrics, batch_size=500)
```
