# Recoder
[![Pypi version](https://img.shields.io/badge/dynamic/json.svg?label=pypi&url=https%3A%2F%2Fpypi.org%2Fpypi%2Frecsys-recoder%2Fjson&query=%24.info.version&colorB=blue)](https://pypi.org/project/recsys-recoder/)
[![Docs status](https://readthedocs.org/projects/recoder/badge/?version=latest)](https://recoder.readthedocs.io/en/latest/)

### Introduction

Recoder is a fast implementation for training collaborative filtering latent
factor models with mini-batch based negative sampling following recent work:
- [Towards Large Scale Training Of Autoencoders For Collaborative Filtering](https://arxiv.org/abs/1809.00999).

Currently the only supported latent factor model is the Autoencoder. SGD Matrix factorization to be added next.

Check out the [Documentation](https://recoder.readthedocs.io/en/latest/).

### Installation
```bash
pip install -U recsys-recoder
```

### Mini-batch based negative sampling
The main contribution of this project is the mini-batch based negative sampling method,
which is based on the simple idea of sampling, for each user, only the negative items
that the other users in the mini-batch have interacted with. This sampling procedure
is biased toward popular items and in order to tune the sampling probability of each
negative item, one has to tune the training batch-size.

### Benchmarks
Benchmarks can be found in the **Results** section of the [paper](https://arxiv.org/abs/1809.00999).
You can expect a large speed-up with small drop in accuracy. For instance, you can get
MovieLens-20M dataset fully trained with mean squared error in less than a minute on a Nvidia Tesla K80 GPU.

### Examples
Check out the `scripts/` directory for some good examples on different datasets.

### Basic Usage

Set `num_neg_samples` in `Recoder.train` to `0` in order to enable mini-batch based
negative sampling, and tune the negative sampling with the `batch_size`. Typically,
for large datasets with large number of items, you may want to increase the number of
negative samples which requires increasing the batch size, but increasing the batch size
too much isn't practical. In this case, set `batch_size` to any practical value, but tune
`num_sampling_users` instead for negative sampling, which generates the batch with a
size of `num_sampling_users` and then slices it into smaller batches of size `batch_size`
and train on those, while keeping the same number of negative samples.


```python
import pandas as pd

from recoder.model import Recoder
from recoder.data import RecommendationDataset

train_df = pd.read_csv('train.csv')

train_dataset = RecommendationDataset()
train_dataset.fill_from_dataframe(dataframe=train_df, num_workers=4)

model_params = {
  'activation_type': 'tanh',
  'noise_prob': 0.5,
}

model = Recoder(hidden_layers=[200], model_params=model_params,
                use_cuda=True, optimizer_type='adam', loss='mse',
                loss_params={'confidence': 3})

model.train(train_dataset=train_dataset, batch_size=500,
            lr=1e-3, weight_decay=2e-5, num_epochs=100,
            num_data_workers=4, num_neg_samples=0)
```

### Further Reading
- [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf)
- [Variational Autoencoders for Collaborative Filtering](https://arxiv.org/abs/1802.05814)

### Citing
Please cite this paper in your publications if it helps your research:
```
@inproceedings{recoder,
  author = {Moussawi, Abdallah},
  title = {Towards Large Scale Training Of Autoencoders For Collaborative Filtering},
  booktitle = {Proceedings of Late-Breaking Results track part of the Twelfth ACM Conference on Recommender Systems},
  series = {RecSys'18},
  year = {2018},
  address = {Vancouver, BC, Canada}
}
```

### Acknowledgements
- I would like to thank [Anghami](https://www.anghami.com) for supporting this work,
and my colleagues, [Helmi Rifai](https://twitter.com/RifaiHelmi) and
[Ramzi Karam](https://twitter.com/ramzikaram), for great discussions on Collaborative Filtering at scale.

- This project started as a fork of [NVIDIA/DeepRecommender](https://github.com/NVIDIA/DeepRecommender),
and although it went in a slightly different direction and was entirely refactored,
the work in [NVIDIA/DeepRecommender](https://github.com/NVIDIA/DeepRecommender) was a
great contribution to the work here.
