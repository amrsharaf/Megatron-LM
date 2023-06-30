from megatron.data.gpt_dataset import get_indexed_dataset_
from megatron.utils import print_rank_0
from megatron.data.gpt_dataset import _num_tokens
import numpy as np
import json

with open('tlgv5.2.0.json', 'r') as f:
    data = json.load(f)
# print keys in a nice format
for key in data.keys():
    data_prefix = key.replace('/mnt/stream/', '/turingnorwayeastpremium_data/')
    print_rank_0(' > data prefix: {}'.format(data_prefix))
    indexed_dataset = get_indexed_dataset_(data_prefix, data_impl='infer', skip_warmup=True)
    tokens_per_epoch = _num_tokens(np.arange(indexed_dataset.sizes.shape[0]), indexed_dataset.sizes)
    print_rank_0(' > tokens per epoch: {:,}'.format(tokens_per_epoch))
