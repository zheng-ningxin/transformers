import argparse
import glob
import json
import logging

import os
import random
import math

import numpy as np
import torch
# from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from nni.compression.pytorch.utils import get_module_by_name
from emmental import MaskedBertConfig, MaskedBertForSequenceClassification
from transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                        BertForSequenceClassification,
                        BertTokenizer,
                        get_linear_schedule_with_warmup,
                                )
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from emmental.modules.masked_nn import MaskedLinear
import nni
import torch
import sys
import os
from nni.algorithms.compression.pytorch.pruning import LevelPruner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ck_path', required=True, help='ckpath of the masked bert')
    args = parser.parse_args()
    config = MaskedBertConfig.from_pretrained(args.ck_path)
    tokenizer = BertTokenizer.from_pretrained(args.ck_path)
    mask_model = MaskedBertForSequenceClassification.from_pretrained(args.ck_path, config=config)

if __name__ == '__main__':
    main()
