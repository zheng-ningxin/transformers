import torch
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification, HubertConfig
from transformers.optimization import AdamW
from datasets import load_dataset
import soundfile as sf
from torchaudio.sox_effects import apply_effects_file
from nni.algorithms.compression.pytorch.pruning import LevelPruner

def finegrain_pruned_hubert(model, sparsity):
    config_list = [{'op_types':['Linear'], 'sparsity':sparsity}]
    pruner = LevelPruner(model, config_list)
    pruner.compress()
    return model, pruner


