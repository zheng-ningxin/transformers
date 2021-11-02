import torch
from copy import deepcopy
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification, HubertConfig, HubertModel
from transformers.optimization import AdamW
from datasets import load_dataset
import soundfile as sf
from torchaudio.sox_effects import apply_effects_file
from nni.algorithms.compression.pytorch.pruning import LevelPruner
from nni.compression.pytorch.utils.utils import get_module_by_name
def finegrain_pruned_hubert(model, sparsity):
    config_list = [{'op_types':['Linear'], 'sparsity':sparsity}]
    pruner = LevelPruner(model, config_list)
    pruner.compress()
    return model, pruner


def copy_tensor(t1, t2):
    shape_list = list(t1.size())
    index = []
    for _size in shape_list:
        index.append(slice(0, _size))
    t1.data = t2.data[index]

def inherit_weight(model, ori_model):
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            print(type(module))
            # if isinstance(module, (torch.nn.LayerNorm, torch.nn.GroupNorm)):
            #     import pdb; pdb.set_trace()
            _, ori_module = get_module_by_name(ori_model, name)
            copy_tensor(module.weight, ori_module.weight)
            # import pdb; pdb.set_trace()
            if hasattr(module, 'bias') and module.bias is not None:
                copy_tensor(module.bias, ori_module.bias)
    # import pdb; pdb.set_trace()
# def coarsegrain_pruned_hubert(model, sparsity):
#     remained = 1- sparsity
#     config = deepcopy(model.config)
#     for i, _ in enumerate(config.conv_dim):
#         if i == len(config.conv_dim) - 1:
#             continue
#         config.conv_dim[i] = int(config.conv_dim[i] * remained) // 16 * 16

#     config.intermediate_size = int(config.intermediate_size * remained) // 16 * 16
#     config.hidden_size = int(config.hidden_size * remained) // 16 * 16
#     config.num_attention_heads = int(config.num_attention_heads*sparsity)
#     new_model = HubertForSequenceClassification(config)
#     inherit_weight(new_model, model)
#     return new_model

def coarsegrain_pruned_hubert(model, sparsity):
    # coarse grained pruning head
    remained = 1- sparsity
    config = deepcopy(model.config)

#    config.conv_dim = [512, 256, 256, 256, 256, 256, 512]

    #config.intermediate_size = 768
    #config.hidden_size = 192
    #config.num_attention_heads = 3
    config.intermediate_size = 512
    config.hidden_size = 128
    config.num_attention_heads = 2
    new_model = HubertForSequenceClassification(config)
    inherit_weight(new_model, model)
    return new_model
