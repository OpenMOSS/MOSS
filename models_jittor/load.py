import os
import json

import torch
import jittor as jt
import numpy as np
from tqdm import tqdm

def load_from_torch_shard_ckpt(model, ckpt_dir):
    """
    Load sharded checkpoints directly from huggingface dir.
    """
    with open(os.path.join(ckpt_dir, 'pytorch_model.bin.index.json')) as fp:
        ckpt_index = json.load(fp)
    
    total_size = ckpt_index['metadata']['total_size']
    weight_map = ckpt_index['weight_map']

    file_weight_map = {}
    for key, value in weight_map.items():
        # key: param name; value: filename.
        if value not in file_weight_map:
            file_weight_map[value] = []
        file_weight_map[value].append(key)

    load_from_map(model, ckpt_dir, file_weight_map)
    # check_state_dict(model, ckpt_dir, file_weight_map)

def load_from_map(model: jt.Module, ckpt_dir, file_weight_map):

    for filename, names in tqdm(file_weight_map.items()):
        cur_state_dict = torch.load(os.path.join(ckpt_dir, filename))
        for key, value in cur_state_dict.items():
            var = jt.Var(value.numpy())
            if value.requires_grad:
                var.start_grad()
            else:
                var.stop_grad()
            cur_state_dict[key] = var

        model.load_state_dict(cur_state_dict)

        # gc to reduce memory usage
        del cur_state_dict
        jt.sync_all()
        jt.gc()

def check_state_dict(model: jt.Module, ckpt_dir, file_weight_map):
    for filename, names in file_weight_map.items():
        cur_state_dict = torch.load(os.path.join(ckpt_dir, filename))
        for name in names:
            assert np.equal(
                model.state_dict()[name].numpy(), cur_state_dict[name].numpy()).all()

        # gc to reduce memory usage
        del cur_state_dict
        jt.sync_all()
        jt.gc()
