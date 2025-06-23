import copy
from typing import Mapping, Tuple, List, Dict, Sequence

import ml_collections
import numpy as np
import torch

from vinafold.data import input_pipeline


FeatureDict = Mapping[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]


def np_to_tensor_dict(
    np_example: Mapping[str, np.ndarray],
    features: Sequence[str],
) -> TensorDict:
    """Creates dict of tensors from a dict of NumPy arrays."""
    to_tensor = lambda t: torch.tensor(t) if type(t) != torch.Tensor else t.clone().detach()
    tensor_dict = {
        k: to_tensor(v) for k, v in np_example.items() if k in features
    }

    return tensor_dict

def make_data_config(
    config: ml_collections.ConfigDict,
    num_res: int,
) -> Tuple[ml_collections.ConfigDict, List[str]]:
    cfg = copy.deepcopy(config)
    mode_cfg = cfg["train"]
    with cfg.unlocked():
        if mode_cfg.crop_size is None:
            mode_cfg.crop_size = num_res

    feature_names = cfg.common.unsupervised_features
    if cfg["train"].supervised:
        feature_names += cfg.supervised.supervised_features

    return cfg, feature_names

def np_example_to_features(
    np_example: FeatureDict,
    config: ml_collections.ConfigDict
):
    np_example = dict(np_example)

    seq_length = np_example["seq_length"]
    num_res = int(seq_length[0]) if seq_length.ndim != 0 else int(seq_length)
    cfg, feature_names = make_data_config(config, num_res=num_res)
 
    if "deletion_matrix_int" in np_example:
        np_example["deletion_matrix"] = np_example.pop(
            "deletion_matrix_int"
        ).astype(np.float32)

    tensor_dict = np_to_tensor_dict(
        np_example=np_example, features=feature_names
    )

    with torch.no_grad():
        features = input_pipeline.process_tensors_from_config(
            tensor_dict,
            cfg.common,
            cfg["train"],
        )

    p = torch.rand(1).item()
    use_clamped_fape_value = float(p < cfg.supervised.clamp_prob)
    features["use_clamped_fape"] = torch.full(
        size=[cfg.common.max_recycling_iters + 1],
        fill_value=use_clamped_fape_value,
        dtype=torch.float32,
    )

    return {k: v for k, v in features.items()}


class FeaturePipeline:
    def __init__(self, config: ml_collections.ConfigDict):
        self.config = config

    def process_features(
        self,
        raw_features: FeatureDict
    ) -> FeatureDict:        
        return np_example_to_features(
            np_example=raw_features,
            config=self.config
        )
