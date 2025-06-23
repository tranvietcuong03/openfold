from functools import partial
import os
from typing import Optional

import ml_collections as mlc
import pytorch_lightning as pl
import torch
from vinafold.np.residue_constants import restypes
from vinafold.data import (
    data_pipeline,
    feature_pipeline,
    mmcif_parsing
)
from vinafold.utils.tensor_utils import dict_multimap
from vinafold.utils.tensor_utils import (
    tensor_tree_map,
)


class VinaFoldDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, alignment_dir: str, config: mlc.ConfigDict,):
        super(VinaFoldDataset, self).__init__()

        self.data_dir = data_dir
        self.alignment_dir = alignment_dir
        self.config = config

        self._chain_ids = list(os.listdir(alignment_dir))
        self._chain_id_to_idx_dict = {
            chain: i for i, chain in enumerate(self._chain_ids)
        }

        self.data_pipeline = data_pipeline.DataPipeline()
        self.feature_pipeline = feature_pipeline.FeaturePipeline(config)

    def _parse_mmcif(self, path, file_id, chain_id, alignment_dir):
        with open(path, 'r') as f:
            mmcif_string = f.read()

        mmcif_object = mmcif_parsing.parse(
            file_id=file_id, mmcif_string=mmcif_string
        )

        if mmcif_object.mmcif_object is None:
            raise list(mmcif_object.errors.values())[0]

        mmcif_object = mmcif_object.mmcif_object

        data = self.data_pipeline.process_mmcif(
            mmcif=mmcif_object,
            alignment_dir=alignment_dir,
            chain_id=chain_id
        )

        return data

    def chain_id_to_idx(self, chain_id):
        return self._chain_id_to_idx_dict[chain_id]

    def idx_to_chain_id(self, idx):
        return self._chain_ids[idx]

    def __getitem__(self, idx):
        name = self.idx_to_chain_id(idx)
        alignment_dir = os.path.join(self.alignment_dir, name)

        file_id, chain_id = name.rsplit('_', 1)
        path = os.path.join(self.data_dir, file_id + ".cif")
        data = self._parse_mmcif(
            path, file_id, chain_id, alignment_dir
        )

        feats = self.feature_pipeline.process_features(data)
        feats["batch_idx"] = torch.tensor(
            [idx for _ in range(feats["aatype"].shape[-1])],
            dtype=torch.int64,
            device=feats["aatype"].device)

        return feats

    def __len__(self):
        return len(self._chain_ids)


def resolution_filter(resolution: int, max_resolution: float) -> bool:
    """Check that the resolution is <= max_resolution permitted"""
    return resolution is not None and resolution <= max_resolution


def aa_count_filter(seqs: list, max_single_aa_prop: float) -> bool:
    """Check if any single amino acid accounts for more than max_single_aa_prop percent of the sequence(s)"""
    counts = {}
    for seq in seqs:
        for aa in seq:
            counts.setdefault(aa, 0)
            if aa not in restypes:
                return False
            else:
                counts[aa] += 1

    total_len = sum([len(i) for i in seqs])
    largest_aa_count = max(counts.values())
    largest_single_aa_prop = largest_aa_count / total_len
    return largest_single_aa_prop <= max_single_aa_prop


def all_seq_len_filter(seqs: list, minimum_number_of_residues: int) -> bool:
    """Check if the total combined sequence lengths are >= minimum_numer_of_residues"""
    total_len = sum([len(i) for i in seqs])
    return total_len >= minimum_number_of_residues


class VinaFoldBatchCollator:
    def __call__(self, prots):
        stack_fn = partial(torch.stack, dim=0)
        return dict_multimap(stack_fn, prots)


class VinaFoldDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, config, stage="train", generator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.stage = stage
        self.generator = generator
        self._prep_batch_properties_probs()

    def _prep_batch_properties_probs(self):
        keyed_probs = []
        stage_cfg = self.config[self.stage]

        max_iters = self.config.common.max_recycling_iters

        if stage_cfg.uniform_recycling:
            recycling_probs = [
                1. / (max_iters + 1) for _ in range(max_iters + 1)
            ]
        else:
            recycling_probs = [
                0. for _ in range(max_iters + 1)
            ]
            recycling_probs[-1] = 1.

        keyed_probs.append(
            ("no_recycling_iters", recycling_probs)
        )

        keys, probs = zip(*keyed_probs)
        max_len = max([len(p) for p in probs])
        padding = [[0.] * (max_len - len(p)) for p in probs]

        self.prop_keys = keys
        self.prop_probs_tensor = torch.tensor(
            [p + pad for p, pad in zip(probs, padding)],
            dtype=torch.float32,
        )

    def _add_batch_properties(self, batch):
        gt_features = batch.pop('gt_features', None)
        samples = torch.multinomial(
            self.prop_probs_tensor,
            num_samples=1,  # 1 per row
            replacement=True,
            generator=self.generator
        )

        aatype = batch["aatype"]
        batch_dims = aatype.shape[:-2]
        recycling_dim = aatype.shape[-1]
        no_recycling = recycling_dim
        for i, key in enumerate(self.prop_keys):
            sample = int(samples[i][0])
            sample_tensor = torch.tensor(
                sample,
                device=aatype.device,
                requires_grad=False
            )
            orig_shape = sample_tensor.shape
            sample_tensor = sample_tensor.view(
                (1,) * len(batch_dims) + sample_tensor.shape + (1,)
            )
            sample_tensor = sample_tensor.expand(
                batch_dims + orig_shape + (recycling_dim,)
            )
            batch[key] = sample_tensor

            if key == "no_recycling_iters":
                no_recycling = sample

        resample_recycling = lambda t: t[..., :no_recycling + 1]
        batch = tensor_tree_map(resample_recycling, batch)
        batch['gt_features'] = gt_features

        return batch

    def __iter__(self):
        it = super().__iter__()

        def _batch_prop_gen(iterator):
            for batch in iterator:
                yield self._add_batch_properties(batch)

        return _batch_prop_gen(it)


class VinaFoldDataModule(pl.LightningDataModule):
    def __init__(self,
                 config: mlc.ConfigDict,
                 train_data_dir: Optional[str] = None,
                 train_alignment_dir: Optional[str] = None,
                 batch_seed: Optional[int] = None,
                 ):
        super(VinaFoldDataModule, self).__init__()

        self.config = config
        self.train_data_dir = train_data_dir
        self.train_alignment_dir = train_alignment_dir
        self.batch_seed = batch_seed

    def setup(self, stage=None):
        self.train_dataset = VinaFoldDataset(
            config=self.config,
            data_dir=self.train_data_dir,
            alignment_dir=self.train_alignment_dir,
        )        

    def train_dataloader(self):
        generator = None
        if self.batch_seed is not None:
            generator = torch.Generator()
            generator = generator.manual_seed(self.batch_seed)

        batch_collator = VinaFoldBatchCollator()

        dl = VinaFoldDataLoader(
            self.train_dataset,
            config=self.config,
            generator=generator,
            batch_size=self.config.data_module.data_loaders.batch_size,
            num_workers=self.config.data_module.data_loaders.num_workers,
            collate_fn=batch_collator,
        )

        return dl

    def val_dataloader(self):
        return [] 

    def predict_dataloader(self):
        return []