import torch
import torch.nn as nn

from vinafold.model.input_embedder import InputEmbedder
from vinafold.model.recycling_embeddrt import RecyclingEmbedder
from vinafold.model.evoformer.evoformer import EvoformerStack
from vinafold.model.heads import AuxiliaryHeads
from vinafold.model.structure_module import StructureModule

import vinafold.np.residue_constants as residue_constants
from vinafold.utils.feats import (
    pseudo_beta_fn,
    atom14_to_atom37,
)
from vinafold.utils.tensor_utils import (
    add,
    tensor_tree_map,
)


class AlphaFold(nn.Module):

    def __init__(self, config):
        super(AlphaFold, self).__init__()

        self.globals = config.globals
        self.config = config.model
        
        self.input_embedder = InputEmbedder(
            **self.config["input_embedder"],
        )

        self.recycling_embedder = RecyclingEmbedder(
            **self.config["recycling_embedder"],
        )

        self.evoformer = EvoformerStack(
            **self.config["evoformer_stack"],
        )

        self.structure_module = StructureModule(
            **self.config["structure_module"],
        )
        self.aux_heads = AuxiliaryHeads(
            self.config["heads"],
        )

    def iteration(self, feats, prevs, _recycle=True):
        outputs = {}

        dtype = next(self.parameters()).dtype
        for k in feats:
            if feats[k].dtype == torch.float32:
                feats[k] = feats[k].to(dtype=dtype)

        batch_dims = feats["target_feat"].shape[:-2]
        n = feats["target_feat"].shape[-2]
        n_seq = feats["msa_feat"].shape[-3]

        inplace_safe = not (self.training or torch.is_grad_enabled())

        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats["msa_mask"]

        m, z = self.input_embedder(
            feats["target_feat"],
            feats["residue_index"],
            feats["msa_feat"],
            inplace_safe=inplace_safe,
        )

        m_1_prev, z_prev, x_prev = reversed([prevs.pop() for _ in range(3)])

        if None in [m_1_prev, z_prev, x_prev]:
            # [*, N, C_m]
            m_1_prev = m.new_zeros(
                (*batch_dims, n, self.config.input_embedder.c_m),
                requires_grad=False,
            )

            # [*, N, N, C_z]
            z_prev = z.new_zeros(
                (*batch_dims, n, n, self.config.input_embedder.c_z),
                requires_grad=False,
            )

            # [*, N, 3]
            x_prev = z.new_zeros(
                (*batch_dims, n, residue_constants.atom_type_num, 3),
                requires_grad=False,
            )

        pseudo_beta_x_prev = pseudo_beta_fn(
            feats["aatype"], x_prev, None
        ).to(dtype=z.dtype)

        # m_1_prev_emb: [*, N, C_m]
        # z_prev_emb: [*, N, N, C_z]
        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m_1_prev,
            z_prev,
            pseudo_beta_x_prev,
            inplace_safe=inplace_safe,
        )

        del pseudo_beta_x_prev


        # [*, S_c, N, C_m]
        m[..., 0, :, :] += m_1_prev_emb

        # [*, N, N, C_z]
        z = add(z, z_prev_emb, inplace=inplace_safe)

        del m_1_prev, z_prev, m_1_prev_emb, z_prev_emb

        m, z, s = self.evoformer(
            m,
            z,
            msa_mask=msa_mask.to(dtype=m.dtype),
            pair_mask=pair_mask.to(dtype=z.dtype),
            chunk_size=self.globals.chunk_size,
            use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
            use_lma=self.globals.use_lma,
            use_flash=self.globals.use_flash,
            inplace_safe=inplace_safe,
            _mask_trans=self.config._mask_trans,
        )

        outputs["msa"] = m[..., :n_seq, :, :]
        outputs["pair"] = z
        outputs["single"] = s

        del z

        # Predict 3D structure
        outputs["sm"] = self.structure_module(
            outputs,
            feats["aatype"],
            mask=feats["seq_mask"].to(dtype=s.dtype)
        )
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        # Save embeddings for use during the next recycling iteration

        # [*, N, C_m]
        m_1_prev = m[..., 0, :, :]

        # [*, N, N, C_z]
        z_prev = outputs["pair"]

        del x_prev

        # [*, N, 3]
        x_prev = outputs["final_atom_positions"]

        return outputs, m_1_prev, z_prev, x_prev

    def forward(self, batch):
        print(batch.keys(), end="\n\n")
        raise Exception("Batch Features Exp")
        m_1_prev, z_prev, x_prev = None, None, None
        prevs = [m_1_prev, z_prev, x_prev]

        is_grad_enabled = torch.is_grad_enabled()

        num_iters = batch["aatype"].shape[-1]
        num_recycles = 0
        for cycle_no in range(num_iters):
            fetch_cur_batch = lambda t: t[..., cycle_no]
            feats = tensor_tree_map(fetch_cur_batch, batch)

            is_final_iter = cycle_no == num_iters - 1
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                outputs, m_1_prev, z_prev, x_prev = self.iteration(
                    feats,
                    prevs,
                    _recycle=(num_iters > 1)
                )

                num_recycles += 1

                if not is_final_iter:
                    del outputs
                    prevs = [m_1_prev, z_prev, x_prev]
                    del m_1_prev, z_prev, x_prev
                else:
                    break

        outputs["num_recycles"] = torch.tensor(num_recycles, device=feats["aatype"].device)

        if "asym_id" in batch:
            outputs["asym_id"] = feats["asym_id"]

        outputs.update(self.aux_heads(outputs))

        return outputs
