batch: Dictionary of arguments outlined in Algorithm 2. Keys must include the official names of the features in thesupplement subsection 1.2.9.

The final dimension of each input must have length equal to
the number of recycling iterations.

Features (without the recycling dimension):

- "aatype" ([*, N_res]): Contrary to the supplement, this tensor of residue indices is not one-hot.
- "target_feat" ([*, N_res, C_tf]): One-hot encoding of the target sequence. C_tf is config.model.input_embedder.tf_dim.
- "residue_index" ([*, N_res]): Tensor whose final dimension consists of consecutive indices from 0 to N_res.
- "msa_feat" ([*, N_seq, N_res, C_msa]): MSA features, constructed as in the supplement. C_msa is config.model.input_embedder.msa_dim.
- "seq_mask" ([*, N_res]): 1-D sequence mask
- "msa_mask" ([*, N_seq, N_res]): MSA mask

dict_keys(['aatype', 'residue_index', 'seq_length', 'all_atom_positions', 'all_atom_mask', 'resolution', 'is_distillation', 'seq_mask', 'msa_mask', 'msa_row_mask', 'atom14_atom_exists', 'residx_atom14_to_atom37', 'residx_atom37_to_atom14', 'atom37_atom_exists', 'atom14_gt_exists', 'atom14_gt_positions', 'atom14_alt_gt_positions', 'atom14_alt_gt_exists', 'atom14_atom_is_ambiguous', 'rigidgroups_gt_frames', 'rigidgroups_gt_exists', 'rigidgroups_group_exists', 'rigidgroups_group_is_ambiguous', 'rigidgroups_alt_gt_frames', 'pseudo_beta', 'pseudo_beta_mask', 'backbone_rigid_tensor', 'backbone_rigid_mask', 'chi_angles_sin_cos', 'chi_mask', 'extra_msa', 'extra_msa_mask', 'extra_msa_row_mask', 'bert_mask', 'true_msa', 'extra_has_deletion', 'extra_deletion_value', 'msa_feat', 'target_feat', 'use_clamped_fape', 'batch_idx', 'no_recycling_iters'])