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
- "pair_mask" ([*, N_res, N_res]): 2-D pair mask
- "extra_msa_mask" ([*, N_extra, N_res]): Extra MSA mask
- "template_mask" ([*, N_templ]): Template mask (on the level of templates, not residues)
- "template_aatype" ([*, N_templ, N_res]): Tensor of template residue indices (indices greater than 19 are clamped to 20 (Unknown))
- "template_all_atom_positions" ([*, N_templ, N_res, 37, 3]): Template atom coordinates in atom37 format
- "template_all_atom_mask" ([*, N_templ, N_res, 37]): Template atom coordinate mask
- "template_pseudo_beta" ([*, N_templ, N_res, 3]): Positions of template carbon ("pseudo-beta") atoms (i.e. C_beta for all residues but glycine, for for which C_alpha is used instead)
- "template_pseudo_beta_mask" ([*, N_templ, N_res]): Pseudo-beta mask