import os
import collections
import contextlib
from multiprocessing import cpu_count
import tempfile
from typing import Mapping, Optional, Sequence, Any, MutableMapping, Union
import numpy as np
from vinafold.data import (
    parsers,
    mmcif_parsing,
    msa_identifiers
)
from vinafold.data.tools import jackhmmer, hhblits, hhsearch, hmmsearch
from vinafold.np import residue_constants

FeatureDict = MutableMapping[str, np.ndarray]
TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]

def make_sequence_features(
    sequence: str, description: str, num_res: int
) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["aatype"] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array(
        [description.encode("utf-8")], dtype=object
    )
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array(
        [sequence.encode("utf-8")], dtype=object
    )
    return features

def make_mmcif_features(
    mmcif_object: mmcif_parsing.MmcifObject, chain_id: str
) -> FeatureDict:
    input_sequence = mmcif_object.chain_to_seqres[chain_id]
    description = "_".join([mmcif_object.file_id, chain_id])
    num_res = len(input_sequence)

    mmcif_feats = {}

    mmcif_feats.update(
        make_sequence_features(
            sequence=input_sequence,
            description=description,
            num_res=num_res,
        )
    )

    all_atom_positions, all_atom_mask = mmcif_parsing.get_atom_coords(
        mmcif_object=mmcif_object, chain_id=chain_id
    )
    mmcif_feats["all_atom_positions"] = all_atom_positions
    mmcif_feats["all_atom_mask"] = all_atom_mask

    mmcif_feats["resolution"] = np.array(
        [mmcif_object.header["resolution"]], dtype=np.float32
    )

    mmcif_feats["release_date"] = np.array(
        [mmcif_object.header["release_date"].encode("utf-8")], dtype=object
    )

    mmcif_feats["is_distillation"] = np.array(0., dtype=np.float32)

    return mmcif_feats

def make_msa_features(msas: Sequence[parsers.Msa]) -> FeatureDict:
    """Constructs a feature dict of MSA features."""
    if not msas:
        raise ValueError("At least one MSA must be provided.")

    int_msa = []
    deletion_matrix = []
    species_ids = []
    seen_sequences = set()
    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(
                f"MSA {msa_index} must contain at least one sequence."
            )
        for sequence_index, sequence in enumerate(msa.sequences):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append(
                [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence]
            )

            deletion_matrix.append(msa.deletion_matrix[sequence_index])
            identifiers = msa_identifiers.get_identifiers(
                msa.descriptions[sequence_index]
            )
            species_ids.append(identifiers.species_id.encode('utf-8'))

    num_res = len(msas[0].sequences[0])
    num_alignments = len(int_msa)
    features = {}
    features["deletion_matrix_int"] = np.array(deletion_matrix, dtype=np.int32)
    features["msa"] = np.array(int_msa, dtype=np.int32)
    features["num_alignments"] = np.array(
        [num_alignments] * num_res, dtype=np.int32
    )
    features["msa_species_identifiers"] = np.array(species_ids, dtype=object)
    return features

def run_msa_tool(
    msa_runner,
    fasta_path: str,
    msa_out_path: str,
    msa_format: str,
    max_sto_sequences: Optional[int] = None,
) -> Mapping[str, Any]:
    """Runs an MSA tool, checking if output already exists first."""
    if(msa_format == "sto" and max_sto_sequences is not None):
        result = msa_runner.query(fasta_path, max_sto_sequences)[0]
    else:
        result = msa_runner.query(fasta_path)[0]

    assert msa_out_path.split('.')[-1] == msa_format
    with open(msa_out_path, "w") as f:
        f.write(result[msa_format])

    return result

def make_dummy_msa_obj(input_sequence) -> parsers.Msa:
    deletion_matrix = [[0 for _ in input_sequence]]
    return parsers.Msa(sequences=[input_sequence],
                       deletion_matrix=deletion_matrix,
                       descriptions=['dummy'])

def make_dummy_msa_feats(input_sequence) -> FeatureDict:
    msa_data_obj = make_dummy_msa_obj(input_sequence)
    return make_msa_features([msa_data_obj])

class AlignmentRunner:
    """Runs alignment tools and saves the results"""
    def __init__(
        self,
        jackhmmer_binary_path: Optional[str] = None,
        hhblits_binary_path: Optional[str] = None,
        uniref90_database_path: Optional[str] = None,
        mgnify_database_path: Optional[str] = None,
        bfd_database_path: Optional[str] = None,
        uniref30_database_path: Optional[str] = None,
        uniclust30_database_path: Optional[str] = None,
        uniprot_database_path: Optional[str] = None,
        template_searcher: Optional[TemplateSearcher] = None,
        use_small_bfd: Optional[bool] = None,
        no_cpus: Optional[int] = None,
        uniref_max_hits: int = 10000,
        mgnify_max_hits: int = 5000,
        uniprot_max_hits: int = 50000,
    ):
        db_map = {
            "jackhmmer": {
                "binary": jackhmmer_binary_path,
                "dbs": [
                    uniref90_database_path,
                    mgnify_database_path,
                    bfd_database_path if use_small_bfd else None,
                    uniprot_database_path,
                ],
            },
            "hhblits": {
                "binary": hhblits_binary_path,
                "dbs": [
                    bfd_database_path if not use_small_bfd else None,
                ],
            },
        }

        for name, dic in db_map.items():
            binary, dbs = dic["binary"], dic["dbs"]
            if(binary is None and not all([x is None for x in dbs])):
                raise ValueError(
                    f"{name} DBs provided but {name} binary is None"
                )

        self.uniref_max_hits = uniref_max_hits
        self.mgnify_max_hits = mgnify_max_hits
        self.uniprot_max_hits = uniprot_max_hits
        self.use_small_bfd = use_small_bfd

        if(no_cpus is None):
            no_cpus = cpu_count()

        self.jackhmmer_uniref90_runner = None
        if(jackhmmer_binary_path is not None and
            uniref90_database_path is not None
        ):
            self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
                binary_path=jackhmmer_binary_path,
                database_path=uniref90_database_path,
                n_cpu=no_cpus,
            )

        self.jackhmmer_small_bfd_runner = None
        self.hhblits_bfd_unirefclust_runner = None
        if(bfd_database_path is not None):
            if use_small_bfd:
                self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
                    binary_path=jackhmmer_binary_path,
                    database_path=bfd_database_path,
                    n_cpu=no_cpus,
                )
            else:
                dbs = [bfd_database_path]
                if(uniref30_database_path is not None):
                    dbs.append(uniref30_database_path)
                if (uniclust30_database_path is not None):
                    dbs.append(uniclust30_database_path)
                self.hhblits_bfd_unirefclust_runner = hhblits.HHBlits(
                    binary_path=hhblits_binary_path,
                    databases=dbs,
                    n_cpu=no_cpus,
                )

        self.jackhmmer_mgnify_runner = None
        if(mgnify_database_path is not None):
            self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
                binary_path=jackhmmer_binary_path,
                database_path=mgnify_database_path,
                n_cpu=no_cpus,
            )

        self.jackhmmer_uniprot_runner = None
        if(uniprot_database_path is not None):
            self.jackhmmer_uniprot_runner = jackhmmer.Jackhmmer(
                binary_path=jackhmmer_binary_path,
                database_path=uniprot_database_path,
                n_cpu=no_cpus
            )

        if(template_searcher is not None and
           self.jackhmmer_uniref90_runner is None
        ):
            raise ValueError(
                "Uniref90 runner must be specified to run template search"
            )

        self.template_searcher = template_searcher

    def run(
        self,
        fasta_path: str,
        output_dir: str,
    ):
        """Runs alignment tools on a sequence"""
        if(self.jackhmmer_uniref90_runner is not None):
            uniref90_out_path = os.path.join(output_dir, "uniref90_hits.sto")

            jackhmmer_uniref90_result = run_msa_tool(
                msa_runner=self.jackhmmer_uniref90_runner,
                fasta_path=fasta_path,
                msa_out_path=uniref90_out_path,
                msa_format='sto',
                max_sto_sequences=self.uniref_max_hits,
            )

            # template_msa = jackhmmer_uniref90_result["sto"]
            # template_msa = parsers.deduplicate_stockholm_msa(template_msa)
            # template_msa = parsers.remove_empty_columns_from_stockholm_msa(
            #     template_msa
            # )

            # if(self.template_searcher is not None):
            #     if(self.template_searcher.input_format == "sto"):
            #         pdb_templates_result = self.template_searcher.query(
            #             template_msa,
            #             output_dir=output_dir
            #         )
            #     elif(self.template_searcher.input_format == "a3m"):
            #         uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(
            #             template_msa
            #         )
            #         pdb_templates_result = self.template_searcher.query(
            #             uniref90_msa_as_a3m,
            #             output_dir=output_dir
            #         )
            #     else:
            #         fmt = self.template_searcher.input_format
            #         raise ValueError(
            #             f"Unrecognized template input format: {fmt}"
            #         )

        if(self.jackhmmer_mgnify_runner is not None):
            mgnify_out_path = os.path.join(output_dir, "mgnify_hits.sto")
            jackhmmer_mgnify_result = run_msa_tool(
                msa_runner=self.jackhmmer_mgnify_runner,
                fasta_path=fasta_path,
                msa_out_path=mgnify_out_path,
                msa_format='sto',
                max_sto_sequences=self.mgnify_max_hits
            )

        if(self.use_small_bfd and self.jackhmmer_small_bfd_runner is not None):
            bfd_out_path = os.path.join(output_dir, "small_bfd_hits.sto")
            jackhmmer_small_bfd_result = run_msa_tool(
                msa_runner=self.jackhmmer_small_bfd_runner,
                fasta_path=fasta_path,
                msa_out_path=bfd_out_path,
                msa_format="sto",
            )
        elif(self.hhblits_bfd_unirefclust_runner is not None):
            uni_name = "uni"
            for db_name in self.hhblits_bfd_unirefclust_runner.databases:
                if "uniref" in db_name.lower():
                    uni_name = f"{uni_name}ref"
                elif "uniclust" in db_name.lower():
                    uni_name = f"{uni_name}clust"

            bfd_out_path = os.path.join(output_dir, f"bfd_{uni_name}_hits.a3m")
            hhblits_bfd_unirefclust_result = run_msa_tool(
                msa_runner=self.hhblits_bfd_unirefclust_runner,
                fasta_path=fasta_path,
                msa_out_path=bfd_out_path,
                msa_format="a3m",
            )

        if(self.jackhmmer_uniprot_runner is not None):
            uniprot_out_path = os.path.join(output_dir, 'uniprot_hits.sto')
            result = run_msa_tool(
                self.jackhmmer_uniprot_runner,
                fasta_path=fasta_path,
                msa_out_path=uniprot_out_path,
                msa_format='sto',
                max_sto_sequences=self.uniprot_max_hits,
            )


@contextlib.contextmanager
def temp_fasta_file(fasta_str: str):
    with tempfile.NamedTemporaryFile('w', suffix='.fasta') as fasta_file:
      fasta_file.write(fasta_str)
      fasta_file.seek(0)
      yield fasta_file.name

def convert_monomer_features(
    monomer_features: FeatureDict,
    chain_id: str
) -> FeatureDict:
    """Reshapes and modifies monomer features for multimer models."""
    converted = {}
    converted['auth_chain_id'] = np.asarray(chain_id, dtype=object)
    unnecessary_leading_dim_feats = {
        'sequence', 'domain_name', 'num_alignments', 'seq_length'
    }
    for feature_name, feature in monomer_features.items():
      if feature_name in unnecessary_leading_dim_feats:
        # asarray ensures it's a np.ndarray.
        feature = np.asarray(feature[0], dtype=feature.dtype)
      elif feature_name == 'aatype':
        # The multimer model performs the one-hot operation itself.
        feature = np.argmax(feature, axis=-1).astype(np.int32)
      elif feature_name == 'template_aatype':
        feature = np.argmax(feature, axis=-1).astype(np.int32)
        new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
        feature = np.take(new_order_list, feature.astype(np.int32), axis=0)
      elif feature_name == 'template_all_atom_masks':
        feature_name = 'template_all_atom_mask'
      converted[feature_name] = feature
    return converted

def int_id_to_str_id(num: int) -> str:
    """Encodes a number as a string, using reverse spreadsheet style naming.

    Args:
      num: A positive integer.

    Returns:
      A string that encodes the positive integer using reverse spreadsheet style,
      naming e.g. 1 = A, 2 = B, ..., 27 = AA, 28 = BA, 29 = CA, ... This is the
      usual way to encode chain IDs in mmCIF files.
    """
    if num <= 0:
      raise ValueError(f'Only positive integers allowed, got {num}.')

    num = num - 1  # 1-based indexing.
    output = []
    while num >= 0:
      output.append(chr(num % 26 + ord('A')))
      num = num // 26 - 1
    return ''.join(output)

def add_assembly_features(
    all_chain_features: MutableMapping[str, FeatureDict],
) -> MutableMapping[str, FeatureDict]:
    """Add features to distinguish between chains.

    Args:
      all_chain_features: A dictionary which maps chain_id to a dictionary of
        features for each chain.

    Returns:
      all_chain_features: A dictionary which maps strings of the form
        `<seq_id>_<sym_id>` to the corresponding chain features. E.g. two
        chains from a homodimer would have keys A_1 and A_2. Two chains from a
        heterodimer would have keys A_1 and B_1.
    """
    # Group the chains by sequence
    seq_to_entity_id = {}
    grouped_chains = collections.defaultdict(list)
    for chain_id, chain_features in all_chain_features.items():
      seq = str(chain_features['sequence'])
      if seq not in seq_to_entity_id:
        seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
      grouped_chains[seq_to_entity_id[seq]].append(chain_features)

    new_all_chain_features = {}
    chain_id = 1
    for entity_id, group_chain_features in grouped_chains.items():
      for sym_id, chain_features in enumerate(group_chain_features, start=1):
        new_all_chain_features[
            f'{int_id_to_str_id(entity_id)}_{sym_id}'] = chain_features
        seq_length = chain_features['seq_length']
        chain_features['asym_id'] = (
            chain_id * np.ones(seq_length)
        ).astype(np.int64)
        chain_features['sym_id'] = (
            sym_id * np.ones(seq_length)
        ).astype(np.int64)
        chain_features['entity_id'] = (
            entity_id * np.ones(seq_length)
        ).astype(np.int64)
        chain_id += 1

    return new_all_chain_features

def pad_msa(np_example, min_num_seq):
    np_example = dict(np_example)
    num_seq = np_example['msa'].shape[0]
    if num_seq < min_num_seq:
      for feat in ('msa', 'deletion_matrix', 'bert_mask', 'msa_mask'):
        np_example[feat] = np.pad(
            np_example[feat], ((0, min_num_seq - num_seq), (0, 0)))
      np_example['cluster_bias_mask'] = np.pad(
          np_example['cluster_bias_mask'], ((0, min_num_seq - num_seq),))
    return np_example


class DataPipeline:
    """Assembles input features."""

    def _parse_msa_data(
        self,
        alignment_dir: str
    ) -> Mapping[str, Any]:
        msa_data = {}
        for f in os.listdir(alignment_dir):
            path = os.path.join(alignment_dir, f)
            filename, ext = os.path.splitext(f)

            if ext == ".a3m":
                with open(path, "r") as fp:
                    msa = parsers.parse_a3m(fp.read())
            elif ext == ".sto" and filename not in ["uniprot_hits", "hmm_output"]:
                with open(path, "r") as fp:
                    msa = parsers.parse_stockholm(
                        fp.read()
                    )
            else:
                continue

            msa_data[f] = msa

        return msa_data

    def _get_msas(self,
        alignment_dir: str,
        input_sequence: Optional[str] = None
    ):
        msa_data = self._parse_msa_data(alignment_dir)
        if(len(msa_data) == 0):
            if(input_sequence is None):
                raise ValueError(
                    """
                    If the alignment dir contains no MSAs, an input sequence 
                    must be provided.
                    """
                )

            msa_data["dummy"] = make_dummy_msa_obj(input_sequence)

        return list(msa_data.values())

    def _process_msa_feats(
        self,
        alignment_dir: str,
        input_sequence: Optional[str] = None
    ) -> Mapping[str, Any]:

        msas = self._get_msas(alignment_dir, input_sequence)
        msa_features = make_msa_features(msas=msas)

        return msa_features

    def process_mmcif(
        self,
        mmcif: mmcif_parsing.MmcifObject,
        alignment_dir: str,
        chain_id: Optional[str] = None
    ) -> FeatureDict:
        if chain_id is None:
            chains = mmcif.structure.get_chains()
            chain = next(chains, None)
            if chain is None:
                raise ValueError("No chains in mmCIF file")
            chain_id = chain.id

        mmcif_feats = make_mmcif_features(mmcif, chain_id)

        input_sequence = mmcif.chain_to_seqres[chain_id]
        msa_features = self._process_msa_feats(alignment_dir, input_sequence)

        return {**mmcif_feats, **msa_features}