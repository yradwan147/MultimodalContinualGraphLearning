"""Multimodal feature extraction for PrimeKG nodes.

Extracts three types of features for knowledge graph nodes:
1. Textual: Clinical descriptions from DrugBank (drugs) and Mayo Clinic/Orphanet (diseases)
2. Molecular: Morgan fingerprints from SMILES strings (drug nodes only)
3. Structural: Derived from graph topology via GNN encoders (computed at training time)

Drug features include: description, indication, pharmacodynamics, mechanism of action,
toxicity, protein binding, metabolism, half-life, route of elimination.
Disease features include: MONDO definition, UMLS description, Mayo Clinic clinical features.

Usage:
    from src.data.features import extract_multimodal_features
    drug_feat, disease_feat = extract_multimodal_features(kg_path, drug_path, disease_path)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def extract_multimodal_features(
    kg_path: str,
    drug_features_path: str,
    disease_features_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract text and structural features for multimodal nodes.

    Args:
        kg_path: Path to PrimeKG CSV.
        drug_features_path: Path to drug features CSV from PrimeKG.
        disease_features_path: Path to disease features CSV from PrimeKG.

    Returns:
        Tuple of (drug_features_df, disease_features_df).
    """
    raise NotImplementedError("Phase 2: Implement multimodal feature extraction")


def compute_morgan_fingerprints(
    smiles_series: "pd.Series",
    radius: int = 2,
    n_bits: int = 1024,
) -> "np.ndarray":
    """Compute Morgan fingerprints from SMILES strings using RDKit.

    Args:
        smiles_series: Pandas Series of SMILES strings.
        radius: Morgan fingerprint radius.
        n_bits: Number of bits in the fingerprint.

    Returns:
        NumPy array of shape (n_molecules, n_bits).
    """
    raise NotImplementedError("Phase 2: Implement Morgan fingerprint computation")


def compute_text_embeddings(
    texts: list[str],
    model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    batch_size: int = 32,
) -> "torch.Tensor":
    """Compute text embeddings using a pretrained language model.

    Args:
        texts: List of text descriptions.
        model_name: HuggingFace model identifier.
        batch_size: Batch size for encoding.

    Returns:
        Tensor of shape (n_texts, hidden_dim).
    """
    raise NotImplementedError("Phase 2: Implement text embedding computation")


def get_node_modality_masks(
    kg: pd.DataFrame,
    drug_features: pd.DataFrame,
    disease_features: pd.DataFrame,
) -> dict[str, "np.ndarray"]:
    """Determine which nodes have which modality features.

    Not all nodes have all modalities: only drugs have molecular features,
    only drugs/diseases have textual descriptions.

    Args:
        kg: PrimeKG DataFrame.
        drug_features: Drug features DataFrame.
        disease_features: Disease features DataFrame.

    Returns:
        Dict with 'has_text' and 'has_mol' boolean arrays indexed by node ID.
    """
    raise NotImplementedError("Phase 2: Implement modality mask computation")
