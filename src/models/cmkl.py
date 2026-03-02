"""CMKL: Continual Multimodal Knowledge Graph Learner.

Assembles the full CMKL model from its 4 key components:
1. Modality-specific encoders (structural, textual, molecular)
2. Cross-modal attention fusion
3. Modality-aware EWC (continual learning regularization)
4. Multimodal memory replay

The core contribution: modality-aware continual learning that leverages
multimodal complementarity to reduce forgetting while handling heterogeneous
distribution shifts across modalities.

Training pipeline per task:
1. Encode: structural, textual, molecular
2. Fuse: cross-modal attention
3. Train: task loss + EWC penalty + replay loss
4. After training: compute Fisher per modality, add to replay buffer
5. Evaluate: on all tasks seen so far

Usage:
    from src.models.cmkl import CMKL
    model = CMKL(config)
    results = model.train_continually(task_sequence)
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.encoders import StructuralEncoder, TextualEncoder, MolecularEncoder
from src.models.fusion import CrossModalAttentionFusion, ConcatenationFusion
from src.models.decoders import TransEDecoder, DistMultDecoder, BilinearDecoder
from src.continual.modality_ewc import ModalityAwareEWC
from src.continual.multimodal_replay import MultimodalMemoryBuffer
from src.continual.distillation import KnowledgeDistillation
from src.baselines._base import (
    load_task_sequence,
    make_triples_factory,
    evaluate_link_prediction,
    get_device,
)

logger = logging.getLogger(__name__)

DECODER_REGISTRY = {
    "TransE": TransEDecoder,
    "DistMult": DistMultDecoder,
    "Bilinear": BilinearDecoder,
}

DEFAULT_CONFIG = {
    "embedding_dim": 256,
    "num_gnn_layers": 2,
    "num_gnn_bases": 30,
    "num_attention_heads": 4,
    "fusion_type": "cross_attention",  # or "concatenation"
    "decoder_type": "DistMult",  # TransE, DistMult, or Bilinear
    "lambda_struct": 10.0,
    "lambda_text": 5.0,
    "lambda_mol": 1.0,
    "replay_buffer_size": 1000,
    "replay_strategy": "full_multimodal",
    "replay_weight": 0.5,
    "lr": 0.001,
    "num_epochs": 50,
    "batch_size": 256,
    "dropout": 0.1,
    "margin": 1.0,
    "neg_ratio": 1,
    "use_distillation": False,
    "distillation_temperature": 2.0,
    "distillation_alpha": 0.5,
}


class CMKL(nn.Module):
    """Continual Multimodal Knowledge Graph Learner.

    Combines structural (R-GCN), textual (BiomedBERT), and molecular
    (Morgan fingerprint MLP) encoders with cross-modal attention fusion
    and modality-aware continual learning mechanisms.

    Args:
        config: Configuration dict with model hyperparameters.
            See DEFAULT_CONFIG for expected keys and defaults.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = {**DEFAULT_CONFIG, **config}
        c = self.config

        # These are set during training when we know the data dimensions
        self.num_entities = c.get("num_entities", 0)
        self.num_relations = c.get("num_relations", 0)
        self.embedding_dim = c["embedding_dim"]

        # --- Component 1: Modality-specific encoders ---
        if self.num_entities > 0 and self.num_relations > 0:
            self._init_encoders()
        else:
            # Encoders will be initialized when data dimensions are known
            self.structural_encoder = None
            self.textual_encoder = None
            self.molecular_encoder = None

        # --- Component 2: Cross-modal fusion ---
        if c["fusion_type"] == "cross_attention":
            self.fusion = CrossModalAttentionFusion(
                embed_dim=c["embedding_dim"],
                num_heads=c["num_attention_heads"],
                dropout=c["dropout"],
            )
        else:
            self.fusion = ConcatenationFusion(
                embed_dim=c["embedding_dim"],
                dropout=c["dropout"],
            )

        # --- Component 3: Decoder ---
        decoder_cls = DECODER_REGISTRY.get(c["decoder_type"], DistMultDecoder)
        if c["decoder_type"] == "Bilinear":
            self.decoder = decoder_cls(
                embedding_dim=c["embedding_dim"],
                num_relations=max(self.num_relations, 1),
            )
        else:
            self.decoder = decoder_cls(embedding_dim=c["embedding_dim"])

        # Relation embeddings for TransE/DistMult decoders
        self.relation_emb = None
        if self.num_relations > 0:
            self.relation_emb = nn.Embedding(self.num_relations, c["embedding_dim"])
            nn.init.xavier_uniform_(self.relation_emb.weight)

        # --- Component 4: Continual learning modules (not nn.Module, separate) ---
        # These are initialized lazily during training
        self.ewc: ModalityAwareEWC | None = None
        self.replay_buffer: MultimodalMemoryBuffer | None = None

        # --- Optional: Knowledge Distillation ---
        self.distillation: KnowledgeDistillation | None = None
        self._teacher_model: CMKL | None = None

        # --- Optional: Node Classification head ---
        self.nc_classifier: nn.Sequential | None = None
        if c.get("use_nc", False):
            num_classes = c.get("num_nc_classes", 10)
            self.nc_classifier = nn.Sequential(
                nn.Linear(c["embedding_dim"], c["embedding_dim"]),
                nn.ReLU(),
                nn.Dropout(c["dropout"]),
                nn.Linear(c["embedding_dim"], num_classes),
            )

    def _init_encoders(self) -> None:
        """Initialize encoders once data dimensions are known."""
        c = self.config
        self.structural_encoder = StructuralEncoder(
            num_nodes=self.num_entities,
            num_relations=self.num_relations,
            embedding_dim=c["embedding_dim"],
            num_layers=c["num_gnn_layers"],
            num_bases=c["num_gnn_bases"],
        )
        self.textual_encoder = TextualEncoder(
            projection_dim=c["embedding_dim"],
        )
        self.molecular_encoder = MolecularEncoder(
            projection_dim=c["embedding_dim"],
            dropout=c["dropout"],
        )

    def init_for_data(
        self,
        num_entities: int,
        num_relations: int,
    ) -> None:
        """Initialize model components that depend on data dimensions.

        Called before training begins once we know the KG size.

        Args:
            num_entities: Total number of entities across all tasks.
            num_relations: Total number of relation types across all tasks.
        """
        self.num_entities = num_entities
        self.num_relations = num_relations
        self._init_encoders()

        # Re-init relation embeddings
        self.relation_emb = nn.Embedding(num_relations, self.embedding_dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Re-init bilinear decoder if needed
        if self.config["decoder_type"] == "Bilinear":
            self.decoder = BilinearDecoder(
                embedding_dim=self.embedding_dim,
                num_relations=num_relations,
            )

    def encode_structural(
        self,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode all nodes through the structural R-GCN encoder.

        Args:
            edge_index: [2, num_edges] edge indices.
            edge_type: [num_edges] edge type labels.

        Returns:
            Structural embeddings [num_entities, embedding_dim].
        """
        return self.structural_encoder(edge_index, edge_type)

    def encode_textual(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Project pre-computed text embeddings through the textual encoder.

        Args:
            text_embeddings: [N_text, 768] pre-computed BiomedBERT embeddings.

        Returns:
            Projected text embeddings [N_text, embedding_dim].
        """
        return self.textual_encoder(text_embeddings)

    def encode_molecular(self, fingerprints: torch.Tensor) -> torch.Tensor:
        """Encode molecular fingerprints through the molecular encoder.

        Args:
            fingerprints: [N_mol, 1024] Morgan fingerprint vectors.

        Returns:
            Molecular embeddings [N_mol, embedding_dim].
        """
        return self.molecular_encoder(fingerprints)

    def forward(
        self,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
        text_embeddings: torch.Tensor | None = None,
        mol_fingerprints: torch.Tensor | None = None,
        node_has_text: torch.Tensor | None = None,
        node_has_mol: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full forward pass: encode all modalities and fuse.

        Args:
            edge_index: [2, E] edge indices for R-GCN.
            edge_type: [E] edge types for R-GCN.
            text_embeddings: [N, 768] pre-computed text features (zeros where missing).
            mol_fingerprints: [N, 1024] Morgan fingerprints (zeros where missing).
            node_has_text: [N] boolean mask for text availability.
            node_has_mol: [N] boolean mask for molecular availability.

        Returns:
            Fused node embeddings [N, embedding_dim].
        """
        N = self.num_entities
        D = self.embedding_dim
        device = next(self.parameters()).device

        # --- Structural encoding (always available) ---
        h_struct = self.encode_structural(edge_index, edge_type)

        # --- Textual encoding ---
        if text_embeddings is not None and node_has_text is not None:
            text_idx = node_has_text.nonzero(as_tuple=True)[0]
            if text_idx.numel() > 0:
                h_text = self.encode_textual(text_embeddings[text_idx])
            else:
                h_text = torch.zeros(0, D, device=device)
        else:
            h_text = torch.zeros(0, D, device=device)
            node_has_text = torch.zeros(N, dtype=torch.bool, device=device)

        # --- Molecular encoding ---
        if mol_fingerprints is not None and node_has_mol is not None:
            mol_idx = node_has_mol.nonzero(as_tuple=True)[0]
            if mol_idx.numel() > 0:
                h_mol = self.encode_molecular(mol_fingerprints[mol_idx])
            else:
                h_mol = torch.zeros(0, D, device=device)
        else:
            h_mol = torch.zeros(0, D, device=device)
            node_has_mol = torch.zeros(N, dtype=torch.bool, device=device)

        # --- Fusion ---
        h_fused = self.fusion(h_struct, h_text, h_mol, node_has_text, node_has_mol)

        return h_fused

    def score_triples(
        self,
        node_embeddings: torch.Tensor,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score (head, relation, tail) triples using the decoder.

        Args:
            node_embeddings: [N, D] fused node embeddings.
            head_ids: [B] head entity indices.
            relation_ids: [B] relation type indices.
            tail_ids: [B] tail entity indices.

        Returns:
            Scores [B].
        """
        h = node_embeddings[head_ids]
        t = node_embeddings[tail_ids]

        if self.config["decoder_type"] == "Bilinear":
            return self.decoder(h, relation_ids, t)
        else:
            r = self.relation_emb(relation_ids)
            return self.decoder(h, r, t)

    def classify_nodes(
        self,
        node_embeddings: torch.Tensor,
        node_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Classify nodes using the optional NC head.

        Args:
            node_embeddings: [N, D] fused node embeddings.
            node_ids: [B] indices of nodes to classify.

        Returns:
            Logits [B, num_classes].

        Raises:
            RuntimeError: If NC head is not initialized (use_nc=False).
        """
        if self.nc_classifier is None:
            raise RuntimeError("NC classifier not initialized. Set use_nc=True in config.")
        return self.nc_classifier(node_embeddings[node_ids])

    def compute_task_loss(
        self,
        node_embeddings: torch.Tensor,
        triples: torch.Tensor,
        entity_to_id: dict[str, int],
        relation_to_id: dict[str, int],
        margin: float = 1.0,
    ) -> torch.Tensor:
        """Compute link prediction loss with negative sampling.

        Args:
            node_embeddings: [N, D] fused node embeddings.
            triples: [B, 3] integer triples (mapped head, relation, tail).
            entity_to_id: Entity-to-ID mapping (for num_entities).
            relation_to_id: Relation-to-ID mapping.
            margin: Margin for ranking loss.

        Returns:
            Scalar loss tensor.
        """
        device = node_embeddings.device
        heads = triples[:, 0]
        rels = triples[:, 1]
        tails = triples[:, 2]

        # Positive scores
        pos_scores = self.score_triples(node_embeddings, heads, rels, tails)

        # Negative sampling: corrupt head or tail
        neg_triples = triples.clone()
        n = neg_triples.shape[0]
        mask = torch.rand(n, device=device) < 0.5
        random_entities = torch.randint(0, self.num_entities, (n,), device=device)
        neg_triples[mask, 0] = random_entities[mask]
        neg_triples[~mask, 2] = random_entities[~mask]

        neg_heads = neg_triples[:, 0]
        neg_rels = neg_triples[:, 1]
        neg_tails = neg_triples[:, 2]
        neg_scores = self.score_triples(node_embeddings, neg_heads, neg_rels, neg_tails)

        # Margin ranking loss
        loss = torch.nn.functional.relu(margin - pos_scores + neg_scores).mean()
        return loss

    def train_continually(
        self,
        task_sequence: OrderedDict[str, dict[str, np.ndarray]],
        entity_to_id: dict[str, int],
        relation_to_id: dict[str, int],
        device: str = "auto",
        text_embeddings: torch.Tensor | None = None,
        mol_fingerprints: torch.Tensor | None = None,
        node_has_text: torch.Tensor | None = None,
        node_has_mol: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
        seed: int = 42,
    ) -> dict:
        """Train CMKL on a sequence of tasks.

        For each task:
        1. Train with combined loss (task + EWC + replay)
        2. Compute per-modality Fisher information
        3. Add exemplars to multimodal memory buffer
        4. Evaluate on all tasks seen so far

        Args:
            task_sequence: OrderedDict of task_name -> {'train': array, 'val': array, 'test': array}.
                Arrays contain string triples (head, relation, tail).
            entity_to_id: Global entity-to-ID mapping.
            relation_to_id: Global relation-to-ID mapping.
            device: Device for training.
            text_embeddings: [N, 768] pre-computed text features.
            mol_fingerprints: [N, 1024] Morgan fingerprints.
            node_has_text: [N] boolean mask.
            node_has_mol: [N] boolean mask.
            edge_index: [2, E] edges for R-GCN.
            edge_type: [E] edge types.
            seed: Random seed.

        Returns:
            Dict with results_matrix, per-task metrics, training logs.
        """
        c = self.config
        device = get_device(device)
        torch.manual_seed(seed)

        # Initialize model for data dimensions
        if self.structural_encoder is None:
            self.init_for_data(len(entity_to_id), len(relation_to_id))
        self.to(device)

        # Initialize continual learning modules
        self.ewc = ModalityAwareEWC(
            self,
            lambda_struct=c["lambda_struct"],
            lambda_text=c["lambda_text"],
            lambda_mol=c["lambda_mol"],
        )
        self.replay_buffer = MultimodalMemoryBuffer(
            max_size=c["replay_buffer_size"],
            strategy=c["replay_strategy"],
        )

        # Initialize distillation if enabled
        if c.get("use_distillation", False):
            self.distillation = KnowledgeDistillation(
                temperature=c.get("distillation_temperature", 2.0),
                alpha=c.get("distillation_alpha", 0.5),
            )
            self._teacher_model = None
            logger.info("Knowledge distillation enabled (T=%.1f, alpha=%.2f)",
                        self.distillation.temperature, self.distillation.alpha)

        # Move multimodal features to device
        if text_embeddings is not None:
            text_embeddings = text_embeddings.to(device)
        if mol_fingerprints is not None:
            mol_fingerprints = mol_fingerprints.to(device)
        if node_has_text is not None:
            node_has_text = node_has_text.to(device)
        if node_has_mol is not None:
            node_has_mol = node_has_mol.to(device)
        if edge_index is not None:
            edge_index = edge_index.to(device)
        if edge_type is not None:
            edge_type = edge_type.to(device)

        task_names = list(task_sequence.keys())
        num_tasks = len(task_names)
        results_matrix = np.zeros((num_tasks, num_tasks))

        optimizer = torch.optim.Adam(self.parameters(), lr=c["lr"])

        for task_idx, task_name in enumerate(task_names):
            logger.info(f"=== Task {task_idx + 1}/{num_tasks}: {task_name} ===")
            task_data = task_sequence[task_name]

            # Map string triples to integer IDs
            train_triples = self._map_triples(
                task_data["train"], entity_to_id, relation_to_id
            )
            train_triples_t = torch.tensor(train_triples, dtype=torch.long, device=device)

            # Training loop
            self.train()
            for epoch in range(c["num_epochs"]):
                epoch_loss = self._train_epoch(
                    train_triples_t,
                    optimizer,
                    entity_to_id,
                    relation_to_id,
                    text_embeddings,
                    mol_fingerprints,
                    node_has_text,
                    node_has_mol,
                    edge_index,
                    edge_type,
                    device,
                )
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(f"  Epoch {epoch + 1}/{c['num_epochs']}: loss={epoch_loss:.4f}")

            # After training on this task:
            # 1. Compute per-modality Fisher
            self._compute_fisher_for_task(
                train_triples_t,
                entity_to_id,
                relation_to_id,
                text_embeddings,
                mol_fingerprints,
                node_has_text,
                node_has_mol,
                edge_index,
                edge_type,
                device,
            )

            # 2. Add exemplars to replay buffer
            self.eval()
            with torch.no_grad():
                h_fused = self.forward(
                    edge_index, edge_type,
                    text_embeddings, mol_fingerprints,
                    node_has_text, node_has_mol,
                )
            # Select a subset of training triples for the buffer
            n_exemplars = min(len(train_triples), c["replay_buffer_size"] // num_tasks)
            indices = np.random.choice(len(train_triples), n_exemplars, replace=False)
            exemplar_triples = train_triples[indices]
            self.replay_buffer.add_exemplars(
                exemplar_triples,
                h_fused,
                text_embeddings,
                mol_fingerprints,
                node_has_text,
                node_has_mol,
                task_id=task_idx,
            )

            # 3. Create teacher copy for distillation on the next task
            if self.distillation is not None:
                self._teacher_model = KnowledgeDistillation.create_teacher_copy(self)
                self._teacher_model.to(device)

            # 4. Evaluate on all tasks seen so far
            self.eval()
            for eval_idx in range(task_idx + 1):
                eval_name = task_names[eval_idx]
                eval_data = task_sequence[eval_name]

                # Use PyKEEN evaluator for comparable metrics
                test_factory = make_triples_factory(
                    eval_data["test"], entity_to_id, relation_to_id,
                )
                # We need a PyKEEN-compatible evaluation, but our model isn't PyKEEN.
                # Instead, compute MRR manually.
                test_mrr = self._evaluate_mrr(
                    eval_data["test"],
                    entity_to_id,
                    relation_to_id,
                    text_embeddings,
                    mol_fingerprints,
                    node_has_text,
                    node_has_mol,
                    edge_index,
                    edge_type,
                    device,
                )
                results_matrix[task_idx, eval_idx] = test_mrr
                logger.info(f"  Eval {eval_name}: MRR={test_mrr:.4f}")

        return {
            "results_matrix": results_matrix.tolist(),
            "task_names": task_names,
            "config": c,
            "seed": seed,
        }

    def _map_triples(
        self,
        triples: np.ndarray,
        entity_to_id: dict[str, int],
        relation_to_id: dict[str, int],
    ) -> np.ndarray:
        """Map string triples to integer IDs.

        Args:
            triples: [N, 3] string triples.
            entity_to_id: Entity mapping.
            relation_to_id: Relation mapping.

        Returns:
            [N, 3] integer triples.
        """
        mapped = np.zeros((len(triples), 3), dtype=np.int64)
        for i, (h, r, t) in enumerate(triples):
            mapped[i] = [entity_to_id.get(h, 0), relation_to_id.get(r, 0), entity_to_id.get(t, 0)]
        return mapped

    def _train_epoch(
        self,
        train_triples: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        entity_to_id: dict,
        relation_to_id: dict,
        text_embeddings: torch.Tensor | None,
        mol_fingerprints: torch.Tensor | None,
        node_has_text: torch.Tensor | None,
        node_has_mol: torch.Tensor | None,
        edge_index: torch.Tensor | None,
        edge_type: torch.Tensor | None,
        device: str,
    ) -> float:
        """Run one training epoch with combined loss.

        Loss = task_loss + ewc_penalty + replay_loss
        """
        c = self.config
        self.train()

        n = train_triples.shape[0]
        perm = torch.randperm(n, device=device)
        train_triples = train_triples[perm]

        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, c["batch_size"]):
            batch = train_triples[start:start + c["batch_size"]]

            # Forward pass: encode + fuse
            h_fused = self.forward(
                edge_index, edge_type,
                text_embeddings, mol_fingerprints,
                node_has_text, node_has_mol,
            )

            # Task loss
            loss = self.compute_task_loss(
                h_fused, batch, entity_to_id, relation_to_id,
                margin=c["margin"],
            )

            # EWC penalty
            if self.ewc is not None:
                ewc_penalty = self.ewc.ewc_loss()
                loss = loss + ewc_penalty

            # Distillation loss
            if (self.distillation is not None
                    and self._teacher_model is not None):
                with torch.no_grad():
                    teacher_h = self._teacher_model.forward(
                        edge_index, edge_type,
                        text_embeddings, mol_fingerprints,
                        node_has_text, node_has_mol,
                    )
                # Compute scores for batch triples over all entities
                heads = batch[:, 0]
                rels = batch[:, 1]
                # Student all-entity scores
                if self.config["decoder_type"] == "Bilinear":
                    s_h = h_fused[heads]
                    s_M = self.decoder.relation_matrices[rels]
                    s_hM = torch.einsum("bi,bij->bj", s_h, s_M)
                    student_scores = s_hM @ h_fused.T
                elif self.config["decoder_type"] == "TransE":
                    s_query = h_fused[heads] + self.relation_emb(rels)
                    student_scores = -torch.cdist(
                        s_query, h_fused, p=self.decoder.p_norm)
                else:  # DistMult
                    s_query = h_fused[heads] * self.relation_emb(rels)
                    student_scores = s_query @ h_fused.T

                # Teacher all-entity scores (same formulation)
                with torch.no_grad():
                    if self.config["decoder_type"] == "Bilinear":
                        t_h = teacher_h[heads]
                        t_M = self._teacher_model.decoder.relation_matrices[rels]
                        t_hM = torch.einsum("bi,bij->bj", t_h, t_M)
                        teacher_scores = t_hM @ teacher_h.T
                    elif self.config["decoder_type"] == "TransE":
                        t_query = teacher_h[heads] + self._teacher_model.relation_emb(rels)
                        teacher_scores = -torch.cdist(
                            t_query, teacher_h, p=self._teacher_model.decoder.p_norm)
                    else:  # DistMult
                        t_query = teacher_h[heads] * self._teacher_model.relation_emb(rels)
                        teacher_scores = t_query @ teacher_h.T

                loss = self.distillation.compute_combined_loss(
                    loss, student_scores, teacher_scores)

            # Replay loss
            if self.replay_buffer is not None and len(self.replay_buffer) > 0:
                replay_triples = self.replay_buffer.get_replay_triples(
                    min(c["batch_size"], len(self.replay_buffer))
                )
                if replay_triples is not None:
                    replay_t = torch.tensor(replay_triples, dtype=torch.long, device=device)
                    replay_loss = self.compute_task_loss(
                        h_fused, replay_t, entity_to_id, relation_to_id,
                        margin=c["margin"],
                    )
                    loss = loss + c["replay_weight"] * replay_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _compute_fisher_for_task(
        self,
        train_triples: torch.Tensor,
        entity_to_id: dict,
        relation_to_id: dict,
        text_embeddings: torch.Tensor | None,
        mol_fingerprints: torch.Tensor | None,
        node_has_text: torch.Tensor | None,
        node_has_mol: torch.Tensor | None,
        edge_index: torch.Tensor | None,
        edge_type: torch.Tensor | None,
        device: str,
    ) -> None:
        """Compute per-modality Fisher after finishing a task."""
        c = self.config

        # Create a simple dataloader from training triples
        dataset = torch.utils.data.TensorDataset(train_triples)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=c["batch_size"], shuffle=True,
        )

        def compute_loss_fn(batch_tuple):
            triples_batch = batch_tuple[0] if isinstance(batch_tuple, (tuple, list)) else batch_tuple
            h_fused = self.forward(
                edge_index, edge_type,
                text_embeddings, mol_fingerprints,
                node_has_text, node_has_mol,
            )
            return self.compute_task_loss(
                h_fused, triples_batch, entity_to_id, relation_to_id,
            )

        self.ewc.compute_modality_fisher(
            compute_loss_fn=compute_loss_fn,
            dataloader=dataloader,
            device=device,
            num_samples=200,
        )

    @torch.no_grad()
    def _evaluate_mrr(
        self,
        test_triples: np.ndarray,
        entity_to_id: dict[str, int],
        relation_to_id: dict[str, int],
        text_embeddings: torch.Tensor | None,
        mol_fingerprints: torch.Tensor | None,
        node_has_text: torch.Tensor | None,
        node_has_mol: torch.Tensor | None,
        edge_index: torch.Tensor | None,
        edge_type: torch.Tensor | None,
        device: str,
        batch_size: int = 256,
    ) -> float:
        """Evaluate MRR on test triples.

        For each test triple (h, r, t):
        - Score all entities as potential tails: score(h, r, e) for all e
        - Rank the true tail among all entities
        - MRR = mean(1/rank)

        Args:
            test_triples: [N, 3] string triples.
            entity_to_id: Entity mapping.
            relation_to_id: Relation mapping.
            text_embeddings, mol_fingerprints, node_has_text, node_has_mol,
            edge_index, edge_type: Multimodal features.
            device: Device.
            batch_size: Evaluation batch size.

        Returns:
            MRR score.
        """
        self.eval()
        mapped = self._map_triples(test_triples, entity_to_id, relation_to_id)
        if len(mapped) == 0:
            return 0.0

        # Get fused embeddings
        h_fused = self.forward(
            edge_index, edge_type,
            text_embeddings, mol_fingerprints,
            node_has_text, node_has_mol,
        )

        ranks = []
        mapped_t = torch.tensor(mapped, dtype=torch.long, device=device)

        for start in range(0, len(mapped_t), batch_size):
            batch = mapped_t[start:start + batch_size]
            heads = batch[:, 0]
            rels = batch[:, 1]
            tails = batch[:, 2]
            B = heads.shape[0]

            # Score all entities as tails for each (h, r) pair
            h_embs = h_fused[heads]  # [B, D]
            if self.config["decoder_type"] == "Bilinear":
                # For bilinear: h^T M_r t for all t
                M = self.decoder.relation_matrices[rels]  # [B, D, D]
                h_M = torch.einsum("bi,bij->bj", h_embs, M)  # [B, D]
                all_scores = h_M @ h_fused.T  # [B, N]
            else:
                r_embs = self.relation_emb(rels)  # [B, D]
                if self.config["decoder_type"] == "TransE":
                    # TransE: -||h + r - t|| for all t
                    query = h_embs + r_embs  # [B, D]
                    all_scores = -torch.cdist(query, h_fused, p=self.decoder.p_norm)  # [B, N]
                else:
                    # DistMult: (h * r) . t for all t
                    query = h_embs * r_embs  # [B, D]
                    all_scores = query @ h_fused.T  # [B, N]

            # Get rank of true tail
            true_scores = all_scores[torch.arange(B, device=device), tails]  # [B]
            # Count how many entities score >= true tail (1-based rank)
            batch_ranks = (all_scores >= true_scores.unsqueeze(1)).sum(dim=1).float()
            ranks.extend(batch_ranks.cpu().tolist())

        if not ranks:
            return 0.0

        mrr = float(np.mean([1.0 / r for r in ranks]))
        return mrr

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint including CL state."""
        state = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
        }
        if self.ewc is not None:
            state["ewc_state"] = self.ewc.state_dict()
        if self.replay_buffer is not None:
            state["replay_state"] = self.replay_buffer.state_dict()
        torch.save(state, str(path))
        logger.info(f"Saved checkpoint to {path}")

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str = "cpu") -> CMKL:
        """Load model from checkpoint."""
        state = torch.load(str(path), map_location=device)
        config = state["config"]
        config["num_entities"] = state["num_entities"]
        config["num_relations"] = state["num_relations"]
        model = cls(config)
        model.init_for_data(state["num_entities"], state["num_relations"])
        model.load_state_dict(state["model_state_dict"])
        if "ewc_state" in state:
            model.ewc = ModalityAwareEWC(model)
            model.ewc.load_state_dict(state["ewc_state"])
        if "replay_state" in state:
            model.replay_buffer = MultimodalMemoryBuffer()
            model.replay_buffer.load_state_dict(state["replay_state"])
        return model.to(device)
