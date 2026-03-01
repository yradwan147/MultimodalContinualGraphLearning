"""Baseline 6: RAG-Based Biomedical Agent for continual KGQA.

Uses retrieval-augmented generation over an evolving biomedical knowledge graph.
Key insight: By updating the retrieval index (not model weights), RAG naturally
avoids catastrophic forgetting. New knowledge is added to the vector store
without modifying the LLM. However, may suffer from retrieval drift.

Usage:
    from src.baselines.rag_agent import BiomedicalRAGAgent
    agent = BiomedicalRAGAgent()
    agent.index_kg_snapshot(triples, features)
    answer = agent.answer_question("What drugs treat disease X?")
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BiomedicalRAGAgent:
    """RAG-based agent for continual biomedical KGQA.

    Args:
        llm_name: LLM model identifier (local or API).
        embedding_model: Sentence embedding model for retrieval.
        persist_dir: Directory for ChromaDB persistence.
    """

    def __init__(
        self,
        llm_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        persist_dir: Optional[str] = None,
    ) -> None:
        self.llm_name = llm_name
        self.embedding_model_name = embedding_model
        self.persist_dir = persist_dir
        self.vectorstore = None
        self.llm = None

    def index_kg_snapshot(
        self,
        kg_triples: list[tuple],
        node_features: dict[str, str],
    ) -> int:
        """Index a KG snapshot into the vector store.

        Each triple is converted to natural language and indexed.
        Node descriptions (text features) are also indexed.

        Args:
            kg_triples: List of (head, relation, tail) triples.
            node_features: Dict mapping node_id -> text description.

        Returns:
            Number of documents indexed.
        """
        raise NotImplementedError("Phase 3: Implement KG indexing")

    def update_with_new_knowledge(
        self,
        new_triples: list[tuple],
        new_node_features: dict[str, str],
    ) -> int:
        """Continual update: add new knowledge to vector store.

        No retraining needed - just add new documents to the index.
        This is the key advantage over parametric CL methods.

        Args:
            new_triples: New triples to add.
            new_node_features: New node descriptions to add.

        Returns:
            Number of new documents added.
        """
        raise NotImplementedError("Phase 3: Implement incremental indexing")

    def answer_question(
        self,
        question: str,
        k: int = 10,
    ) -> str:
        """Answer a biomedical question using RAG over the KG.

        1. Retrieve relevant triples/descriptions from vector store.
        2. Feed retrieved context + question to LLM.
        3. Return answer.

        Args:
            question: Natural language question.
            k: Number of documents to retrieve.

        Returns:
            Generated answer string.
        """
        raise NotImplementedError("Phase 3: Implement RAG question answering")
