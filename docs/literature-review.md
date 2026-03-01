# Literature Review: Multimodal Continual Graph Learning

**Status:** To be completed in Phase 1
**Target:** 30+ annotated papers across 8 categories

---

## 1. Biomedical Knowledge Graphs

### PrimeKG
- **Citation:** Chandak et al., 2023
- **Key:** 129K nodes, 8.1M edges, 10 node types, 30 edge types. Integrates 20 biomedical resources.
- **Relevance:** Primary KG for this project.

### PrimeKG-U
- **Citation:** CellAwareGNN team, Feb 2026
- **Key:** Expanded to ~140K nodes, ~14M edges.
- **Relevance:** Potential t2 snapshot.

### TarKG
- **Citation:** Zhou et al., 2024
- **Key:** 15 entity types, 43 relation types, incorporates chemical structures.
- **Relevance:** Alternative biomedical KG for comparison.

---

## 2. Models Built on PrimeKG

### TxGNN
- **Citation:** Huang et al., Nature Medicine, 2024
- **Key:** GNN for zero-shot drug repurposing. 49.2% higher accuracy on indication tasks.
- **Relevance:** Evaluation protocol reference for drug-disease LP.

### ProCyon
- **Citation:** Queen et al., 2025
- **Key:** Multimodal foundation model for protein phenotypes.
- **Relevance:** Multimodal approach on PrimeKG.

---

## 3. Continual Knowledge Graph Embedding (CKGE)

### EWC for KGE
- **Citation:** 2024
- **Key:** Fisher Information Matrix penalty. Reduced forgetting from 12.62% to 6.85% on FB15k-237.
- **Relevance:** Baseline 3 in our project.

### LKGE
- **Citation:** AAAI 2023
- **Key:** Masked KG autoencoder + embedding transfer + regularization.
- **Relevance:** Baseline 5, open-source framework.

### BER
- **Citation:** 2024
- **Key:** Attention-based representative fact selection for replay + distillation.
- **Relevance:** Inspiration for Baseline 4.

*(More papers to be added in Phase 1)*

---

## 4. Multimodal Continual Graph Learning

### MSCGL
- **Citation:** Cai et al., WWW 2022
- **Key:** Seminal work. AdaMGNN with NAS + Group Sparse Regularization.
- **Relevance:** Only existing MCGL method. No official code.

### MoDE
- **Citation:** NeurIPS 2025
- **Key:** Addresses intra-modal AND inter-modal forgetting.
- **Relevance:** Defines the inter-modal forgetting challenge we address.

---

## 5. RAG-Based Biomedical Systems

### AMG-RAG
- **Citation:** EMNLP Findings 2025
- **Key:** Agentic Medical Graph-RAG. 74.1% F1 on MEDQA.
- **Relevance:** Baseline 6 inspiration.

---

## 6. Benchmarks for CKGE

### PS-CKGE
- **Citation:** SIGIR 2025
- **Key:** Benchmarks for CKGE under pattern shifts.
- **Relevance:** Methodology comparison for our benchmark design.

---

## 7. Evaluation Metrics

*(To be expanded: AP, AF, BWT, FWT, REM definitions and references)*

---

## 8. Gap Analysis

*(To be written: What's missing in the literature that our project addresses)*
