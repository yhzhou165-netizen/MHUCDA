# MHUCDA

**MHUCDA** is a deep learning model for miRNA-disease association prediction, built on multi-view hypergraph encoding, dual-branch collaborative learning, and uncertainty-aware adaptive fusion.

---

## Architecture

The model consists of three core modules:

- **Multi-view Hypergraph Encoder** — Encodes miRNA and disease nodes from three similarity views (functional/sequence/Gaussian for miRNA; semantic/target/Gaussian for disease) using hypergraph convolution layers, with cross-view InfoNCE contrastive loss to align representations.
- **Dual-Branch Network** — Two parallel branches capture complementary signals:
  - *MHM branch*: meta-path-based GIN convolution + hierarchical semantic attention + cooperative cross-attention between miRNA and disease nodes.
  - *HHN branch*: GCN on the bipartite miRNA-disease graph + local subgraph modeling with virtual nodes and deep ResNet GCN.
- **Adaptive Gated Fusion** — Per-sample gate network dynamically weights the two branch predictions. An uncertainty estimator calibrates final scores toward 0.5 when the branches disagree strongly.

---

## Requirements

```
Python >= 3.8
PyTorch >= 1.12
DGL
torch-geometric
scikit-learn
pandas
numpy
matplotlib
scipy
```

---

## Dataset

Place the following files under `./datasets/`:

```
Association Matrixs.csv
integrated_circ_sim_integrated_circ_sim_1_0_.csv
CFS_CFS_1_0_.csv
circ_gipk_circ_gipk_1_0_.csv
integrated_dise_sim_integrated_dise_sim_1_0_.csv
MeSHSemanticSimilarity_MeSHSemanticSimilarity_1_0_.csv
dis_gipk_dis_gipk_1_0_.csv
```

The dataset covers **561 miRNAs** and **100 diseases**.

---

## Usage

```bash
python main.py
```

Results and model checkpoints are saved to `./savemodel/MHUCDA/`. ROC and PR curves are saved under `figures/` in each run folder.

---

## Configuration

Key hyperparameters are defined in the `Config` class in `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `kfold` | 5 | Number of cross-validation folds |
| `epoch` | 250 | Max training epochs |
| `patience` | 150 | Early stopping patience |
| `lr` | 0.0007 | Learning rate |
| `fm` / `fd` | 128 | miRNA / disease embedding dimension |
| `batchSize` | 128 | Batch size |
| `cl_temperature` | 0.40 | Contrastive loss temperature |
| `swa_start` | 180 | Epoch to start SWA weight averaging |

---

## File Structure

```
├── main.py                  # Entry point and config
├── model.py                 # MHUCDA model assembly
├── train.py                 # Training loop, evaluation, plotting
├── dual_branch.py           # MHM and HHN branch implementations
├── hypergraph_encoder.py    # Multi-view hypergraph encoder
├── adaptive_fusion.py       # Adaptive gated fusion module
├── basicModules.py          # Shared base layers
├── datapro.py               # Data loading and graph construction
└── datasets/                # Input data directory
```