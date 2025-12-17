# Characterizing Graph Datasets Beyond Homophily

**Understanding GNN Accuracy Through Label Informativeness**

**Authors**

- Atharva Shetwe (24b4537)
    
- Arhan Khade (24b4532)
    
- Vaibhav Negi (24b4503)
    

---

## 📌 Overview

This project revisits the widely held assumption that **homophily** is the primary determinant of Graph Neural Network (GNN) performance. Focusing on the **Graph Convolutional Network (GCN)** and using **semi-synthetic graph datasets derived from Cora**, we demonstrate that:

> **Homophily alone is neither sufficient nor reliable for predicting GNN accuracy.**

Instead, we show that **Label Informativeness (LI)** — the amount of predictive information a node’s neighbors provide about its label — is a far stronger and more consistent predictor of GCN performance.

Our results align with and empirically validate findings from prior work by **Platonov et al. (2022)**.

---

## 🧠 Key Contributions

- **Critique of standard homophily metrics**  
    Demonstrates why edge, node, and class homophily are unreliable due to sensitivity to class count and imbalance.
    
- **Use of adjusted homophily (`h_adj`)**  
    Adopts a normalized, comparable homophily measure that satisfies the constant-baseline property.
    
- **Empirical validation of Label Informativeness (LI)**  
    Shows that GCN accuracy correlates strongly with LI and weakly with homophily.
    
- **Controlled semi-synthetic experimentation**  
    Uses stochastic block models (SBMs) with fixed node features and labels to isolate structural effects.
    

---

## 🧪 Experimental Setup

### Dataset

- **Cora citation dataset**
    
- Restricted to the **4 largest classes**
    
- Node features and labels are preserved
    

### Graph Construction

- Synthetic graphs generated using a **Stochastic Block Model (SBM)**
    
- 4 blocks × 250 nodes = **1000 nodes**
    
- Edge probabilities controlled by:
    
    - `p0`: intra-class (homophily)
        
    - `p1`: structured inter-class (informative heterophily)
        
    - `p2`: random inter-class (uninformative heterophily)
        

### Configurations Tested

|Configuration|Homophily|LI|Description|
|---|---|---|---|
|Config 1|High|High|Strong homophily + structured neighbors|
|Config 2|High|Low|Homophily without informative neighbors|
|Config 3|Low|High|Predictive heterophily|
|Config 4|Low|Low|Random, uninformative structure|

---

## 📊 Results

|Configuration|h_adj|LI|GCN Test Accuracy|
|---|---|---|---|
|High H, High LI|0.8667|0.7655|**0.9960**|
|High H, Low LI|0.7333|0.5390|0.9920|
|Low H, High LI|-0.3333|0.7155|**0.7040**|
|Low H, Low LI|-0.3333|0.5000|**0.5440**|

### Key Takeaways

- GCNs can perform **well on heterophilous graphs** if LI is high
    
- GCNs can perform **poorly on homophilous graphs** if LI is low
    
- **LI explains learnability; homophily only explains similarity**
    

---

## 🧩 Code Structure

The main experiment is contained in a single notebook/script:

- **`finalized_IE211_GNN.ipynb`**
    
    - Loads Cora dataset
        
    - Identifies top 4 classes
        
    - Generates SBM graphs
        
    - Computes adjusted homophily & LI
        
    - Trains a GCN using PyTorch Geometric
        
    - Logs accuracy and saves results
        

Results are serialized to:

```bash
sbm_gcn_results.pkl
```

---

## ⚙️ Requirements

Install the following dependencies:

```bash
pip install numpy scipy networkx scikit-learn matplotlib torch torch-geometric
```

> ⚠️ PyTorch Geometric requires additional setup depending on your CUDA / PyTorch version.  
> Refer to: [https://pytorch-geometric.readthedocs.io/](https://pytorch-geometric.readthedocs.io/)

---

## ▶️ How to Run

1. Clone the repository
    
2. Open the notebook or run the script in Google Colab / local environment
    
3. Execute all cells
    

The script will automatically:

- Download Cora dataset
    
- Generate SBM graphs
    
- Train GCNs for all configurations
    
- Print results and save them to disk
    

---

## 📚 Reference

Platonov, O., Kuznedelev, D., Babenko, A., & Prokhorenkova, L. (2022).  
**Characterizing graph datasets for node classification: Homophily–Heterophily dichotomy and beyond.**  
arXiv:2209.06177  
[https://doi.org/10.48550/arxiv.2209.06177](https://doi.org/10.48550/arxiv.2209.06177)

---

## 👥 Work Distribution

- **Arhan Khade**  
    Background research, theory, and experimental design
    
- **Atharva Shetwe**  
    GCN implementation and experimental execution
    
- **Vaibhav Negi**  
    Result analysis, interpretation, and documentation
    

---

## 🏁 Conclusion

This project demonstrates that **Label Informativeness is the true structural driver of GCN performance**, not homophily.  
Homophily explains _who is similar_ — LI explains _who is learnable_.

---
