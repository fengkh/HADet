# BA<sup>3</sup>-Det

<p align="left">
  <img src="https://img.shields.io/badge/Status-Under_Review-orange.svg" alt="Status">
  <img src="https://img.shields.io/badge/Code-Architecture_Release-blue.svg" alt="Code">
</p>

> **Notice:** This repository serves as the supplementary code base for the manuscript **"BA<sup>3</sup>-Det: A Boundary-Aware and Anti-Dilution Attention Network for Microvascular Invasion Detection"**, which is currently under peer review. 

## 📌 Repository Status
To facilitate the peer-review process and ensure algorithmic reproducibility, we have released the complete PyTorch implementation of the **BA<sup>3</sup>-Det** network architecture. 

Due to double-blind/single-blind review policies, the training pipelines and pre-trained model weights will be fully open-sourced upon the formal acceptance of the paper.

## 📂 Core Architecture
The following structural files are provided for methodological verification. The codebase is highly modularized and strictly corresponds to the mathematical formulations and topological diagrams described in the Method section of the manuscript.

### 1. Attention Modules
* `fca_block.py`: Implementation of the Fourier-based Channel Attention (FCA) block.
* `csa_block.py`: Implementation of the Cell-level Spatial Attention (CSA) block.

### 2. Network Topology
* `basic_modules.py`: Foundational structural components (`CBS`, `GSBN`, and `Focus`).
* `backbone.py`: The hierarchical feature extraction backbone integrating FCA and CSA blocks.
* `neck.py`: The PAFPN-based feature fusion neck network.
* `BA3Det.py`: The final model encapsulation, including the specific Decoupled Head implementation.

## 📊 Dataset Access
The **L-MVI (Liver-Microvascular Invasion)** dataset, comprising 552 rigorously curated and expert-annotated histopathology images, is complete and ready for evaluation.

**Data Sharing Policy:** Due to institutional regulations regarding sensitive medical data sharing, the dataset cannot be directly hosted on a public open-source platform. However, it is fully accessible for academic and peer-review purposes. 

If you require access to the dataset for verification or research, please contact me directly at: **fengkh.scu@gmail.com**.

## ⚙️ Upcoming Updates (Upon Acceptance)
- [x] Release core attention blocks (`FCA` and `CSA`).
- [x] Release full backbone, neck, and decoupled head integration.
- [ ] Release `train.py`, `val.py`, and data loaders.
- [ ] Release pre-trained model weights.