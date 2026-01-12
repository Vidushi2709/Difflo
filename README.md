# 🌸 Difflo — Class-Conditional Diffusion for Flower Image Generation

**Difflo** is a learning-focused implementation of a **class-conditional denoising diffusion model**, built from scratch to study how diffusion models behave under **low-compute constraints**.

The project emphasizes architectural correctness, conditioning mechanisms, and failure-mode analysis rather than state-of-the-art image quality.

---

## 🧠 Overview

Diffusion models learn to generate data by reversing a gradual noising process. Starting from Gaussian noise, the model iteratively denoises samples to recover structure from the learned data distribution.

In this project, a **UNet-based class-conditional diffusion model** is trained to generate **102 flower species**, with experiments around:

* noise schedules
* class conditioning
* EMA stabilization
* DDIM sampling speed–quality trade-offs

The focus is on **understanding behavior and limitations**, not maximizing benchmark scores.

---

## 📁 Project Structure

```
Difflo/
├── diffusion_model_conditional.ipynb   # Class-conditional DDPM + DDIM
├── report.md                           # Detailed technical analysis
├── figures/                            # Plots & generation samples
└── README.md
```

---

## 📊 Experimental Setup

* **Dataset**: Oxford Flowers-102
* **Resolution**: 64×64
* **Classes**: 102
* **Model**: UNet with self-attention + class embeddings
* **Training**: Single Kaggle P100 GPU
* **Sampling**: DDIM (20–200 steps)
* **Evaluation**: Fréchet Inception Distance (FID)

---

## 📉 Results & Interpretation

* **FID**: High (≈200 range under current settings)

This is expected given:

* limited training duration (≈150 epochs)
* low image resolution
* per-class data scarcity
* constrained model capacity

Rather than optimizing FID, the project analyzes **why quality degrades** under these constraints and how architectural choices affect outcomes.

Generated samples show recognizable flower structure and class-specific traits but lack fine-grained detail.

---

## 🔬 What This Project Explores

* Class conditioning via learned embeddings
* Noise schedule comparisons (linear vs cosine variants)
* EMA effects on sampling stability
* DDIM vs DDPM efficiency
* Failure modes in low-compute diffusion training

A full technical breakdown is available in `report.md`.

---

## 🌼 Why “Difflo”?

**Difflo = Diffusion + Flowers** — a small, focused environment to observe how diffusion models gradually transform noise into structured images, step by step.

---

## 📌 Notes

* This is a **learning and analysis project**, not a production model.
* Results reflect compute and data constraints, not architectural errors.
* The implementation is intended to be readable, modifiable, and reproducible.

---

## 📚 References

Based on foundational work including:

* Ho et al., *DDPM* (NeurIPS 2020)
* Song et al., *DDIM* (ICLR 2021)
* Nichol & Dhariwal, *Improved DDPM* (ICML 2021)

---