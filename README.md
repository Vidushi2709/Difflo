# 🌸 Difflo — Diffusion Models for Flower Image Generation

**Difflo** is a lightweight project exploring denoising diffusion models for generating realistic flower images from pure noise. The goal is to understand how structure, color, and fine texture emerge through iterative denoising, using a compact and interpretable setup.

---

## 🧠 Overview

Diffusion models learn to reverse a gradual noising process. Starting from Gaussian noise, the model iteratively reconstructs samples from the target image distribution.

In this project, a diffusion model is trained on a flower image dataset current dataset, learning to generate visually coherent flowers by progressively removing noise. The emphasis is on clarity, experimentation, and reproducibility rather than scale.

---

## 📁 Project Structure

```
Difflo/
├── diffusion-model.ipynb              # Base diffusion training and sampling pipeline
├── diffusion-model-conditional.ipynb  # Conditional diffusion variant (101 oxford flower dataset)
├── diffusion-model-attn-unet-try1.ipynb # Attention-UNet experiments
├── ddm_folders/
│   └── checkpoint/
│       └── checkpoint.weights.h5      # Saved model weights (of diffusion-model.iynb)
├── output/                             # Generated image samples (of diffusion-model.ipynb)
└── README.md
```

---

## 📊 Results 

* **Dataset:** Flowers 
* **Model:** Denoising Diffusion Model (DDM)
* **Evaluation:** Frechet Inception Distance (FID)
* **FID Score:** **6.55** 

The model captures petal-level structure, color gradients, and overall flower composition effectively given its relatively small size and training budget.

---

## 🌼 Why “Difflo”?

*Difflo* comes from **Diffusion + Flowers**. It reflects the slow, iterative, and surprisingly calm process by which diffusion models transform noise into structured visual patterns — much like a flower gradually blooming.

---

## 🔬 Notes

* This project is intended for learning and experimentation.
* The notebooks are self-contained and designed to be easy to modify.
* Further improvements may include better schedulers, classifier-free guidance, and larger backbones.

---

## 📌 Acknowledgements

Inspired by foundational work on denoising diffusion probabilistic models (DDPMs) and open-source diffusion implementations.
