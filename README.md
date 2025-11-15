# 🌸 Difflo — Diffusion Model on Flowers

**Difflo** is a lightweight denoising diffusion model trained to generate realistic flower images from pure noise.  
It’s a small exploration into how structure, color, and texture can emerge from randomness through iterative denoising.

---

### 🧠 Overview
Diffusion models gradually learn to reverse a noising process — starting from random Gaussian noise and reconstructing the original image distribution step by step.  
This notebook trains such a model on a **flower dataset**, letting it *learn how to bloom* from the noise itself 🌷  

---

### 📁 Project Structure
- `diffusion-model.ipynb` — complete training and sampling workflow  
- `ddm_folders/checkpoint/` — saved model weights and configuration files  
- `output/` — generated flower samples  

---

### 📊 Results
- **Dataset:** Flowers  
- **Model:** Denoising Diffusion Model (DDM)  
- **FID Score:** 6.55  
- **Observation:** The model captures petal patterns and color gradients surprisingly well for its size.  

---

### 🌼 Why “Difflo”?
Because it’s *Diffusion + Flowers* and because something about it feels calm, patient, and a little bit poetic (hehe)
