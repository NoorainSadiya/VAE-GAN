# Variational Autoencoder (VAE) & Deep Convolutional GAN (DCGAN)
### Generative Deep Learning for Synthetic Medical Image Generation using PyTorch

---

## 🧠 Overview
This repository explores two foundational **deep generative models** —  
a **Variational Autoencoder (VAE)** and a **Deep Convolutional GAN (DCGAN)** —  
trained on the **MedMNIST BloodMNIST** dataset to generate synthetic medical images.

The goal of this project is to understand how **probabilistic (VAE)** and **adversarial (GAN)** learning frameworks differ in:
- Representation learning  
- Image reconstruction quality  
- Training dynamics and stability  

Both models were implemented **from scratch** using **PyTorch**, without using pre-built architectures.

---

## ⚙️ Models

### 🔹 Variational Autoencoder (VAE)
**Objective:**  
Learn a latent representation of images and reconstruct them by minimizing **reconstruction loss + KL-divergence**.

**Architecture:**
- Encoder: 2 convolutional layers → linear layers for μ and σ  
- Latent Sampling: Reparameterization trick (μ + σ * ε)  
- Decoder: Linear → transposed convolutions to reconstruct the image  
- Activation: ReLU in hidden layers, Sigmoid in output  

**Key Insight:**  
VAEs produce *blurry but semantically meaningful* images due to their probabilistic nature.

### 🔹 Deep Convolutional GAN (DCGAN)
**Objective:**  
Generate realistic images using a **minimax adversarial game** between:
- **Generator (G):** Learns to synthesize fake images  
- **Discriminator (D):** Learns to distinguish real from fake images  

**Architecture:**
- **Generator:** ConvTranspose2d + BatchNorm + ReLU layers  
- **Discriminator:** Conv2d + BatchNorm + LeakyReLU layers  
- **Loss:** Binary Cross-Entropy (BCE) for both G and D  
- **Optimization:** Adam (β₁ = 0.5, β₂ = 0.999)

**Key Insight:**  
Training stabilizes when D(G(z)) ≈ 0.5 — meaning the discriminator can no longer distinguish real vs. fake.

---

## 🧩 Dataset
- **Dataset:** [MedMNIST BloodMNIST](https://medmnist.com/)
- **Images:** 3 × 64 × 64 RGB blood cell images  
- **Classes:** 8 blood cell types  
- **Usage:** Only training split used (GANs don’t require validation/test splits)

---

## 🧰 Tech Stack
| Component | Tool |
|------------|------|
| Language | Python |
| Framework | PyTorch |
| Visualization | Matplotlib |
| Dataset | MedMNIST |
| Hardware | CUDA GPU / CPU fallback |

---

## ⚙️ Setup

### Clone the repository
```bash
git clone https://github.com/<your-username>/vae-gan-bloodmnist.git
cd vae-gan-bloodmnist
```
## 🚀 Training
```bash
# Train the VAE
python train_vae.py

# Train the DCGAN
python train_gan.py
```
---

## 🔍 Key Learnings
- Built VAE and DCGAN architectures manually in PyTorch
- Learned KL-divergence regularization and reparameterization trick
- Understood adversarial loss dynamics between Generator and Discriminator
- Applied weight initialization, normalization, and tuning for stable GAN training
- Compared latent-space behavior between probabilistic and adversarial models

---

## 📚 Future Work
- Implement WGAN-GP for improved convergence
- Train Conditional GANs for class-specific generation
- Perform latent-space arithmetic and visualization
