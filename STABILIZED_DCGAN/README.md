# Stabilized DCGAN — Photonic Crystal

A **Deep Convolutional Generative Adversarial Network (DCGAN)** trained on **34 grayscale photonic crystal images** at a resolution of **128×128** using **PyTorch**. This represents **Stage 3** in the architecture progression and is the first model to achieve stable training while successfully learning the fundamental topology of photonic crystal structures from the complete dataset.

---

## Project Motivation

The previous implementation (**DCGAN_STAGE_1**) closely followed the original DCGAN paper by training on **22 images** at a **20×20 resolution**. Although this approach reduced the severity of mode collapse compared to the Vanilla GAN, it still produced blurry outputs with limited structural diversity. More importantly, the low image resolution was insufficient to preserve the intricate geometric characteristics of photonic crystal structures.

To overcome these limitations, this stage abandons the original paper configuration and focuses on training a significantly more stable DCGAN using the **entire 34-image dataset** at a substantially higher resolution of **128×128**, allowing the model to capture meaningful spatial information.

---

## Improvements over DCGAN Stage 1

| Modification                                                                | Purpose                                                                                                                                   |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Image Resolution:** 20×20 → 128×128                                       | Enables the model to preserve the geometric details of photonic crystal structures.                                                       |
| **Training Dataset:** 22 → 34 Images                                        | Utilizes the complete dataset to improve generalization.                                                                                  |
| **Spectral Normalization** applied to every discriminator convolution layer | Restricts the discriminator's Lipschitz constant, preventing it from overpowering the generator during training.                          |
| **Minibatch Standard Deviation Layer**                                      | Encourages output diversity by allowing the discriminator to evaluate variation across generated samples, thereby reducing mode collapse. |
| **DCGAN Weight Initialization** (Normal distribution, mean = 0, std = 0.02) | Improves training stability during the initial optimization stages.                                                                       |
| **Label Smoothing** (Real labels = 0.9)                                     | Prevents the discriminator from becoming excessively confident on real samples, leading to more balanced adversarial learning.            |
| **BCEWithLogitsLoss**                                                       | Provides a numerically stable implementation by combining the sigmoid activation and binary cross-entropy loss into a single operation.   |

---

## Network Architecture

### Generator

**Input:** Random latent vector (`NOISE_DIM = 100`)

```text
ConvTranspose2d : 100 → 512 | 4×4 | Stride 1 | BatchNorm | ReLU
ConvTranspose2d : 512 → 256 | 4×4 | Stride 2 | BatchNorm | ReLU
ConvTranspose2d : 256 → 128 | 4×4 | Stride 2 | BatchNorm | ReLU
ConvTranspose2d : 128 →  64 | 4×4 | Stride 2 | BatchNorm | ReLU
ConvTranspose2d :  64 →   1 | 4×4 | Stride 2 | Tanh
```

**Output:** `(1, 128, 128)`

---

### Discriminator

**Input:** `(1, 128, 128)`

```text
SpectralNorm(Conv2d) :   1 →  32 | 4×4 | Stride 2 | LeakyReLU(0.2)
SpectralNorm(Conv2d) :  32 →  64 | 4×4 | Stride 2 | LeakyReLU(0.2)
SpectralNorm(Conv2d) :  64 → 128 | 4×4 | Stride 2 | LeakyReLU(0.2)
SpectralNorm(Conv2d) : 128 → 256 | 4×4 | Stride 2 | LeakyReLU(0.2)
MinibatchStdDev
SpectralNorm(Conv2d) : 257 →   1 | 4×4 | Stride 1
```

**Output:** Single scalar prediction.

---

## Training Configuration

| Hyperparameter                           | Value                           |
| ---------------------------------------- | ------------------------------- |
| Image Resolution                         | 128 × 128                       |
| Channels                                 | 1 (Grayscale)                   |
| Noise Dimension                          | 100                             |
| Batch Size                               | 8                               |
| Training Epochs                          | 300                             |
| Learning Rate                            | 2 × 10⁻⁴                        |
| Feature Maps (Generator / Discriminator) | 32                              |
| Optimizer                                | Adam (`β₁ = 0.5`, `β₂ = 0.999`) |
| Loss Function                            | BCEWithLogitsLoss               |
| Label Smoothing                          | Real Labels = 0.9               |

---

## Experimental Results

Training remained stable throughout the complete **300 epochs**, with both the Generator and Discriminator losses converging toward the expected **Nash Equilibrium (≈0.693)** and maintaining that equilibrium consistently. Unlike the previous architectures, **no mode collapse** was observed during training.

The generator successfully learned several important structural characteristics of the photonic crystal dataset, including:

* Periodic silicon-air hole lattice.
* Diagonal waveguide defect channel.
* Approximate silicon-air intensity distribution.

Despite these improvements, the generated images still exhibit relatively **low contrast** and **blurred hole boundaries** when compared with real Lumerical FDTD simulation images. Although the overall topology is preserved, the model struggles to reproduce fine structural details, indicating the practical performance limit of a conventional DCGAN when trained on only **34 samples**.

These observations suggest that while the architecture successfully captures global structural information, it lacks an effective mechanism for accurately matching the underlying data distribution.

This limitation directly motivated the transition to a **Wasserstein GAN with Gradient Penalty (WGAN-GP)**.

---

## Generated Output

*(Insert generated sample images here.)*

---

## Motivation for the Next Architecture

The **BCEWithLogitsLoss** objective treats the discriminator as a binary classifier, encouraging predictions that approach either **0** or **1**. As training progresses and the discriminator becomes increasingly confident, the generator receives progressively weaker gradients, making optimization difficult and reducing image quality.

To overcome this limitation, the subsequent architecture adopts the **Wasserstein Distance**, which measures the distance between the real and generated data distributions rather than performing binary classification. This formulation provides smoother gradients, improves optimization stability, and enables more effective learning under extremely limited data conditions.

Consequently, the next stage of this research transitions to **WGAN-GP**, where the Gradient Penalty further enforces the Lipschitz constraint required for stable Wasserstein optimization.

➡ **See `V0_WGAN-GP/` for the next stage of the architecture progression.**

