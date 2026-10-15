# Gaussian Variational Autoencoder (VAE) with Full Covariance via Cholesky Decomposition

A PyTorch implementation of a **Gaussian Variational Autoencoder (VAE)** trained on the MNIST dataset of handwritten digits. 

Unlike standard VAE architectures that assume an isotropic or diagonal covariance matrix (mean-field approximation), this model parameterizes a **full covariance matrix** for the approximate posterior $q_\theta(z \mid x)$ through its lower-triangular **Cholesky factor** $L(x)$.

---

## 1. Probabilistic Formulation & Mathematical Assumptions

A Variational Autoencoder defines a directed latent-variable generative model:

```
          p(z) = N(0, I)
             \
              \  Decoder p_phi(x | z)
               v
               x
               ^
              /  Encoder q_theta(z | x)
             /
          Input x
```

### 1.1 Prior Distribution over Latent Variables
The latent representations $z \in \mathbb{R}^d$ are assumed to follow a standard multivariate normal distribution:
$$p(z) = \mathcal{N}\left(z; \mathbf{0}, \mathbf{I}\right)$$
- **Assumption**: Prior latent dimensions are mutually independent, centered at zero with unit variance.

---

### 1.2 Approximate Posterior (Encoder)
The true posterior $p(z \mid x)$ is intractable. We approximate it using a variational Gaussian family $q_\theta(z \mid x)$:
$$q_\theta(z \mid x) = \mathcal{N}\left(z; \mu_\theta(x), \Sigma_\theta(x)\right)$$

- **Full Covariance Assumption**: Rather than constraining $\Sigma_\theta(x)$ to be diagonal, we model pairwise correlations between latent dimensions.
- **Cholesky Decomposition**:
  Any symmetric, positive-definite covariance matrix $\Sigma \in \mathbb{R}^{d \times d}$ can be factored uniquely as:
  $$\Sigma_\theta(x) = L(x) L(x)^T$$
  where $L(x)$ is a lower-triangular matrix with strictly positive diagonal entries ($L_{ii} > 0$).
- **Number of Estimated Parameters**:
  A lower-triangular matrix of size $d \times d$ contains $\frac{d(d+1)}{2}$ independent entries. The encoder network therefore predicts:
  1. Mean vector: $\mu_\theta(x) \in \mathbb{R}^d$
  2. Lower-triangular entries: $l(x) \in \mathbb{R}^{\frac{d(d+1)}{2}}$
- **Ensuring Positive Definiteness**:
  The diagonal entries are passed through the exponential function:
  $$L_{ii} = \exp(l_{ii})$$
  which guarantees $L_{ii} > 0$, ensuring $\Sigma_\theta(x)$ is invertible and strictly positive definite.

---

### 1.3 Likelihood / Generative Model (Decoder)
The decoder models the conditional likelihood of the flattened image $x \in \mathbb{R}^D$ ($D = 28 \times 28 = 784$):
$$p_\phi(x \mid z) = \mathcal{N}\left(x; \mu_\phi(z), \sigma^2 \mathbf{I}_D\right)$$

- **Assumption**: Continuous multivariate Gaussian with isotropic diagonal covariance scaled by fixed variance $\sigma^2$.
- **Decoder Variance**: $\sigma$ is treated as a fixed hyperparameter (`decoder_sigma = 1.0` by default).

---

## 2. Loss Function: Negative ELBO Derivation

The model is trained by maximizing the **Evidence Lower Bound (ELBO)**, which is equivalent to minimizing the negative ELBO per batch:

$$\mathcal{L}(\theta, \phi; x) = -\mathbb{E}_{q_\theta(z \mid x)}\left[\log p_\phi(x \mid z)\right] + D_{\text{KL}}\left(q_\theta(z \mid x) \,\parallel\, p(z)\right)$$

### 2.1 Reconstruction Loss
Under the isotropic Gaussian likelihood $p_\phi(x \mid z) = \mathcal{N}(x; \mu_\phi(z), \sigma^2 \mathbf{I}_D)$:

$$-\log p_\phi(x \mid z) = \frac{D}{2}\log(2\pi) + D\log(\sigma) + \frac{1}{2\sigma^2} \|x - \mu_\phi(z)\|^2$$

Dropping constant terms with respect to network parameters $\theta$ and $\phi$, the per-sample reconstruction error corresponds to the scaled squared Euclidean distance:

$$  \mathcal{L}_{\text{recon}}(x, \mu_\phi(z)) = \frac{1}{2\sigma^2} \sum_{k=1}^D (x_k - \mu_{\phi, k}(z))^2$$

---

### 2.2 Analytical Kullback-Leibler Divergence
For two multivariate Gaussians $q(z) = \mathcal{N}(\mu, \Sigma)$ and $p(z) = \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$, the KL divergence has the closed form:

$$D_{\text{KL}}\left(q(z) \,\parallel\, p(z)\right) = \frac{1}{2} \left[ \text{Tr}(\Sigma) + \|\mu\|^2 - d - \log \det(\Sigma) \right]$$

Exploiting the Cholesky factor $\Sigma = L L^T$:
1. **Trace Term**:
   $$\text{Tr}(\Sigma) = \text{Tr}(L L^T) = \sum\limits_{i=1}^d \sum\limits_{k=1}^i L_{ik}^2 = \|L\|_F^2$$
2. **Log-Determinant Term**:
   $$\det(\Sigma) = \det(L L^T) = (\det(L))^2 = \left(\prod\limits_{i=1}^d L_{ii}\right)^2$$

   $$\log \det(\Sigma) = 2 \sum_{i=1}^d \log L_{ii}$$

Substituting both into the KL formula gives:
$$D_{\text{KL}} = \frac{1}{2} \left[ \sum\limits_{i=1}^d \sum\limits_{k=1}^i L_{ik}^2 + \sum\limits_{j=1}^d \mu_j^2 - d - 2 \sum\limits_{i=1}^d \log (L_{ii}) \right]$$

---

### 2.3 Multivariate Reparameterization Trick
To propagate gradients through latent sampling, the stochasticity is isolated into an auxiliary standard normal variable:

$$\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$$

$$z = \mu_\theta(x) + L(x) \epsilon$$

Since $L(x) \epsilon \sim \mathcal{N}(\mathbf{0}, L L^T) = \mathcal{N}(\mathbf{0}, \Sigma_\theta(x))$, $z$ is distributed exactly according to $q_\theta(z \mid x)$.

---

### 2.4 Batch-Averaged Optimization Objective
For a mini-batch of $B$ observations $\{x^{(b)}\}_{b=1}^B$, the loss minimized by the optimizer is the empirical mean:

$$\mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{b=1}^B \left( \mathcal{L}_{\text{recon}}\left(x^{(b)}, \mu_\phi(z^{(b)})\right) + D_{\text{KL}}\left(q_\theta\left(z \mid x^{(b)}\right) \,\parallel\, p(z)\right) \right)$$

---

## 3. Network Architecture & Layer Specifications

### 3.1 Encoder Architecture
The encoder maps a single-channel $28 \times 28$ image into the parameters of the variational distribution in a $d$-dimensional latent space ($d = 16$):

```
Input: [B, 1, 28, 28]
  │
  ▼
Flatten ───────────────► Linear(784, 196) ──► ReLU
                              │
                              ▼
                         Linear(196, 49) ───► ReLU
                         /            \
                        /              \
                       ▼                ▼
             Linear(49, 16)       Linear(49, 16*(16+1)/2)
                   │                      │
             Mean vector mu           L_vector
              [B, 16]                 [B, 136]
```

- **Input Dimension**: $28 \times 28 = 784$
- **Hidden Layer 1 (`lin1`)**: `nn.Linear(784, 196)` + `F.relu`
- **Hidden Layer 2 (`lin2`)**: `nn.Linear(196, 49)` + `F.relu`
- **Latent Mean Head (`mu`)**: `nn.Linear(49, 16)` $\to \mu_\theta(x) \in \mathbb{R}^{B \times 16}$
- **Cholesky Factor Head (`L_matrix`)**: `nn.Linear(49, 136)` $\to \frac{16 \times 17}{2} = 136$ values per sample

---

### 3.2 Cholesky Factor Assembly (`build_cholesky_factor`)
1. Allocates a zeros tensor of shape `[B, 16, 16]`.
2. Fills the lower-triangular entries using precomputed lower-triangular indices `torch.tril_indices(16, 16)`.
3. Replaces the diagonal entries with $\exp(\text{diag}(L))$ via `diagonal_scatter`.
4. Returns the batch of lower-triangular matrices $L \in \mathbb{R}^{B \times 16 \times 16}$.

---

### 3.3 Decoder Architecture
The decoder maps a latent vector $z \in \mathbb{R}^{16}$ back to the reconstructed image space $\mathbb{R}^{784}$:

```
Latent Sample z: [B, 16]
  │
  ▼
Linear(16, 49) ───► ReLU
  │
  ▼
Linear(49, 196) ──► ReLU
  │
  ▼
Linear(196, 784)
  │
  ▼
mu_z: [B, 784] ───► Reshape to [B, 1, 28, 28]
```

- **Hidden Layer 1 (`lin1`)**: `nn.Linear(16, 49)` + `F.relu`
- **Hidden Layer 2 (`lin2`)**: `nn.Linear(49, 196)` + `F.relu`
- **Output Layer (`mu`)**: `nn.Linear(196, 784)` without activation, representing the Gaussian mean image $\mu_\phi(z)$.

---

## 4. Training Procedure

- **Dataset**: MNIST (60,000 training images, batch size = 64)
- **Image Preprocessing**: Scaled to float in range $[0, 1]$ and normalized to $[-1, 1]$ with mean $0.5$ and std $0.5$.
- **Optimizer**: Adam ($\text{lr} = 10^{-3}$)
- **Epochs**: 10
- **Loss Computation**: Per-sample reconstruction MSE and analytical KL divergence, averaged across each mini-batch.
