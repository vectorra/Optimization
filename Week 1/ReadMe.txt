
This function has multiple global minima for testing optimizer robustness and noise handling.

---

Optimizers

| Optimizer                 | Variant       |
|--------------------------|---------------|
| Momentum Gradient Descent| clean / noisy |
| AMSGrad                  | clean / noisy |
| Polak-Ribiere CG         | clean / noisy |

Each optimizer implements a `.step()` and `.reset()` method. The noisy versions simulate stochastic behavior by adding Gaussian noise to the update.

