from autograd import grad, hessian
import autograd.numpy as anp
from matplotlib.animation import FuncAnimation
import numpy as np

def himmelblau(x):
    return (x[0]**2 + x[1] - 11.0)**2 + (x[0] + x[1]**2 - 7.0)**2

objective = himmelblau
objective_grad = grad(objective)


def backtracking_line_search(f, grad, x, p, alpha=1.0, rho=0.5, c=1e-4):
    fx = f(x)
    grad_fx = grad(x)

    while f(x + alpha * p) > fx + c * alpha * anp.dot(grad_fx, p):
        alpha *= rho  # shrink alpha
    return x + alpha * p

class Levenberg_Marquardt:
    def __init__(self, f, lamda = 1e-2, y_acc = 0.1, y_rej = 10.0, eps = 1e-10):
        self.lamda = lamda
        self.y_acc = y_acc
        self.y_rej = y_rej
        self.eps = eps
        self.f = f
    def step(self, x, grad):
        H = hessian(self.f)(x)
        D = anp.diag(H)
        D_clipped = anp.maximum(D, self.eps)
        H_reg = H + self.lamda * anp.diag(D_clipped)

        H = H + self.lamda * anp.diag(D)
        p = anp.linalg.solve(H, grad)
        x_new = x + p

        if self.f(x_new) < self.f(x):
            self.lamda *= self.y_acc 
            return x_new
        self.lamda *= self.y_rej
        return x


class L_BFGS:
    def __init__(self, f, grad, line_search_fn = backtracking_line_search, memory=10):
        self.grad = grad
        self.line_search_fn = line_search_fn
        self.memory = memory
        self.ds = [] 
        self.ys = [] 
        self.qs = [] 
        self.f = f
    def step(self, x, grad):
        g = self.grad(x)
        m = len(self.ds)

        if m > 0:
            q = g.copy()

            for i in reversed(range(m)):
                self.qs[i] = q.copy()
                rho_i = anp.dot(self.ys[i], self.ds[i])
                if rho_i == 0.0:
                    continue
                alpha_i = anp.dot(self.ds[i], q) / rho_i

                alpha_i = anp.dot(self.ds[i], q) / rho_i
                q = q - alpha_i * self.ys[i]

            ys_last = self.ys[-1]
            ds_last = self.ds[-1]
            denom = anp.dot(ys_last, ys_last)
            if abs(denom) < 1e-8:
                denom = 1e-8
            z = (ys_last * ds_last * q) / denom 

            for i in range(m):
                rho_i = anp.dot(self.ys[i], self.ds[i])
                if rho_i == 0.0:
                    continue
                alpha_i = anp.dot(self.ds[i], q) / rho_i

                beta = (anp.dot(self.ds[i], self.qs[i]) - anp.dot(self.ys[i], z)) / rho_i
                z = z + self.ds[i] * beta

            p = -z
        else:
            p = -g 

        x_new = self.line_search_fn(self.f, self.grad, x, p)
        g_new = self.grad(x_new)

        s_k = x_new - x
        y_k = g_new - g

        self.ds.append(s_k)
        self.ys.append(y_k)
        self.qs.append(anp.zeros_like(x))

        if len(self.ds) > self.memory:
            self.ds.pop(0)
            self.ys.pop(0)
            self.qs.pop(0)

        return x_new
    



class AMSGrad_noisy:
    def __init__(self, lr=0.01, dim=2, beta1=0.9, beta2=0.999, eps=1e-8, noise_std=0.3):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.noise_std = noise_std
        self.m = np.zeros(dim)
        self.v = np.zeros(dim)
        self.v_hat = np.zeros(dim)
        self.t = 0
    def step(self, x, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)
        self.v_hat = np.maximum(self.v_hat, self.v)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat_corr = self.v_hat / (1 - self.beta2**self.t)
        noise = np.random.normal(0, self.noise_std, size=x.shape)
        return x - self.lr * m_hat / (np.sqrt(v_hat_corr) + self.eps) + noise
    def reset(self):
        self.m[:] = 0
        self.v[:] = 0
        self.v_hat[:] = 0
        self.t = 0

class PolakRibiereCG_noisy:
    def __init__(self, dim=2, alpha_init=1.0, rho=0.5, c=1e-4, noise_std=0.25):
        self.alpha_init = alpha_init
        self.rho = rho
        self.c = c
        self.noise_std = noise_std
        self.prev_grad = None
        self.prev_dir = None
        self.st = 0
    def step(self, x, grad):
        self.st += 1
        if self.prev_grad is None:
            d = -grad
        else:
            y = grad - self.prev_grad
            beta = max(0.0, np.dot(grad, y) / (np.dot(self.prev_grad, self.prev_grad) + 1e-8))
            d = -grad + beta * self.prev_dir

        alpha = self.alpha_init
        fx = objective(x)
        while objective(x + alpha * d) > fx + self.c * alpha * np.dot(grad, d):
            alpha *= self.rho

        self.prev_grad = grad
        self.prev_dir = d
        noise = np.random.normal(0, self.noise_std, size=x.shape)
        x_new = x + alpha * d + noise
        if not self.st % 20:
            self.reset()
        return x_new
    def reset(self):
        self.prev_grad = None
        self.prev_dir = None

# --- Optimizers ---
class MomentumGD_noisy:
    def __init__(self, lr=0.0001, momentum=0.9, dim=2, base_noise_std=0.3):
        self.lr = lr
        self.momentum = momentum
        self.v = np.zeros(dim)
        self.base_noise_std = base_noise_std
        self.t = 1
    def step(self, x, grad):
        self.v = self.momentum * self.v - self.lr * grad
        noise_scale = self.base_noise_std
        self.t += 1
        return x + self.v + np.random.normal(0, noise_scale, size=x.shape)
    def reset(self):
        self.v = np.zeros_like(self.v)
        self.t = 1

class MomentumGD:
    def __init__(self, lr=0.0001, momentum=0.9, dim=2):
        self.lr = lr
        self.momentum = momentum
        self.v = np.zeros(dim)
    def step(self, x, grad):
        self.v = self.momentum * self.v - self.lr * grad
        return x + self.v
    def reset(self):
        self.v = np.zeros_like(self.v)

class PolakRibiereCG:
    def __init__(self, dim=2, alpha_init=1.0, rho=0.5, c=1e-4):
        self.alpha_init = alpha_init
        self.rho = rho
        self.c = c
        self.prev_grad = None
        self.prev_dir = None
        self.st = 0
    def step(self, x, grad):
        self.st +=1
        if self.prev_grad is None:
            d = -grad
        else:
            y = grad - self.prev_grad
            beta = max(0.0, np.dot(grad, y) / (np.dot(self.prev_grad, self.prev_grad) + 1e-8))
            d = -grad + beta * self.prev_dir

        alpha = self.alpha_init
        fx = objective(x)
        while objective(x + alpha * d) > fx + self.c * alpha * np.dot(grad, d):
            alpha *= self.rho

        self.prev_grad = grad
        self.prev_dir = d
        x_new = x + alpha * d
        if not self.st % 20:
            self.reset()
        return x_new
    def reset(self):
        self.prev_grad = None
        self.prev_dir = None

class AMSGrad:
    def __init__(self, lr=0.01, dim=2, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(dim)
        self.v = np.zeros(dim)
        self.v_hat = np.zeros(dim)
        self.t = 0
    def step(self, x, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)
        self.v_hat = np.maximum(self.v_hat, self.v)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat_corr = self.v_hat / (1 - self.beta2**self.t)
        return x - self.lr * m_hat / (np.sqrt(v_hat_corr) + self.eps)
    def reset(self):
        self.m[:] = 0
        self.v[:] = 0
        self.v_hat[:] = 0
        self.t = 0
