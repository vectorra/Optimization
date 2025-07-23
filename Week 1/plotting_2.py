import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from autograd import grad
import autograd.numpy as anp
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- Objective Function and Gradient ---
def himmelblau(x):
    return (x[0]**2 + x[1] - 11.0)**2 + (x[0] + x[1]**2 - 7.0)**2

objective = himmelblau
objective_grad = grad(objective)

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
    def step(self, x, grad):
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
        return x + alpha * d + noise
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
    def step(self, x, grad):
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
        return x + alpha * d
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

# --- Optimizers ---
optimizers = {
    'Momentum GD': MomentumGD(lr=0.001, momentum=0.9),
    'Polak-Ribiere CG': PolakRibiereCG(dim=2),
    'AMSGrad': AMSGrad(lr=0.01),
    'Momentum (noisy)': MomentumGD_noisy(lr=0.0001, momentum=0.9),
    'AMSGrad (noisy)': AMSGrad_noisy(lr=0.01),
    'Polak-Ribiere CG (noisy)': PolakRibiereCG_noisy(dim=2)
}


# --- Visualization Setup ---
start_point = np.array([-1.9, 1.9])
max_iters = 250

xlist = np.linspace(-5, 5, 400)
ylist = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(xlist, ylist)
Z = np.vectorize(lambda x, y: objective(anp.array([x, y])))(X, Y)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, edgecolor='none')
ax.contour(X, Y, Z, zdir='z', offset=np.min(Z)-10, cmap='viridis')

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(np.min(Z)-10, np.max(Z)+10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.set_title('Optimization Trajectories on Himmelblau Function')

colors = ['crimson', 'dodgerblue', 'darkorange', 'green', 'magenta', 'teal', 'black']
markers = ['o', '^', 's', 'D', '*', 'P', 'x']
linestyles = ['-', '--', ':', '-.', '-', '--', ':']
marker_sizes = [8, 9, 10, 8, 9, 8, 8]
line_widths = [3, 3, 2.5, 2.5, 2.5, 2.5, 2.5]

trajectories = []
points = []
lines = []

trail_len = 15

for i, (name, opt) in enumerate(optimizers.items()):
    pos = start_point.copy()
    trajectories.append([pos.copy()])
    p, = ax.plot([pos[0]], [pos[1]], [objective(pos)],
                 marker=markers[i], color=colors[i],
                 label=name, markersize=marker_sizes[i], linestyle='None')
    points.append(p)

    l, = ax.plot([pos[0]], [pos[1]], [objective(pos)],
                 color=colors[i], lw=line_widths[i], linestyle=linestyles[i])
    lines.append(l)

ax.legend(fontsize=12)

# --- Animation ---
def update(frame):
    for i, (name, opt) in enumerate(optimizers.items()):
        pos = trajectories[i][-1]
        grad_val = objective_grad(pos)
        pos_new = opt.step(pos, grad_val)
        trajectories[i].append(pos_new)

        trail_points = trajectories[i][-trail_len:]
        data = np.array(trail_points)
        zdata = np.array([objective(p) for p in data])

        points[i].set_data([data[-1, 0]], [data[-1, 1]])
        points[i].set_3d_properties([zdata[-1]])
        lines[i].set_data(data[:, 0], data[:, 1])
        lines[i].set_3d_properties(zdata)

    return points + lines

anim = FuncAnimation(fig, update, frames=max_iters, interval=50, blit=False)

# --- Reset on Key Press ---
def reset_animation(event):
    if event.key == 'r':
        print("Restarting animation")
        for traj in trajectories:
            traj.clear()
            traj.append(start_point.copy())
        for opt in optimizers.values():
            if hasattr(opt, 'reset'):
                opt.reset()
        anim.frame_seq = anim.new_frame_seq()

fig.canvas.mpl_connect('key_press_event', reset_animation)

plt.show()
