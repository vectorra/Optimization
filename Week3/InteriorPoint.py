import numpy as np
from numpy.linalg import lstsq


class InteriorPointSolver:
    def __init__(self, problem, delta=1e-9, gamma=1.1, rho_init=1.0, gap_min=0.5):
        self.p = problem
        self.delta = delta
        self.gamma = gamma
        self.rho = rho_init
        self.gap_min = gap_min

    def f(self, x):
        return self.p.cost @ x[self.p.N:]

    def grad_f(self, x):
        g = np.zeros(self.p.n_vars)
        g[self.p.N:] = self.p.cost
        return g

    def hess_f(self, x):
        return np.zeros((self.p.n_vars, self.p.n_vars))

    def gs(self, x):
        return self.p.G @ x - self.p.h_vec

    def grad_gs(self, x):
        return self.p.G

    def hess_gs(self, x):
        return [np.zeros((self.p.n_vars, self.p.n_vars)) for _ in range(len(self.p.h_vec))]

    def grad_lagrangian(self, x, mu, lamb):
        return self.grad_f(x) + self.grad_gs(x).T @ mu + self.p.A_eq.T @ lamb

    def hess_lagrangian(self, x, mu):
        H = self.hess_f(x)
        for i, Hgi in enumerate(self.hess_gs(x)):
            H += mu[i] * Hgi
        return H

    def solve(self):
        x = np.concatenate([np.ones(self.p.N) * 22.0, np.ones(self.p.N) * 1.0])
        g0 = self.gs(x)
        g0 = np.where(g0 > -1e-6, -1e-6, g0)
        mu = -1.0 / g0
        lamb = np.zeros(self.p.A_eq.shape[0])

        history = []
        trajectory = []

        for step in range(150):
            g = self.gs(x)
            Jg = self.grad_gs(x)
            gradL = self.grad_lagrangian(x, mu, lamb)
            r_dual = gradL
            r_cent = -np.diag(mu) @ g - (1 / self.rho) * np.ones_like(g)
            r_pri = self.p.A_eq @ x - self.p.b_eq
            gap = -g @ mu + r_pri @ lamb
            history.append((step, gap))

            H = self.hess_lagrangian(x, mu) + self.delta * np.eye(self.p.n_vars)

            KKT_top = np.hstack([H, Jg.T, self.p.A_eq.T])
            KKT_mid = np.hstack([-np.diag(mu) @ Jg, -np.diag(g), np.zeros((len(g), self.p.A_eq.shape[0]))])
            KKT_bot = np.hstack([self.p.A_eq, np.zeros((self.p.A_eq.shape[0], len(g) + self.p.A_eq.shape[0]))])
            KKT = np.vstack([KKT_top, KKT_mid, KKT_bot])

            rhs = -np.concatenate([r_dual, r_cent, r_pri])
            d = lstsq(KKT, rhs, rcond=None)[0]

            dx = d[:self.p.n_vars]
            dmu = d[self.p.n_vars:self.p.n_vars + len(g)]
            dlamb = d[self.p.n_vars + len(g):]

            alpha = 1.0
            while np.any(self.gs(x + alpha * dx) >= 0) or np.any(mu + alpha * dmu <= 0):
                alpha *= 0.9
                if alpha < 1e-10: break

            x += alpha * dx
            mu += alpha * dmu
            lamb += alpha * dlamb
            self.rho *= self.gamma

            print(f"step: {step}, Gap: {gap:.4f}, Alpha: {alpha}, Rho: {self.rho:.4f}")
            trajectory.append((step, x[:self.p.N].copy(), x[self.p.N:].copy(), gap))

            # if gap < self.gap_min:
            #     break

        return x, history, trajectory
