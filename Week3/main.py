import numpy as np
from InteriorPoint import InteriorPointSolver


N = 24
dt = 1
C = 5
R = 2
T_init = 24
T_min = 20
T_max = 24
u_max = 5
delta = 1e-9
gamma = 1.1
rho_init = 1.0
gap_min = 0.5

k = np.arange(1, N + 1)
T_out = 23 + 10 * np.exp(-0.5 * ((k - 12.5) / 6) ** 2)
cost = np.array([0.1 if (i >= 22 or i < 7) else 0.3 for i in range(N)])

n_vars = 2 * N
A_eq = np.zeros((N - 1, n_vars))
b_eq = np.zeros(N - 1)

class CoolingProblem:
    def __init__(self, N, dt, C, R, T_init, T_min, T_max, u_max, cost, T_out):
        self.N = N
        self.n_vars = 2 * N
        self.cost = cost
        self.T_out = T_out
        self.T_init = T_init

        self.A_eq, self.b_eq = self._make_dynamics_constraints(N, dt, C, R, T_out, T_init)
        self.G, self.h_vec = self._make_inequality_constraints(N, T_min, T_max, u_max)

    def _make_dynamics_constraints(self, N, dt, C, R, T_out, T_init):
        A_eq = np.zeros((N - 1, 2 * N))
        b_eq = np.zeros(N - 1)
        for i in range(N - 1):
            A_eq[i, i] = -1 + dt / (C * R)
            A_eq[i, i + 1] = 1
            A_eq[i, N + i] = dt / C
            b_eq[i] = dt / (C * R) * T_out[i]

        A_init = np.zeros((1, 2 * N)); A_init[0, 0] = 1
        b_init = np.array([T_init])
        return np.vstack([A_init, A_eq]), np.concatenate([b_init, b_eq])

    def _make_inequality_constraints(self, N, T_min, T_max, u_max):
        G, h = [], []
        for i in range(N):
            g = np.zeros(2 * N); g[i] = 1; G.append(g); h.append(T_max)
            g = np.zeros(2 * N); g[i] = -1; G.append(g); h.append(-T_min)
        for i in range(N):
            g = np.zeros(2 * N); g[N + i] = 1; G.append(g); h.append(u_max)
            g = np.zeros(2 * N); g[N + i] = -1; G.append(g); h.append(0)
        return np.array(G), np.array(h)


def f(x, cost, N): return cost @ x[N:]
def grad_f(x, cost, N): g = np.zeros_like(x); g[N:] = cost; return g
def hess_f(x): return np.zeros((len(x), len(x)))


if __name__ == "__main__":
    problem = CoolingProblem(N, dt, C, R, T_init, T_min, T_max, u_max, cost, T_out)
    solver = InteriorPointSolver(problem)
    solved = solver.solve()